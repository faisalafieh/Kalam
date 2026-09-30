"""
Download a slice of the AMI Meeting Corpus test set and build an eval manifest.

AMI is the standard public benchmark for meeting transcription and diarization:
real, unscripted 3-5 person meetings with hand-made word-level transcripts.
Licence: CC BY 4.0 (https://groups.inf.ed.ac.uk/ami/corpus/license.shtml).

Uses the official test split from the Full-corpus-ASR partition, with the
diarization references from BUTSpeechFIT/AMI-diarization-setup (the setup
pyannote itself reports its AMI numbers on).

    python eval/prepare_ami.py                      # 4 meetings, ~1.6 h audio
    python eval/prepare_ami.py --all                # all 16 test meetings, ~9 h
    python eval/prepare_ami.py --mic distant        # one far-field room mic
    python eval/prepare_ami.py --split dev          # development meetings, for tuning

Tune settings on --split dev, then report on the test split once. Choosing
settings by looking at test scores overstates how well the system generalises.

--mic headset   Mix-Headset: every participant's headset mixed together.
                Clean audio, and the condition pyannote publishes DER for.
--mic distant   Array1-01: a single microphone on the table. Closest to a
                laptop or phone recording a room, and much harder.
"""

import argparse
import io
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

AMI = "https://groups.inf.ed.ac.uk/ami"
ANNOTATIONS = f"{AMI}/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AUDIO = AMI + "/AMICorpusMirror/amicorpus/{m}/audio/{m}.{suffix}.wav"
SETUP = "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main"

MIC_SUFFIX = {"headset": "Mix-Headset", "distant": "Array1-01"}

# One meeting from each series (different rooms and groups) in each split.
DEFAULT_MEETINGS = {
    "test": ["ES2004a", "IS1009a", "TS3003a", "EN2002a"],
    "dev": ["ES2011a", "IS1008a", "TS3004a", "IB4001"],
}

UTTERANCE_GAP = 1.0  # seconds of silence that splits one speaker's words into utterances


def fetch(url: str) -> bytes:
    print(f"  GET {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.read()
    except Exception as e:
        sys.exit(f"Download failed for {url}: {e}")


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(fetch(url))
    tmp.rename(dest)


def words_from_xml(xml_bytes: bytes):
    """Yield (start, end, word) for real words, skipping punctuation tokens."""
    root = ET.fromstring(xml_bytes)
    last = 0.0
    for el in root:
        if not el.tag.endswith("w") or el.get("punc") == "true":
            continue
        text = (el.text or "").strip()
        if not text:
            continue
        start = float(el.get("starttime", last))
        end = float(el.get("endtime", start))
        last = end
        yield start, end, text


def build_reference(zf: zipfile.ZipFile, meeting: str) -> list[dict]:
    """Chronological utterances [{speaker, start, end, text}] for one meeting."""
    pattern = re.compile(rf"words/{meeting}\.([A-Z])\.words\.xml$")
    utts = []
    for name in zf.namelist():
        m = pattern.search(name)
        if not m:
            continue
        speaker = m.group(1)
        cur = None
        for start, end, word in words_from_xml(zf.read(name)):
            if cur and start - cur["end"] <= UTTERANCE_GAP:
                cur["text"] += " " + word
                cur["end"] = end
            else:
                if cur:
                    utts.append(cur)
                cur = {"speaker": speaker, "start": start, "end": end, "text": word}
        if cur:
            utts.append(cur)
    if not utts:
        sys.exit(f"No word annotations found for {meeting}")
    utts.sort(key=lambda u: u["start"])
    return utts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mic", choices=MIC_SUFFIX, default="headset")
    ap.add_argument("--split", choices=DEFAULT_MEETINGS, default="test")
    ap.add_argument("--all", action="store_true", help="every meeting in the split (16 test / 18 dev)")
    ap.add_argument("--meetings", nargs="+", help="explicit meeting IDs")
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "ami"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.meetings:
        meetings = args.meetings
    elif args.all:
        meetings = fetch(f"{SETUP}/lists/{args.split}.meetings.txt").decode().split()
    else:
        meetings = DEFAULT_MEETINGS[args.split]

    ann_zip = out / "ami_public_manual_1.6.2.zip"
    print("Annotations")
    download(ANNOTATIONS, ann_zip)
    zf = zipfile.ZipFile(io.BytesIO(ann_zip.read_bytes()))

    suffix = MIC_SUFFIX[args.mic]
    prefix = "manifest" if args.split == "test" else f"manifest_{args.split}"
    manifest = out / f"{prefix}_{args.mic}.jsonl"
    with open(manifest, "w") as mf:
        for m in meetings:
            print(m)
            audio = out / "audio" / f"{m}.{suffix}.wav"
            rttm = out / "rttm" / f"{m}.rttm"
            uem = out / "uem" / f"{m}.uem"
            ref = out / "ref" / f"{m}.json"

            download(AUDIO.format(m=m, suffix=suffix), audio)
            download(f"{SETUP}/only_words/rttms/{args.split}/{m}.rttm", rttm)
            download(f"{SETUP}/uems/{args.split}/{m}.uem", uem)
            if not ref.exists():
                ref.parent.mkdir(parents=True, exist_ok=True)
                ref.write_text(json.dumps(build_reference(zf, m), indent=0))

            mf.write(json.dumps({
                "id": m,
                "audio": str(audio.relative_to(out)),
                "reference": str(ref.relative_to(out)),
                "rttm": str(rttm.relative_to(out)),
                "uem": str(uem.relative_to(out)),
            }) + "\n")

    print(f"\nManifest written: {manifest}")
    tag = f"ami-{args.mic}" if args.split == "test" else f"ami-{args.split}-{args.mic}"
    print(f"Next:  python eval/evaluate.py --manifest {manifest} --tag {tag}")


if __name__ == "__main__":
    main()
