"""
Scoring for Kalam: WER, cpWER and DER.

WER    Word error rate of the transcript, ignoring who spoke.
cpWER  Concatenated minimum-permutation WER (CHiME-6 / NOTSOFAR style).
       Each speaker's words are concatenated, hypothesis speakers are matched to
       reference speakers by the assignment that minimises total errors, and a
       word attributed to the wrong person counts as an error. This is the
       metric that answers "is the transcript right about who said what".
DER    Diarization error rate (missed speech + false alarm + speaker confusion),
       computed with pyannote.metrics, no forgiveness collar, overlap included.

All WER-family numbers are pooled: total errors / total reference words,
not an average of per-file percentages.
"""

from __future__ import annotations

from collections import defaultdict

import jiwer
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from whisper_normalizer.english import EnglishTextNormalizer

    _normalize = EnglishTextNormalizer()
except ImportError:  # fallback keeps the script usable, but numbers are stricter
    import re

    def _normalize(s: str) -> str:
        s = re.sub(r"[^\w\s']", " ", s.lower())
        return " ".join(s.split())


def normalize(text: str) -> str:
    """Whisper's English normaliser: lowercase, strip punctuation, spell out
    contractions, standardise numbers and spelling, drop fillers (um, uh)."""
    return " ".join(_normalize(text).split())


def _errors(ref: str, hyp: str) -> dict:
    """Edit counts between two already-normalised strings."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return {"sub": 0, "del": 0, "ins": len(hyp_words), "ref_words": 0}
    if not hyp_words:
        return {"sub": 0, "del": len(ref_words), "ins": 0, "ref_words": len(ref_words)}
    o = jiwer.process_words(ref, hyp)
    return {
        "sub": o.substitutions,
        "del": o.deletions,
        "ins": o.insertions,
        "ref_words": len(ref_words),
    }


def wer_counts(ref_utts: list[dict], hyp_segs: list[dict]) -> dict:
    """Speaker-agnostic WER. Both inputs are chronological lists of {"text": ...}."""
    ref = normalize(" ".join(u["text"] for u in ref_utts))
    hyp = normalize(" ".join(s["text"] for s in hyp_segs))
    return _errors(ref, hyp)


def _by_speaker(items: list[dict]) -> dict[str, str]:
    grouped = defaultdict(list)
    for it in items:
        grouped[it["speaker"]].append(it["text"])
    return {spk: normalize(" ".join(t)) for spk, t in grouped.items()}


def cpwer_counts(ref_utts: list[dict], hyp_segs: list[dict]) -> dict:
    """Speaker-attributed WER with the best one-to-one speaker mapping.

    Unmatched reference speakers are scored against an empty stream (all
    deletions); unmatched hypothesis speakers against an empty reference (all
    insertions). So over- or under-counting speakers is penalised.
    """
    ref = _by_speaker(ref_utts)
    hyp = _by_speaker(hyp_segs)
    ref_spk, hyp_spk = list(ref), list(hyp)
    n = max(len(ref_spk), len(hyp_spk))
    ref_streams = [ref[s] for s in ref_spk] + [""] * (n - len(ref_spk))
    hyp_streams = [hyp[s] for s in hyp_spk] + [""] * (n - len(hyp_spk))

    pair = [[_errors(r, h) for h in hyp_streams] for r in ref_streams]
    cost = np.array([[p["sub"] + p["del"] + p["ins"] for p in row] for row in pair])
    rows, cols = linear_sum_assignment(cost)

    total = {"sub": 0, "del": 0, "ins": 0, "ref_words": 0}
    mapping = {}
    for r, c in zip(rows, cols):
        for k in total:
            total[k] += pair[r][c][k]
        if r < len(ref_spk) and c < len(hyp_spk):
            mapping[hyp_spk[c]] = ref_spk[r]
    total["mapping"] = mapping
    total["ref_speakers"] = len(ref_spk)
    total["hyp_speakers"] = len(hyp_spk)
    return total


def rate(c: dict) -> float:
    return (c["sub"] + c["del"] + c["ins"]) / c["ref_words"] if c["ref_words"] else float("nan")


# ---- diarization -----------------------------------------------------------


def load_rttm(path: str):
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p or p[0] != "SPEAKER":
                continue
            start, dur, spk = float(p[3]), float(p[4]), p[7]
            ann[Segment(start, start + dur)] = spk
    return ann


def load_uem(path: str):
    from pyannote.core import Segment, Timeline

    tl = Timeline()
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                tl.add(Segment(float(p[2]), float(p[3])))
    return tl


def der_components(ref_rttm: str, hyp_turns: list, uem_path: str | None = None) -> dict:
    """hyp_turns: [(start, end, label), ...] straight from the diarization pipeline."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref = load_rttm(ref_rttm)
    hyp = Annotation()
    for start, end, label in hyp_turns:
        hyp[Segment(start, end)] = label
    uem = load_uem(uem_path) if uem_path else None

    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    d = metric(ref, hyp, uem=uem, detailed=True)
    return {
        "total": d["total"],
        "missed": d["missed detection"],
        "false_alarm": d["false alarm"],
        "confusion": d["confusion"],
    }


def der_rate(c: dict) -> float:
    return (c["missed"] + c["false_alarm"] + c["confusion"]) / c["total"] if c["total"] else float("nan")
