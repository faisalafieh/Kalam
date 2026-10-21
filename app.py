"""
Transcription + speaker attribution API.

POST /transcribe   multipart/form-data, field name "audio"
  optional fields: language (en | ar | mixed | auto), dialect (auto | msa | gulf |
  levantine | egyptian | maghrebi), vocabulary (comma-separated names/terms),
  num_speakers / min_speakers / max_speakers
  -> {"segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.4, "text": "..."}],
      "speakers": 2, "duration": 61.2, "language": "ar", "dialect": "gulf"}

GET  /health       -> {"status": "ok", "model": "...", "diarization": true}
"""

import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import arabic

# ---- configuration (set these as environment variables on the host) --------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")        # tiny | base | small | medium | large-v3
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")           # int8 on CPU, float16 on GPU
DEVICE = os.getenv("DEVICE", "cpu")
DIARIZE = os.getenv("DIARIZE", "true").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN")                           # required for pyannote
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_MB = int(os.getenv("MAX_MB", "200"))
# Fix the spoken language. "auto" lets Whisper guess from the first 30 s, which
# fails on accented speakers (e.g. Dutch speakers of English get Dutch output).
# "mixed" re-detects the language on every segment, for Arabic-English code-switching.
# Callers can override this per request.
LANGUAGE = os.getenv("LANGUAGE", "en")
# Arabic dialect: auto | msa | gulf | levantine | egyptian | maghrebi (see arabic.py).
DIALECT = os.getenv("DIALECT", "auto")
# Whisper model for Arabic and mixed requests. Small Whisper models are much weaker
# on Arabic than on English, so Arabic gets its own (larger) model, loaded on the
# first Arabic request unless PRELOAD_ARABIC is set.
ARABIC_MODEL = os.getenv("ARABIC_MODEL", "large-v3")
PRELOAD_ARABIC = os.getenv("PRELOAD_ARABIC", "false").lower() == "true"
# Seconds of audio transcribed to guess the dialect when dialect is "auto".
DIALECT_PROBE_SECONDS = int(os.getenv("DIALECT_PROBE_SECONDS", "90"))
# Names and terms to hint to Whisper on every window (judges, parties, case terms),
# comma-separated. Requests can add their own.
VOCABULARY = os.getenv("VOCABULARY", "")
# Optional ffmpeg audio filter applied while converting, e.g.
# "highpass=f=80,loudnorm" lifts quiet, far-from-the-mic speakers. Off by default:
# measure on your dev recordings before switching it on.
AUDIO_FILTER = os.getenv("AUDIO_FILTER", "")
# With word timestamps on, skip silent stretches longer than this many seconds when
# Whisper looks like it is hallucinating into silence. 0 = off.
HALLUCINATION_SILENCE = float(os.getenv("HALLUCINATION_SILENCE", "0")) or None
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))                # 1 = greedy: faster, skips more speech
# Conditioning each 30 s window on the previous text lets one bad window derail
# the rest (repetition loops, skipped stretches). Off is safer for long meetings.
CONDITION_ON_PREVIOUS = os.getenv("CONDITION_ON_PREVIOUS", "false").lower() == "true"
# Assign speakers word by word (using Whisper word timestamps) instead of giving
# a whole segment to one speaker. Fixes segments that span a change of speaker.
WORD_SPEAKERS = os.getenv("WORD_SPEAKERS", "true").lower() == "true"
# Separate overlapping speakers into one audio stream each (pyannote PixIT model,
# trained on AMI) and transcribe every stream on its own, so crosstalk is not lost.
# Heavier: needs `pip install "pyannote.audio[separation]==3.3.2"` and a GPU in practice.
SEPARATE = os.getenv("SEPARATE", "false").lower() == "true"
SEPARATION_MODEL = "pyannote/speech-separation-ami-1.0"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
# ---------------------------------------------------------------------------

state = {"asr": None, "diar": None, "models": {}}

ARABIC_LANGUAGES = ("ar", "mixed")


def model_for(language: str) -> str:
    """Which Whisper checkpoint serves this language setting."""
    return ARABIC_MODEL if language in ARABIC_LANGUAGES else WHISPER_MODEL


def get_whisper(name: str):
    """Load a Whisper model once and reuse it across requests."""
    from faster_whisper import WhisperModel

    models = state.setdefault("models", {})
    if name not in models:
        models[name] = WhisperModel(name, device=DEVICE, compute_type=COMPUTE_TYPE)
    return models[name]


def load_models():
    """Load the ASR model and (optionally) the diarization pipeline.

    Shared by the API server and eval/evaluate.py so both run identical models.
    The ASR model is the one for the configured LANGUAGE.
    """
    asr = get_whisper(model_for(LANGUAGE))
    diar = None
    if DIARIZE:
        if not HF_TOKEN:
            print("DIARIZE is on but HF_TOKEN is missing — running without speaker labels.")
        else:
            from pyannote.audio import Pipeline

            diar = Pipeline.from_pretrained(
                SEPARATION_MODEL if SEPARATE else DIARIZATION_MODEL, use_auth_token=HF_TOKEN
            )
            if DEVICE == "cuda":
                import torch

                diar.to(torch.device("cuda"))  # pyannote stays on CPU unless moved
    return asr, diar


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup rather than per request."""
    state["asr"], state["diar"] = load_models()
    if PRELOAD_ARABIC:
        get_whisper(ARABIC_MODEL)
    yield
    state.clear()


app = FastAPI(title="Transcription API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,      # set to your Netlify URL in production
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


def to_wav(src: str, dst: str) -> None:
    """Normalise any container the browser sends into 16 kHz mono PCM."""
    filters = ["-af", AUDIO_FILTER] if AUDIO_FILTER else []
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src, *filters, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst],
        capture_output=True,
    )
    if result.returncode != 0:
        raise HTTPException(400, f"Could not decode audio: {result.stderr.decode()[-400:]}")


class SpeakerIndex:
    """Fast lookup of which diarized speaker was talking during [start, end]."""

    def __init__(self, turns):
        import bisect

        self._bisect = bisect
        self.turns = sorted((float(t.start), float(t.end), label) for t, _, label in turns)
        self.starts = [t[0] for t in self.turns]
        self.max_len = max((e - s for s, e, _ in self.turns), default=0.0)

    def speaker(self, start: float, end: float) -> str:
        """Speaker with the most overlap; if nobody overlaps, the nearest turn."""
        if not self.turns:
            return "SPEAKER"
        hi = self._bisect.bisect_right(self.starts, end)
        lo = self._bisect.bisect_left(self.starts, start - self.max_len)
        best, best_overlap = None, 0.0
        for s, e, label in self.turns[lo:hi]:
            overlap = min(end, e) - max(start, s)
            if overlap > best_overlap:
                best, best_overlap = label, overlap
        if best is not None:
            return best
        mid = (start + end) / 2
        i = self._bisect.bisect_left(self.starts, mid)
        near = self.turns[max(0, i - 2): i + 2]
        return min(near, key=lambda t: max(t[0] - mid, mid - t[1], 0.0))[2]


def speaker_for(turns, start: float, end: float) -> str:
    """Pick the speaker whose turns overlap this segment the most."""
    return SpeakerIndex(turns).speaker(start, end)


def attribute(segments, index: SpeakerIndex) -> list[dict]:
    """Split Whisper segments into speaker runs.

    With word timestamps, every word is labelled on its own and consecutive words
    from the same speaker are merged back into one segment. Without them (or for
    a segment with no word timings), the whole segment goes to one speaker.
    """
    out = []

    def emit(speaker, start, end, text):
        text = text.strip()
        if not text:
            return
        if index.turns and out and out[-1]["speaker"] == speaker and start - out[-1]["end"] < 1.0:
            out[-1]["text"] += " " + text
            out[-1]["end"] = round(end, 2)
        else:
            out.append({"speaker": speaker, "start": round(start, 2), "end": round(end, 2), "text": text})

    for seg in segments:
        words = getattr(seg, "words", None)
        if WORD_SPEAKERS and words:
            run_spk, run_start, run_end, run_text = None, 0.0, 0.0, ""
            for w in words:
                spk = index.speaker(w.start, w.end)
                if spk != run_spk and run_text:
                    emit(run_spk, run_start, run_end, run_text)
                    run_text = ""
                if not run_text:
                    run_spk, run_start = spk, w.start
                run_text += w.word
                run_end = w.end
            if run_text:
                emit(run_spk, run_start, run_end, run_text)
        else:
            emit(index.speaker(seg.start, seg.end), seg.start, seg.end, seg.text)
    return out


def load_wav(path: str):
    """Read the 16 kHz mono wav produced by to_wav() as float32 samples.

    Passing samples to Whisper skips its own PyAV-based decoder, which breaks
    whenever the installed PyAV is newer than faster-whisper expects.
    """
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sr} Hz (run it through to_wav first)")
    return np.ascontiguousarray(audio)


@dataclass
class Request:
    """Per-recording settings. Defaults come from the environment."""

    language: str = LANGUAGE
    dialect: str = DIALECT
    vocabulary: str = VOCABULARY
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    @property
    def is_arabic(self) -> bool:
        return self.language in ARABIC_LANGUAGES


def hints(req: Request, dialect: str | None) -> str | None:
    """Text passed to Whisper as hotwords: dialect primer plus caller vocabulary."""
    parts = []
    if dialect in arabic.PRIMERS:
        parts.append(arabic.PRIMERS[dialect])
    terms = [t.strip() for t in req.vocabulary.split(",") if t.strip()]
    if terms:
        parts.append("، ".join(terms) if dialect else ", ".join(terms))
    return " ".join(parts) or None


def _asr_options(req: Request | None = None, hotwords: str | None = None) -> dict:
    req = req or Request()
    lang = req.language
    opts = dict(
        language=None if lang in ("auto", "mixed") else lang,
        beam_size=BEAM_SIZE,
        condition_on_previous_text=CONDITION_ON_PREVIOUS,
        vad_filter=True,
    )
    if lang == "mixed":
        opts["multilingual"] = True  # detect the language on every segment
    if hotwords:
        opts["hotwords"] = hotwords
    return opts


def resolve_dialect(audio, asr, req: Request) -> tuple[str | None, bool]:
    """Return (dialect, was_detected). None for non-Arabic requests.

    With dialect "auto", transcribe the opening DIALECT_PROBE_SECONDS as plain
    Arabic and score it against dialect marker words (arabic.detect_dialect).
    """
    if not req.is_arabic:
        return None, False
    if req.dialect != "auto":
        return req.dialect, False
    probe = audio[: DIALECT_PROBE_SECONDS * 16000]
    segments, _ = asr.transcribe(
        probe, language="ar", beam_size=BEAM_SIZE, condition_on_previous_text=False, vad_filter=True
    )
    dialect, _ = arabic.detect_dialect(" ".join(s.text for s in segments))
    return dialect, True


def speaker_hints(req: Request) -> dict:
    """Known speaker counts sharply cut diarization errors; pass them when given."""
    if req.num_speakers:
        return {"num_speakers": req.num_speakers}
    return {k: v for k, v in (("min_speakers", req.min_speakers), ("max_speakers", req.max_speakers)) if v}


def run_separated(wav: str, asr, sep, req: Request | None = None, opts: dict | None = None) -> dict:
    """Separation mode: one isolated stream per speaker, each transcribed alone.

    Every word in a stream belongs to that stream's speaker, so overlapping speech
    is transcribed for both people instead of one voice drowning out the other.
    """
    import numpy as np

    req = req or Request()
    opts = opts or _asr_options(req)
    diarization, sources = sep(wav, **speaker_hints(req))
    out, language, duration = [], None, 0.0
    # Source column s belongs to the s-th label (pyannote model card convention).
    for s, speaker in enumerate(diarization.labels()):
        if s >= sources.data.shape[1]:
            break
        audio = np.ascontiguousarray(sources.data[:, s], dtype=np.float32)
        peak = float(np.abs(audio).max())
        if peak < 1e-4:
            continue  # silent stream
        audio *= 0.9 / peak  # separated streams come out at arbitrary gain
        segments, info = asr.transcribe(audio, **opts)
        language = language or info.language
        duration = max(duration, info.duration)
        for seg in segments:
            text = seg.text.strip()
            if text:
                out.append({"speaker": speaker, "start": round(seg.start, 2),
                            "end": round(seg.end, 2), "text": text})
    out.sort(key=lambda x: (x["start"], x["end"]))
    turns = [(t.start, t.end, label) for t, _, label in diarization.itertracks(yield_label=True)]
    return {
        "segments": out,
        "speakers": len({x["speaker"] for x in out}),
        "duration": round(duration, 1),
        "language": language or LANGUAGE,
        "turns": turns,
    }


def run_pipeline(wav: str, asr, diar, req: Request | None = None) -> dict:
    """Transcribe a 16 kHz mono wav and attribute each segment to a speaker.

    This is the whole Kalam pipeline; /transcribe and eval/evaluate.py both call it.
    """
    req = req or Request()
    audio = load_wav(wav)
    dialect, detected = resolve_dialect(audio, asr, req)
    opts = _asr_options(req, hints(req, dialect))

    if SEPARATE and diar is not None:
        result = run_separated(wav, asr, diar, req, opts)
    else:
        word_ts = WORD_SPEAKERS and diar is not None
        if word_ts and HALLUCINATION_SILENCE:
            opts["hallucination_silence_threshold"] = HALLUCINATION_SILENCE
        segments, info = asr.transcribe(audio, word_timestamps=word_ts, **opts)
        segments = list(segments)

        turns = []
        if diar is not None:
            turns = list(diar(wav, **speaker_hints(req)).itertracks(yield_label=True))

        out = attribute(segments, SpeakerIndex(turns))
        result = {
            "segments": out,
            "speakers": len({s["speaker"] for s in out}),
            "duration": round(info.duration, 1),
            "language": info.language,
            "turns": [(t.start, t.end, label) for t, _, label in turns],
        }
    result["dialect"] = dialect
    result["dialect_detected"] = detected
    return result


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": WHISPER_MODEL,
        "arabic_model": ARABIC_MODEL,
        "loaded_models": sorted(state.get("models", {})),
        "diarization": state.get("diar") is not None,
    }


LANGUAGE_CHOICES = ("en", "ar", "mixed", "auto")


def build_request(language, dialect, vocabulary, num_speakers, min_speakers, max_speakers) -> Request:
    req = Request()
    if language:
        language = language.strip().lower()
        if language not in LANGUAGE_CHOICES:
            raise HTTPException(400, f"language must be one of {', '.join(LANGUAGE_CHOICES)}")
        req.language = language
    if dialect:
        dialect = dialect.strip().lower()
        if dialect not in ("auto", *arabic.DIALECTS):
            raise HTTPException(400, f"dialect must be auto or one of {', '.join(arabic.DIALECTS)}")
        req.dialect = dialect
    if vocabulary:
        if len(vocabulary) > 500:
            raise HTTPException(400, "vocabulary is limited to 500 characters")
        req.vocabulary = ", ".join(t for t in (VOCABULARY, vocabulary) if t)
    for name, value in (("num_speakers", num_speakers), ("min_speakers", min_speakers),
                        ("max_speakers", max_speakers)):
        if value is not None and not 1 <= value <= 20:
            raise HTTPException(400, f"{name} must be between 1 and 20")
        setattr(req, name, value)
    return req


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
    dialect: str | None = Form(None),
    vocabulary: str | None = Form(None),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    if state["asr"] is None:
        raise HTTPException(503, "Model still loading, try again in a moment.")
    req = build_request(language, dialect, vocabulary, num_speakers, min_speakers, max_speakers)

    workdir = tempfile.mkdtemp()
    try:
        raw = os.path.join(workdir, "input")
        with open(raw, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        size_mb = os.path.getsize(raw) / 1048576
        if size_mb > MAX_MB:
            raise HTTPException(413, f"File is {size_mb:.0f} MB, limit is {MAX_MB} MB.")

        wav = os.path.join(workdir, "audio.wav")
        to_wav(raw, wav)

        asr = state["asr"] if model_for(req.language) == model_for(LANGUAGE) else get_whisper(model_for(req.language))
        result = run_pipeline(wav, asr, state["diar"], req)
        result.pop("turns")  # raw diarization turns are only needed for evaluation
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
