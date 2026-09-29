"""
Transcription + speaker attribution API.

POST /transcribe   multipart/form-data, field name "audio"
  -> {"segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.4, "text": "..."}],
      "speakers": 2, "duration": 61.2}

GET  /health       -> {"status": "ok", "model": "...", "diarization": true}
"""

import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ---- configuration (set these as environment variables on the host) --------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")        # tiny | base | small | medium | large-v3
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")           # int8 on CPU, float16 on GPU
DEVICE = os.getenv("DEVICE", "cpu")
DIARIZE = os.getenv("DIARIZE", "true").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN")                           # required for pyannote
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_MB = int(os.getenv("MAX_MB", "200"))
# ---------------------------------------------------------------------------

state = {"asr": None, "diar": None}


def load_models():
    """Load the ASR model and (optionally) the diarization pipeline.

    Shared by the API server and eval/evaluate.py so both run identical models.
    """
    from faster_whisper import WhisperModel

    asr = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    diar = None
    if DIARIZE:
        if not HF_TOKEN:
            print("DIARIZE is on but HF_TOKEN is missing — running without speaker labels.")
        else:
            from pyannote.audio import Pipeline

            diar = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
            )
    return asr, diar


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup rather than per request."""
    state["asr"], state["diar"] = load_models()
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
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst],
        capture_output=True,
    )
    if result.returncode != 0:
        raise HTTPException(400, f"Could not decode audio: {result.stderr.decode()[-400:]}")


def speaker_for(turns, start: float, end: float) -> str:
    """Pick the speaker whose turns overlap this segment the most."""
    best, best_overlap = "SPEAKER_00", 0.0
    for turn, _, label in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best, best_overlap = label, overlap
    return best


def run_pipeline(wav: str, asr, diar) -> dict:
    """Transcribe a 16 kHz mono wav and attribute each segment to a speaker.

    This is the whole Kalam pipeline; /transcribe and eval/evaluate.py both call it.
    """
    segments, info = asr.transcribe(wav, vad_filter=True, beam_size=1)
    segments = list(segments)

    turns = []
    if diar is not None:
        turns = list(diar(wav).itertracks(yield_label=True))

    out = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        out.append(
            {
                "speaker": speaker_for(turns, seg.start, seg.end) if turns else "SPEAKER",
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": text,
            }
        )

    return {
        "segments": out,
        "speakers": len({s["speaker"] for s in out}),
        "duration": round(info.duration, 1),
        "language": info.language,
        "turns": [(t.start, t.end, label) for t, _, label in turns],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": WHISPER_MODEL,
        "diarization": state.get("diar") is not None,
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if state["asr"] is None:
        raise HTTPException(503, "Model still loading, try again in a moment.")

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

        result = run_pipeline(wav, state["asr"], state["diar"])
        result.pop("turns")  # raw diarization turns are only needed for evaluation
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
