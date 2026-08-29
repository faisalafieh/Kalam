# Transcription API

Speaker-attributed transcription for recorded hearings and meetings.
Takes an audio file, returns a transcript where every line is labelled with who spoke it.

Pairs with the browser recorder frontend (hosted separately on Netlify).

## What it does

1. Accepts any audio container the browser produces (webm/opus, mp4, wav)
2. Normalises it to 16 kHz mono with ffmpeg
3. Transcribes with faster-whisper
4. Diarizes with pyannote 3.1
5. Assigns each transcript segment to the speaker with the largest time overlap

## API

```
POST /transcribe        multipart/form-data, field "audio"
GET  /health
```

Response:

```json
{
  "segments": [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.4, "text": "You may proceed."},
    {"speaker": "SPEAKER_01", "start": 2.5, "end": 6.1, "text": "Thank you, turning to the exhibit."}
  ],
  "speakers": 2,
  "duration": 61.2,
  "language": "en"
}
```

## Before it will run

`pyannote/speaker-diarization-3.1` is a gated model. Two steps, both required:

1. Log in to Hugging Face and accept the terms on the model page **and** on `pyannote/segmentation-3.0`
2. Create a read token in Settings → Access Tokens, and set it as `HF_TOKEN`

Without the token the server still starts, but returns transcripts with no speaker labels.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your HF_TOKEN
export $(cat .env | xargs)
uvicorn app:app --reload
```

Needs ffmpeg installed (`brew install ffmpeg`, or `apt install ffmpeg`).

Test it:

```bash
curl -F "audio=@sample.wav" http://localhost:8000/transcribe
```

## Deploy

Any host that runs a Dockerfile works. Memory is the binding constraint:
whisper-small plus pyannote needs roughly 2 GB, so a 512 MB free tier will
be killed mid-request.

| Host | Notes |
|---|---|
| Render | Docker runtime, `render.yaml` included. Free tier too small for diarization; the paid starter instance fits. |
| Modal | Serverless, charges per second of execution, free monthly credits. Best fit if the model needs a GPU. |
| Fly.io | Scale-to-zero, small always-free allowance. |

Set `ALLOWED_ORIGINS` to your frontend's exact URL once deployed, otherwise
the browser will block the upload.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token, required for diarization |
| `WHISPER_MODEL` | `small` | `tiny` and `base` are much faster on CPU and much less accurate |
| `DIARIZE` | `true` | Set `false` to return transcripts without speaker labels |
| `DEVICE` | `cpu` | `cuda` when a GPU is available |
| `COMPUTE_TYPE` | `int8` | `float16` on GPU |
| `ALLOWED_ORIGINS` | `*` | Comma-separated list of allowed frontend origins |
| `MAX_MB` | `200` | Rejects uploads larger than this |

## Handling of audio

Uploads are written to a temporary directory and deleted when the request
finishes, whether it succeeded or not. Nothing is written to a database or
retained between requests.
