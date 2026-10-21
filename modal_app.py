"""
Kalam on Modal: the same FastAPI app as app.py, served from a GPU container.

Deployed by .github/workflows/deploy-modal.yml (or locally: `modal deploy modal_app.py`).
Needs a Modal secret named "kalam-hf" holding HF_TOKEN (and optionally
ALLOWED_ORIGINS); the workflow creates it from the GitHub secrets.

Why these choices:
- GPU (T4, needs a card on the Modal account) with float16: large-v3 runs many
  times faster than real time. Without a card it runs on 8 CPUs with
  large-v3-turbo instead: slower, but no payment method needed.
- One model for every language: WHISPER_MODEL and ARABIC_MODEL are the same,
  so a single copy is loaded and shared.
- Models are downloaded while the image is built, so a cold start loads them from
  disk instead of re-downloading several GB.
- Scales to zero after SCALEDOWN seconds idle, so free credits are only spent
  while recordings are being transcribed. max_containers=1 caps the spend.
"""

import os

import modal

# GPU needs a payment method on the Modal account (free credits still apply).
# Leave MODAL_GPU empty to run on CPUs, which works on a card-less account but is
# several times slower; set it to "T4" (GitHub variable MODAL_GPU) once a card is added.
GPU = os.environ.get("MODAL_GPU", "").strip() or None
# On CPU, large-v3-turbo keeps waits reasonable; on GPU, full large-v3 is affordable.
MODEL = os.environ.get("MODAL_MODEL", "").strip() or ("large-v3" if GPU else "large-v3-turbo")
CPUS = 8.0
MEMORY_MB = 16384
SCALEDOWN = 300  # seconds a warm container waits for the next request

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "fastapi==0.115.6",
        "python-multipart==0.0.20",
        "faster-whisper==1.1.1",
        "pyannote.audio==3.3.2",
        "huggingface_hub<1.0",
        "soundfile",
        "matplotlib",
        "torch==2.5.1",
        "torchaudio==2.5.1",
    )
    .env({
        "HF_HOME": "/models",
        # Baked in so the container evaluates this file with the same settings as the deploy.
        "MODAL_GPU": GPU or "",
        "MODAL_MODEL": MODEL,
        "DEVICE": "cuda" if GPU else "cpu",
        "COMPUTE_TYPE": "float16" if GPU else "int8",
        "OMP_NUM_THREADS": "8",
        "WHISPER_MODEL": MODEL,
        "ARABIC_MODEL": MODEL,
        "DIARIZE": "true",
        "PYTHONUNBUFFERED": "1",
    })
)


def download_models():
    """Bake Whisper and pyannote weights into the image."""
    import os

    from faster_whisper import download_model
    from pyannote.audio import Pipeline

    download_model(MODEL)
    Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"])


image = image.run_function(download_models, secrets=[modal.Secret.from_name("kalam-hf")]).add_local_python_source(
    "app", "arabic"
)

# Same name as the existing deployment, so its URL (…--kalam-api-web.modal.run) keeps working.
app = modal.App("kalam-api", image=image)


@app.function(
    gpu=GPU,
    cpu=CPUS,
    memory=MEMORY_MB,
    secrets=[modal.Secret.from_name("kalam-hf")],
    scaledown_window=SCALEDOWN,
    max_containers=1,
    timeout=1800,
)
@modal.asgi_app()
def web():
    import app as kalam  # imported here so it reads the container's environment

    return kalam.app
