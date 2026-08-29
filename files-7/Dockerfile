FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache model downloads inside the image layer, not on every cold start
ENV HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
