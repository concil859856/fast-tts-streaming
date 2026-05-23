# syntax=docker/dockerfile:1.7
#
# Streaming voice-clone TTS server (Qwen3-TTS-12Hz-1.7B-Base) for the
# Vocence voice-agent pipeline. Runs on NVIDIA RTX 4090 (24 GB).
#
# Build:
#   docker build -t docker.io/<ns>/fast-tts-streaming:latest .
# Run (on the rented box):
#   docker run -d --gpus all --restart=unless-stopped \
#     -p 8111:8111 \
#     -e QWEN3_TTS_API_KEY=<key> \
#     docker.io/<ns>/fast-tts-streaming:latest

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # HF cache path so we can mount a volume to skip re-download across pulls
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf/transformers \
    QWEN3_TTS_HOST=0.0.0.0 \
    QWEN3_TTS_PORT=8111

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git curl ca-certificates \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Torch + CUDA-12.8 wheel (matches the driver on the rented 4090s).
# Pinning the index URL here means a `docker pull` later doesn't accidentally
# resolve to a CPU wheel when the user's pip config is something weird.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.5.1" torchaudio

# Faster-qwen3-tts (the inference engine) from PyPI. Pinned >=0.2.6 so we
# always have the streaming generator and the CUDA-graph capture path.
RUN pip install "faster-qwen3-tts>=0.2.6" "qwen-tts>=0.1.1"

# Copy this server's source.
WORKDIR /app
COPY pyproject.toml ./
COPY qwen3_tts_streaming/ ./qwen3_tts_streaming/
COPY samples/ ./samples/

# Install the service itself. --no-deps because torch + faster-qwen3-tts
# are already pinned above; we don't want pip to re-resolve them.
RUN pip install --no-deps -e .

# Cache mount point for HuggingFace model weights. Recommend mounting a
# persistent volume here so model download (~3.4 GB) only happens once per box.
VOLUME /cache/hf

EXPOSE 8111

# Container is healthy when /healthz returns 200 AND warmup is past.
# 60s start-period covers the model load + CUDA-graph capture.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${QWEN3_TTS_PORT:-8111}/healthz || exit 1

CMD ["qwen3-tts-streaming"]
