FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_PROGRESS_BAR=off \
    HF_HOME=/app/hf_cache \
    MODELS_DIR=/app/storage/models \
    MODERN_MOTION_PORT=8011 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0;PTX" \
    CUDA_MODULE_LOADING=LAZY \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.7"

WORKDIR /app

# 1. System dependencies (with BuildKit apt cache)
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    curl \
    wget \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

# 2. Python dependencies & Blackwell sm_120 PyTorch upgrade
RUN pip install --no-cache-dir --progress-bar off --upgrade \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --progress-bar off -r /app/requirements.txt

# 3. Storage and engine layout (engines and models mounted via volumes for persistence and fast rebuilds)
RUN mkdir -p /app/engines \
    /app/hf_cache \
    /app/storage/models \
    /app/storage/motion/uploads \
    /app/storage/motion/outputs

# 5. Application server code (copied at the very end so code adjustments build in <1s)
WORKDIR /app/server
COPY . /app/server

EXPOSE 8011

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8011/health || exit 1

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8011"]
