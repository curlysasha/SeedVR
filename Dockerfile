# SeedVR RunPod Serverless Dockerfile with improved caching
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Non-interactive setup and runtime defaults
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# System dependencies (rarely change, keep early for caching)
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev git wget curl \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
    libgstreamer-plugins-bad1.0-0 gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    build-essential cmake pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libatlas-base-dev gfortran \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first so code changes keep cache hits
COPY requirements.txt ./

# Core ML stack
RUN pip3 install --no-cache-dir \
    torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Utilities required before copying the codebase
RUN pip3 install --no-cache-dir huggingface_hub

# Project dependencies from requirements (share PyTorch index for matching wheels)
RUN pip3 install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.txt

# Additional runtime packages that are not listed in requirements
RUN pip3 install --no-cache-dir \
    runpod==1.6.2 \
    opencv-contrib-python==4.9.0.80 \
    imageio==2.34.0 imageio-ffmpeg==0.5.1 \
    accelerate==0.27.2 \
    transformers==4.38.2 \
    diffusers==0.29.1 \
    omegaconf==2.3.0 \
    einops==0.7.0 \
    mediapy==1.2.0 \
    tqdm==4.66.2 \
    psutil==5.9.8

# Flash attention and Apex (fallback compilation if wheel unavailable)
RUN pip3 install --no-cache-dir flash_attn==2.5.9.post1 --no-build-isolation
RUN pip3 install --no-cache-dir \
    https://huggingface.co/ByteDance-Seed/SeedVR2-3B/resolve/main/apex-0.1-cp310-cp310-linux_x86_64.whl || \
    (git clone https://github.com/NVIDIA/apex && \
     cd apex && \
     pip3 install -v --disable-pip-version-check --no-cache-dir \
       --no-build-isolation --config-settings "--build-option=--cpp_ext" \
       --config-settings "--build-option=--cuda_ext" ./ && \
     cd .. && rm -rf apex)

# Install PyAV (requires libs from apt layer above)
RUN pip3 install --no-cache-dir av==11.0.0

# Copy application code (including ckpts/ with pre-downloaded weights, if present) last
COPY . .

# Ensure precomputed text embeddings are present
RUN test -f pos_emb.pt && test -f neg_emb.pt

# Serverless defaults
ENV RUNPOD_SERVERLESS=1

# Health check script
RUN echo '#!/bin/bash\npython3 -c "import torch; print(f\"CUDA Available: {torch.cuda.is_available()}\"); print(f\"GPU Count: {torch.cuda.device_count()}\"); import sys; sys.exit(0 if torch.cuda.is_available() else 1)"' > /health_check.sh && \
    chmod +x /health_check.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /health_check.sh

EXPOSE 8080

CMD ["python3", "handler.py"]

# Build: docker build -t seedvr-serverless .
# Run:   docker run --gpus all seedvr-serverless
