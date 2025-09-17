# SeedVR RunPod Serverless Dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# CRITICAL: Set non-interactive mode first to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PIP_ROOT_USER_ACTION=ignore
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev git wget curl \
    # Graphics and video libraries
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
    libgstreamer-plugins-bad1.0-0 gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    # Build tools for compilation
    build-essential cmake pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libatlas-base-dev gfortran \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone SeedVR repository
RUN git clone https://github.com/bytedance-seed/SeedVR.git . && \
    git checkout main

# Install PyTorch with CUDA support (heaviest, keep first for caching)
RUN pip3 install --no-cache-dir \
    torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install requirements.txt dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional packages required for SeedVR in ONE layer (for optimal caching)
RUN pip3 install --no-cache-dir \
    # RunPod SDK
    runpod \
    # Additional video/image processing
    opencv-python==4.9.0.80 \
    opencv-contrib-python \
    imageio imageio-ffmpeg \
    # ML packages
    accelerate \
    transformers \
    diffusers \
    # Utility packages
    omegaconf \
    einops \
    mediapy \
    tqdm \
    psutil

# Install flash attention (specific version for SeedVR)
RUN pip3 install --no-cache-dir flash_attn==2.5.9.post1 --no-build-isolation

# Install apex - try pre-built wheel first, then compile
RUN pip3 install --no-cache-dir \
    https://huggingface.co/ByteDance-Seed/SeedVR2-3B/resolve/main/apex-0.1-cp310-cp310-linux_x86_64.whl || \
    (git clone https://github.com/NVIDIA/apex && \
     cd apex && \
     pip3 install -v --disable-pip-version-check --no-cache-dir \
     --no-build-isolation --config-settings "--build-option=--cpp_ext" \
     --config-settings "--build-option=--cuda_ext" ./ && \
     cd .. && rm -rf apex)

# Create necessary directories for models and embeddings
RUN mkdir -p ckpts/

# Install huggingface_hub for model downloads
RUN pip3 install --no-cache-dir huggingface_hub

# Download SeedVR2-7B model only
RUN python3 -c "
from huggingface_hub import snapshot_download
import os

print('Downloading SeedVR2-7B model...')
snapshot_download(
    repo_id='ByteDance-Seed/SeedVR2-7B',
    local_dir='./ckpts/',
    cache_dir='./cache/',
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=['*.pth', '*.safetensors', '*.json', '*.txt', '*.md'],
    ignore_patterns=['*.bin', '*.onnx']
)

print('SeedVR2-7B model download completed!')
print('Downloaded files:')
import os
for root, dirs, files in os.walk('./ckpts/'):
    for file in files:
        print(f'  {os.path.join(root, file)}')
"

# Generate text embeddings using the SeedVR text encoder
COPY generate_embeddings.py ./
RUN python3 generate_embeddings.py

# Copy the serverless handler
COPY handler.py ./

# Set environment variable for RunPod serverless mode
ENV RUNPOD_SERVERLESS=1

# Health check script
RUN echo '#!/bin/bash\npython3 -c "import torch; print(f\"CUDA Available: {torch.cuda.is_available()}\"); print(f\"GPU Count: {torch.cuda.device_count()}\"); import sys; sys.exit(0 if torch.cuda.is_available() else 1)"' > /health_check.sh && \
    chmod +x /health_check.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /health_check.sh

# Expose port for health checks (not used in serverless mode)
EXPOSE 8080

# Set the command to run the serverless handler
CMD ["python3", "handler.py"]

# Build instructions:
# docker build -t seedvr-serverless .
#
# Local test:
# docker run --gpus all -e RUNPOD_SERVERLESS=1 seedvr-serverless
#
# RunPod deployment:
# 1. Push to Docker Hub: docker push username/seedvr-serverless:latest
# 2. Create RunPod serverless endpoint with this image
# 3. Set GPU type to H100 (recommended) or A100 
# 4. Set timeout to 300-600 seconds for video processing