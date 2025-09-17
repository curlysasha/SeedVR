# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
SeedVR is a state-of-the-art video restoration framework using Diffusion Transformers. The project includes two main models:
- **SeedVR**: Multi-step diffusion model for generic video restoration (CVPR 2025 Highlight)
- **SeedVR2**: One-step video restoration via adversarial post-training

This is a research codebase implementing large-scale diffusion transformers for video super-resolution, denoising, and restoration tasks.

## Essential Commands

### Environment Setup
```bash
# Create conda environment
conda create -n seedvr python=3.10 -y
conda activate seedvr

# Install dependencies
pip install -r requirements.txt
pip install flash_attn==2.5.9.post1 --no-build-isolation

# Install apex (required for training/inference)
pip install apex-0.1-cp310-cp310-linux_x86_64.whl
```

### Model Download
```python
from huggingface_hub import snapshot_download

save_dir = "ckpts/"
repo_id = "ByteDance-Seed/SeedVR2-3B"  # or SeedVR2-7B, SeedVR-3B, SeedVR-7B
cache_dir = save_dir + "/cache"

snapshot_download(
    cache_dir=cache_dir,
    local_dir=save_dir,
    repo_id=repo_id,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.json", "*.safetensors", "*.pth", "*.bin", "*.py", "*.md", "*.txt"],
)
```

### Inference Commands
```bash
# SeedVR2-3B (one-step, faster)
torchrun --nproc-per-node=NUM_GPUS projects/inference_seedvr2_3b.py \
    --video_path INPUT_FOLDER \
    --output_dir OUTPUT_FOLDER \
    --seed SEED_NUM \
    --res_h OUTPUT_HEIGHT \
    --res_w OUTPUT_WIDTH \
    --sp_size NUM_SP

# SeedVR2-7B (one-step, higher quality)
torchrun --nproc-per-node=NUM_GPUS projects/inference_seedvr2_7b.py \
    --video_path INPUT_FOLDER \
    --output_dir OUTPUT_FOLDER \
    --seed SEED_NUM \
    --res_h OUTPUT_HEIGHT \
    --res_w OUTPUT_WIDTH \
    --sp_size NUM_SP

# Multi-step models (SeedVR-3B/7B) available for comparison
```

## Architecture & Key Components

### Core Model Architecture
- **NaDiT (Native Resolution Diffusion Transformer)**: Main model architecture in `models/dit_v2/nadit.py`
- **Sequence Parallel**: Multi-GPU inference via sequence parallelism for handling large resolutions
- **Window Attention**: Adaptive window mechanisms for efficient high-resolution processing
- **Video VAE**: Temporal-aware video encoder/decoder in `models/video_vae_v3/`

### Configuration System
- **YAML Configs**: Model configurations in `configs_3b/` and `configs_7b/`
- **OmegaConf**: Hierarchical configuration management with parameter inheritance
- **Model Variants**: 3B and 7B parameter versions with different window strategies

### Distributed Training Framework
- **Multi-GPU Support**: Comprehensive distributed training in `common/distributed/`
- **FSDP Integration**: Fully Sharded Data Parallel with hybrid sharding strategies
- **Sequence Parallel**: Custom implementation for long sequence video processing
- **Gradient Checkpointing**: Memory-efficient training for large models

### Data Processing Pipeline
- **Video Transforms**: Specialized video preprocessing in `data/video/transforms/`
- **Image Transforms**: Resolution-aware transforms in `data/image/transforms/`
- **NA Resize**: Native resolution processing without fixed aspect ratios
- **Divisible Crop**: Ensures proper patch-based processing

## Technical Implementation Details

### Memory Management
- **GPU Requirements**: 1 H100-80G for 720p videos, 4 H100-80G for 1080p/2K
- **VAE Slicing**: Temporal slicing for large video processing (`split_size: 4`)
- **Memory Limits**: Configurable memory limits for conv and norm operations
- **BFloat16**: Mixed precision training and inference

### Diffusion Framework
- **Schedule Types**: Linear interpolation (lerp) schedules in `common/diffusion/schedules/`
- **Samplers**: Euler sampling with v-parameterization in `common/diffusion/samplers/`
- **Timesteps**: LogitNormal training distribution, uniform trailing for sampling
- **Loss Functions**: V-parameterization loss with CFG support

### Model Variants and Configurations
- **SeedVR2-3B**: One-step model, faster inference, good quality
- **SeedVR2-7B**: One-step model, highest quality, slower inference  
- **SeedVR-3B/7B**: Multi-step models for comparison and ablation studies
- **Window Strategies**: Different attention window patterns for resolution adaptation

## Important Notes

### GPU and Memory Requirements
- Minimum 1 H100-80G for basic 720p video processing
- 4 H100-80G recommended for high-resolution (1080p+) videos
- Sequence parallel (`sp_size`) parameter controls GPU distribution
- Flash Attention 2.5.9.post1 required for efficient attention computation

### Model Limitations
- Prototype models may not perfectly align with paper results
- Not robust to very heavy degradations or large motions
- May oversharpen on lightly degraded inputs (especially 720p AIGC videos)
- Strong generation ability can sometimes produce overly enhanced details

### Color Fix Integration
- Optional color correction available via `color_fix.py` from StableSR
- Place the file at `./projects/video_diffusion_sr/color_fix.py` to enable
- Uses wavelet reconstruction for color consistency

### Inference Optimization
- Use sequence parallel for multi-GPU speedup
- Adjust `res_h` and `res_w` based on available GPU memory
- Consider using SeedVR2 models for faster one-step inference
- Future updates will include Tile-VAE and Progressive Aggregation Sampling

### Development Environment
- Python 3.10 recommended (3.9 supported via environment.yml)
- CUDA 12.1+ required for flash attention and distributed training
- Apex must be compiled for specific Python/PyTorch/CUDA combination
- All dependencies pinned for reproducibility

## File Structure Understanding
- `projects/`: Main inference scripts for different model variants
- `models/dit_v2/`: Latest NaDiT transformer architecture
- `models/dit/`: Legacy transformer implementation  
- `common/`: Shared utilities for distributed training, diffusion, caching
- `data/`: Video and image preprocessing pipelines
- `configs_*/`: YAML configurations for different model sizes