#!/usr/bin/env python3

import os
import sys
import io
import base64
import tempfile
import shutil
import torch
import mediapy
import traceback
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional
import zipfile
import time

# Set environment variables before imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['RUNPOD_SERVERLESS'] = '1'

import runpod


class SeedVRManager:
    def __init__(self):
        self.runner = None
        self.initialized = False
        self.temp_dir = None
        
    def initialize_model(self, model_type="seedvr2_7b", sp_size=1):
        """Initialize SeedVR model for serverless inference"""
        try:
            print(f"Initializing {model_type} model...")
            
            # Import all required modules
            from einops import rearrange
            from omegaconf import OmegaConf
            import datetime
            from torchvision.transforms import Compose, Lambda, Normalize
            from torchvision.io.video import read_video
            from torchvision.io import read_image
            
            from data.image.transforms.divisible_crop import DivisibleCrop
            from data.image.transforms.na_resize import NaResize
            from data.video.transforms.rearrange import Rearrange
            
            from common.distributed import (
                get_device,
                init_torch,
            )
            from common.distributed.advanced import (
                get_data_parallel_rank,
                get_data_parallel_world_size,
                get_sequence_parallel_rank,
                get_sequence_parallel_world_size,
                init_sequence_parallel,
            )
            from projects.video_diffusion_sr.infer import VideoDiffusionInfer
            from common.config import load_config
            from common.distributed.ops import sync_data
            from common.seed import set_seed
            
            # Configure sequence parallel
            if sp_size > 1:
                init_sequence_parallel(sp_size)
            
            # Load configuration for 7B model
            if model_type.startswith("seedvr2_7b"):
                config_path = './configs_7b/main.yaml'
                checkpoint_path = './ckpts/seedvr2_ema_7b.pth'
            else:
                # Default to 7B model
                print(f"Unknown model type {model_type}, defaulting to seedvr2_7b")
                model_type = "seedvr2_7b"
                config_path = './configs_7b/main.yaml'
                checkpoint_path = './ckpts/seedvr2_ema_7b.pth'
            
            config = load_config(config_path)
            self.runner = VideoDiffusionInfer(config)
            OmegaConf.set_readonly(self.runner.config, False)
            
            # Initialize torch
            init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
            
            # Configure models
            self.runner.configure_dit_model(device="cuda", checkpoint=checkpoint_path)
            self.runner.configure_vae_model()
            
            # Set memory limit
            if hasattr(self.runner.vae, "set_memory_limit"):
                self.runner.vae.set_memory_limit(**self.runner.config.vae.memory_limit)
            
            # Configure diffusion (default settings for one-step)
            self.runner.config.diffusion.cfg.scale = 1.0
            self.runner.config.diffusion.cfg.rescale = 0.0
            self.runner.config.diffusion.timesteps.sampling.steps = 1
            self.runner.configure_diffusion()
            
            self.model_type = model_type
            self.sp_size = sp_size
            self.initialized = True
            
            print(f"✅ {model_type} model initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing model: {str(e)}")
            traceback.print_exc()
            return False
    
    def process_video(self, video_data: bytes, 
                     res_h: int = 1280, 
                     res_w: int = 720,
                     seed: int = 666,
                     cfg_scale: float = 1.0,
                     cfg_rescale: float = 0.0,
                     sample_steps: int = 1) -> Dict[str, Any]:
        """Process video for restoration"""
        
        if not self.initialized:
            return {"error": "Model not initialized"}
        
        try:
            # Set up temporary directories
            with tempfile.TemporaryDirectory() as temp_dir:
                input_dir = os.path.join(temp_dir, "input")
                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(input_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)
                
                # Save input video
                input_path = os.path.join(input_dir, "input_video.mp4")
                with open(input_path, 'wb') as f:
                    f.write(video_data)
                
                print(f"Processing video: {os.path.getsize(input_path)} bytes")
                
                # Import necessary modules for processing
                from einops import rearrange
                from torchvision.transforms import Compose, Lambda, Normalize
                from torchvision.io.video import read_video
                from data.image.transforms.divisible_crop import DivisibleCrop
                from data.image.transforms.na_resize import NaResize
                from data.video.transforms.rearrange import Rearrange
                from common.distributed import get_device
                from common.distributed.ops import sync_data
                from common.seed import set_seed
                from tqdm import tqdm
                
                # Set random seed
                set_seed(seed, same_across_ranks=True)
                
                # Update configuration
                self.runner.config.diffusion.cfg.scale = cfg_scale
                self.runner.config.diffusion.cfg.rescale = cfg_rescale
                self.runner.config.diffusion.timesteps.sampling.steps = sample_steps
                self.runner.configure_diffusion()
                
                # Load text embeddings
                text_pos_embeds = torch.load('pos_emb.pt')
                text_neg_embeds = torch.load('neg_emb.pt')
                text_embeds_dict = {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}
                
                # Set up video transforms
                video_transform = Compose([
                    NaResize(
                        resolution=(res_h * res_w) ** 0.5,
                        mode="area",
                        downsample_only=False,
                    ),
                    Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
                    DivisibleCrop((16, 16)),
                    Normalize(0.5, 0.5),
                    Rearrange("t c h w -> c t h w"),
                ])
                
                # Read and process video
                video, _, _ = read_video(input_path, pts_unit='sec', output_format='TCHW')
                video = video.float() / 255.0
                
                # Apply transforms
                video = video_transform(video)
                
                # Cut videos for sequence parallel
                def cut_videos(videos, sp_size):
                    t = videos.size(1)
                    if t == 1:
                        return videos
                    if t <= 4 * sp_size:
                        padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
                        padding = torch.cat(padding, dim=1)
                        videos = torch.cat([videos, padding], dim=1)
                        return videos
                    if (t - 1) % (4 * sp_size) == 0:
                        return videos
                    else:
                        padding = [videos[:, -1].unsqueeze(1)] * (
                            4 * sp_size - ((t - 1) % (4 * sp_size))
                        )
                        padding = torch.cat(padding, dim=1)
                        videos = torch.cat([videos, padding], dim=1)
                        return videos
                
                video = cut_videos(video, self.sp_size)
                
                # Encode to latent space
                with torch.no_grad():
                    cond_latents = [self.runner.vae.encode(video[None]).squeeze(0)]
                
                # Generation step
                def _move_to_cuda(x):
                    return [i.to(get_device()) for i in x]
                
                noises = [torch.randn_like(latent) for latent in cond_latents]
                aug_noises = [torch.randn_like(latent) for latent in cond_latents]
                
                noises, aug_noises, cond_latents = sync_data((noises, aug_noises, cond_latents), 0)
                noises, aug_noises, cond_latents = list(
                    map(lambda x: _move_to_cuda(x), (noises, aug_noises, cond_latents))
                )
                
                cond_noise_scale = 0.0
                
                def _add_noise(x, aug_noise):
                    t = torch.tensor([1000.0], device=get_device()) * cond_noise_scale
                    shape = torch.tensor(x.shape[1:], device=get_device())[None]
                    t = self.runner.timestep_transform(t, shape)
                    x = self.runner.schedule.forward(x, aug_noise, t)
                    return x
                
                conditions = [
                    self.runner.get_condition(
                        noise,
                        task="sr",
                        latent_blur=_add_noise(latent_blur, aug_noise),
                    )
                    for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
                ]
                
                # Inference
                print("Running inference...")
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
                    video_tensors = self.runner.inference(
                        noises=noises,
                        conditions=conditions,
                        dit_offload=True,
                        **text_embeds_dict,
                    )
                
                samples = [
                    (
                        rearrange(video[:, None], "c t h w -> t c h w")
                        if video.ndim == 3
                        else rearrange(video, "c t h w -> t c h w")
                    )
                    for video in video_tensors
                ]
                
                # Decode and save
                restored_videos = []
                for sample in samples:
                    sample = torch.stack([sample], dim=0)
                    with torch.no_grad():
                        decode_result = self.runner.vae.decode(sample)
                        decode_result = torch.clamp((decode_result + 1.0) / 2.0, 0.0, 1.0)
                        decode_result = (decode_result * 255.0).to(torch.uint8)
                        restore_result = rearrange(decode_result, "b c t h w -> b t h w c").cpu().numpy()
                    
                    # Save video
                    output_path = os.path.join(output_dir, f"restored_video_{len(restored_videos)}.mp4")
                    mediapy.write_video(output_path, restore_result[0], fps=24)
                    
                    # Read and encode to base64
                    with open(output_path, 'rb') as f:
                        video_bytes = f.read()
                        video_b64 = base64.b64encode(video_bytes).decode('utf-8')
                        restored_videos.append(video_b64)
                
                # Cleanup
                del video_tensors, samples, cond_latents, noises, aug_noises
                gc.collect()
                torch.cuda.empty_cache()
                
                return {
                    "success": True,
                    "videos": restored_videos,
                    "model_type": self.model_type,
                    "resolution": f"{res_h}x{res_w}",
                    "processing_info": {
                        "seed": seed,
                        "cfg_scale": cfg_scale,
                        "sample_steps": sample_steps,
                        "sequence_parallel_size": self.sp_size
                    }
                }
                
        except Exception as e:
            print(f"❌ Error processing video: {str(e)}")
            traceback.print_exc()
            return {"error": f"Processing failed: {str(e)}"}


# Global manager instance
seedvr_manager = SeedVRManager()


def handler(job):
    """RunPod serverless handler"""
    try:
        job_input = job.get("input", {})
        
        # Extract parameters
        action = job_input.get("action", "restore")
        
        if action == "initialize":
            # Initialize model
            model_type = job_input.get("model_type", "seedvr2_7b")
            sp_size = job_input.get("sp_size", 1)
            
            success = seedvr_manager.initialize_model(model_type, sp_size)
            
            if success:
                return {
                    "message": f"Model {model_type} initialized successfully",
                    "model_type": model_type,
                    "sp_size": sp_size,
                    "ready": True
                }
            else:
                return {"error": "Failed to initialize model"}
        
        elif action == "restore":
            # Process video restoration
            if not seedvr_manager.initialized:
                # Auto-initialize with default settings
                print("Auto-initializing model...")
                model_type = job_input.get("model_type", "seedvr2_7b")
                sp_size = job_input.get("sp_size", 1)
                success = seedvr_manager.initialize_model(model_type, sp_size)
                if not success:
                    return {"error": "Failed to initialize model"}
            
            # Get video data
            video_base64 = job_input.get("video")
            if not video_base64:
                return {"error": "No video data provided"}
            
            try:
                video_data = base64.b64decode(video_base64)
            except Exception as e:
                return {"error": f"Invalid base64 video data: {str(e)}"}
            
            # Get processing parameters
            res_h = job_input.get("res_h", 1280)
            res_w = job_input.get("res_w", 720)
            seed = job_input.get("seed", 666)
            cfg_scale = job_input.get("cfg_scale", 1.0)
            cfg_rescale = job_input.get("cfg_rescale", 0.0)
            sample_steps = job_input.get("sample_steps", 1)
            
            # Process video
            result = seedvr_manager.process_video(
                video_data=video_data,
                res_h=res_h,
                res_w=res_w,
                seed=seed,
                cfg_scale=cfg_scale,
                cfg_rescale=cfg_rescale,
                sample_steps=sample_steps
            )
            
            return result
            
        elif action == "health":
            # Health check
            return {
                "status": "healthy",
                "model_initialized": seedvr_manager.initialized,
                "model_type": getattr(seedvr_manager, 'model_type', None),
                "cuda_available": torch.cuda.is_available(),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
            
    except Exception as e:
        print(f"❌ Handler error: {str(e)}")
        traceback.print_exc()
        return {"error": f"Handler failed: {str(e)}"}


if __name__ == "__main__":
    print("🚀 Starting SeedVR RunPod Serverless Handler")
    runpod.serverless.start({"handler": handler})