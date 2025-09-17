#!/usr/bin/env python3
"""
Download SeedVR2-7B model for Docker build
"""

import os
import sys

def download_seedvr_model():
    """Download SeedVR2-7B model from HuggingFace"""
    try:
        from huggingface_hub import snapshot_download
        
        print('🚀 Starting SeedVR2-7B model download...')
        
        # Download SeedVR2-7B model
        snapshot_download(
            repo_id='ByteDance-Seed/SeedVR2-7B',
            local_dir='./ckpts/',
            cache_dir='./cache/',
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=['*.pth', '*.safetensors', '*.json', '*.txt', '*.md'],
            ignore_patterns=['*.bin', '*.onnx']
        )
        
        print('✅ SeedVR2-7B model download completed!')
        
        # List downloaded files
        print('📁 Downloaded files:')
        for root, dirs, files in os.walk('./ckpts/'):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                print(f'  {file_path} ({file_size:.1f} MB)')
        
        # Check for required model file
        model_file = './ckpts/seedvr2_ema_7b.pth'
        if os.path.exists(model_file):
            size_gb = os.path.getsize(model_file) / 1024 / 1024 / 1024
            print(f'✅ Model file found: {model_file} ({size_gb:.1f} GB)')
            return True
        else:
            print(f'❌ Model file not found: {model_file}')
            return False
            
    except Exception as e:
        print(f'❌ Error downloading model: {str(e)}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = download_seedvr_model()
    if success:
        print('🎉 Model download completed successfully!')
        sys.exit(0)
    else:
        print('💥 Model download failed!')
        sys.exit(1)