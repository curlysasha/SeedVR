# SeedVR RunPod Serverless API Documentation

## Overview
SeedVR RunPod Serverless API provides video restoration capabilities using state-of-the-art diffusion transformer models. The API supports both SeedVR2-3B and SeedVR2-7B models for one-step video enhancement.

## Supported Actions

### 1. Health Check
Check the health status of the serverless instance.

**Request:**
```json
{
  "input": {
    "action": "health"
  }
}
```

**Response:**
```json
{
  "status": "healthy",
  "model_initialized": true,
  "model_type": "seedvr2_3b",
  "cuda_available": true,
  "gpu_memory": 85899345920
}
```

### 2. Initialize Model
Explicitly initialize a specific model variant.

**Request:**
```json
{
  "input": {
    "action": "initialize",
    "model_type": "seedvr2_7b",
    "model_variant": "normal",  // or "sharp" for enhanced detail
    "sp_size": 1  // sequence parallel size (1 for single GPU)
  }
}
```

**Response:**
```json
{
  "message": "Model seedvr2_7b (normal) initialized successfully",
  "model_type": "seedvr2_7b",
  "model_variant": "normal",
  "sp_size": 1,
  "ready": true
}
```

### 3. Media Restoration (Video/Image)
Process video or image for restoration/enhancement.

**Request:**
```json
{
  "input": {
    "action": "restore",
    "media": "base64_encoded_media_data",  // or "video"/"image" for backward compatibility
    "media_type": "auto",   // "auto", "video", "image" (default: "auto" - detects from data)
    "res_h": 1280,         // target height (default: 1280)
    "res_w": 720,          // target width (default: 720)
    "seed": 666,           // random seed (default: 666)
    "cfg_scale": 1.0,      // CFG scale (default: 1.0 for one-step)
    "cfg_rescale": 0.0,    // CFG rescale (default: 0.0)
    "sample_steps": 1,     // sampling steps (default: 1)
    "model_type": "seedvr2_7b",  // optional: auto-init if not initialized
    "model_variant": "normal",   // optional: "normal" or "sharp" (default: "normal")
    "sp_size": 1           // optional: sequence parallel size
  }
}
```

**Response:**
```json
{
  "success": true,
  "media": [
    "base64_encoded_restored_media_1"
  ],
  "videos": [...],  // Backward compatibility: same as "media"
  "media_type": "video",  // or "image" 
  "model_type": "seedvr2_7b",
  "model_variant": "normal",  // or "sharp"
  "resolution": "1280x720",
  "processing_info": {
    "seed": 666,
    "cfg_scale": 1.0,
    "sample_steps": 1,
    "sequence_parallel_size": 1
  }
}
```

## Model Variants

### SeedVR2-7B Model Variants

#### Normal Variant
- **Best for**: Balanced restoration with natural results
- **Memory**: ~25-35GB VRAM  
- **Speed**: 60-120 seconds per video
- **Checkpoint**: `seedvr2_ema_7b.pth`
- **Characteristics**: More conservative enhancement, preserves original characteristics

#### Sharp Variant
- **Best for**: Maximum detail and sharpness
- **Memory**: ~25-35GB VRAM
- **Speed**: 60-120 seconds per video  
- **Checkpoint**: `seedvr2_ema_7b_sharp.pth`
- **Characteristics**: Enhanced detail extraction, more aggressive sharpening

## Resolution Guidelines

### Recommended Resolutions
- **720p**: 1280x720 (default, balanced)
- **1080p**: 1920x1080 (requires more VRAM)
- **480p**: 854x480 (faster processing)

### Memory Requirements by Resolution
- **720p (1280x720)**: 1 H100-80G minimum
- **1080p (1920x1080)**: 1 H100-80G (tight), 4 H100 recommended
- **2K+ resolutions**: 4 H100-80G required

## Error Handling

### Common Errors
```json
{
  "error": "Model not initialized"
}
```

```json
{
  "error": "No video data provided"
}
```

```json
{
  "error": "Invalid base64 video data: ..."
}
```

```json
{
  "error": "Processing failed: CUDA out of memory"
}
```

## Usage Examples

### Python Client Examples

#### Video Processing
```python
import requests
import base64

# Read video file
with open("input_video.mp4", "rb") as f:
    video_data = f.read()
    video_b64 = base64.b64encode(video_data).decode('utf-8')

# API request
payload = {
    "input": {
        "action": "restore",
        "media": video_b64,
        "media_type": "video",  # or "auto" for automatic detection
        "res_h": 1280,
        "res_w": 720,
        "seed": 42
    }
}

response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json=payload
)

result = response.json()
if result.get("success"):
    # Decode restored video
    restored_video_b64 = result["media"][0]  # or result["videos"][0]
    restored_video_data = base64.b64decode(restored_video_b64)
    
    with open("restored_video.mp4", "wb") as f:
        f.write(restored_video_data)
    
    print("Video restored successfully!")
else:
    print(f"Error: {result.get('error')}")
```

#### Image Processing
```python
import requests
import base64

# Read image file
with open("input_image.jpg", "rb") as f:
    image_data = f.read()
    image_b64 = base64.b64encode(image_data).decode('utf-8')

# API request
payload = {
    "input": {
        "action": "restore",
        "media": image_b64,
        "media_type": "image",  # explicitly set as image
        "res_h": 1280,
        "res_w": 720,
        "seed": 42
    }
}

response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json=payload
)

result = response.json()
if result.get("success"):
    # Decode restored image
    restored_image_b64 = result["media"][0]
    restored_image_data = base64.b64decode(restored_image_b64)
    
    with open("restored_image.png", "wb") as f:
        f.write(restored_image_data)
    
    print(f"Image restored successfully! Type: {result.get('media_type')}")
else:
    print(f"Error: {result.get('error')}")
```

### cURL Example
```bash
# Health check
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "health"}}'

# Video restoration
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "restore",
      "video": "'$(base64 -w 0 input_video.mp4)'",
      "res_h": 1280,
      "res_w": 720
    }
  }'
```

## Performance Optimization Tips

### For Best Speed
- Use SeedVR2-3B model
- Keep resolution at 720p or lower
- Use `sample_steps: 1` (default)
- Use `cfg_scale: 1.0` (default)

### For Best Quality
- Use SeedVR2-7B model
- Use higher resolutions (1080p+)
- Ensure sufficient VRAM available

### Memory Management
- Process one video at a time
- Use smaller resolutions for longer videos
- Consider video duration vs resolution trade-offs

## Deployment Notes

### RunPod Configuration
- **GPU**: H100 recommended, A100 minimum
- **Timeout**: 300-600 seconds
- **Container Registry**: Docker Hub or custom registry
- **Scaling**: Auto-scale based on demand

### Model Files Required
- Model checkpoints: `ckpts/seedvr2_ema_3b.pth` and/or `ckpts/seedvr2_ema_7b.pth`
- Text embeddings: `pos_emb.pt`, `neg_emb.pt`
- Configuration files: `configs_3b/main.yaml`, `configs_7b/main.yaml`

### Environment Variables
- `RUNPOD_SERVERLESS=1`: Enables serverless mode
- `CUDA_VISIBLE_DEVICES=0`: GPU selection
- `NVIDIA_VISIBLE_DEVICES=all`: GPU access

## Limitations

### Current Limitations
- Single video processing per request
- MP4 format recommended for input
- Maximum video length depends on VRAM
- No real-time streaming support

### Model Limitations
- Not robust to very heavy degradations
- May oversharpen lightly degraded content
- Performance varies with motion complexity
- Best results on natural video content

## Troubleshooting

### Common Issues
1. **"CUDA out of memory"**: Reduce resolution or use smaller model
2. **"Model initialization failed"**: Check model files are present
3. **"Invalid video format"**: Ensure MP4 format with standard codecs
4. **Slow processing**: Consider using 3B model or lower resolution

### Debug Information
Enable verbose logging by setting environment variable:
```bash
export RUNPOD_DEBUG=1
```