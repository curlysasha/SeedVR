# SeedVR RunPod Serverless Deployment Guide

## 🚀 Quick Deploy

### 1. Build Docker Image
```bash
# Build image (will download 7B model ~25GB)
docker build -t seedvr-serverless .

# Tag for Docker Hub
docker tag seedvr-serverless yourusername/seedvr-serverless:latest

# Push to Docker Hub
docker push yourusername/seedvr-serverless:latest
```

### 2. Create RunPod Serverless Endpoint
1. Go to [RunPod Serverless](https://www.runpod.io/serverless)
2. Click "New Endpoint"
3. Configure:
   - **Name**: `SeedVR Video Restoration`
   - **Docker Image**: `yourusername/seedvr-serverless:latest`
   - **GPU Type**: H100 SXM (recommended) or A100 80GB
   - **Timeout**: 600 seconds (10 minutes)
   - **Idle Timeout**: 300 seconds
   - **Memory**: 64GB minimum
   - **Workers**: 1-5 (based on demand)

### 3. Test Your Endpoint
```bash
python3 test_seedvr_endpoint.py \
  --endpoint-id YOUR_ENDPOINT_ID \
  --api-key YOUR_API_KEY \
  --action comprehensive \
  --create-test-video
```

## 📋 Configuration Details

### Model Information
- **Model**: SeedVR2-7B (highest quality)
- **Model Size**: ~25GB
- **Processing**: One-step video restoration
- **VRAM Requirements**: 25-35GB

### GPU Requirements
| Resolution | GPU Type | VRAM | Processing Time |
|------------|----------|------|-----------------|
| 720p       | H100 SXM | 25GB | 30-60s |
| 1080p      | H100 SXM | 35GB | 60-120s |
| 2K+        | H100 SXM | 40GB+ | 120-300s |

### Default Settings
- **Resolution**: 1280x720 (720p)
- **CFG Scale**: 1.0 (one-step optimal)
- **Sample Steps**: 1 (one-step)
- **Model**: SeedVR2-7B only

## 🔧 API Usage

### Basic Video Restoration
```python
import requests
import base64

# Read video
with open("input.mp4", "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode()

# API call
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "input": {
            "action": "restore",
            "video": video_b64,
            "res_h": 1280,
            "res_w": 720
        }
    }
)

result = response.json()
if result.get("success"):
    # Save restored video
    restored_video = base64.b64decode(result["videos"][0])
    with open("output.mp4", "wb") as f:
        f.write(restored_video)
```

### Health Check
```python
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={"input": {"action": "health"}}
)
```

## 🎯 Performance Optimization

### For Best Speed
- Use 720p resolution: `"res_h": 1280, "res_w": 720`
- Keep videos under 10 seconds
- Use H100 SXM GPUs

### For Best Quality
- Use 1080p resolution: `"res_h": 1920, "res_w": 1080` 
- Longer processing time acceptable
- Ensure sufficient GPU memory

### Memory Management
- Model auto-loads on first request (~30s)
- Subsequent requests are faster
- Cold start: 30-60s
- Warm requests: Processing time only

## 🐛 Troubleshooting

### Common Issues

#### "CUDA out of memory"
**Solution**: Reduce resolution or use shorter videos
```json
{
  "input": {
    "action": "restore",
    "video": "...",
    "res_h": 960,
    "res_w": 540
  }
}
```

#### "Model not initialized"
**Solution**: The model auto-initializes on first request. Wait 30-60s for cold start.

#### "Request timeout"
**Solution**: Increase timeout or reduce video complexity
```python
# Increase timeout in test script
python3 test_seedvr_endpoint.py --timeout 900  # 15 minutes
```

#### Slow Processing
**Solution**: Check GPU type and video specifications
- Ensure H100 SXM GPU
- Use compressed MP4 videos
- Avoid very high frame rates

### Debug Mode
Set environment variable for verbose logging:
```bash
export RUNPOD_DEBUG=1
```

## 📊 Cost Estimation

### RunPod Pricing (approximate)
- **H100 SXM**: $4.99/hour when active
- **A100 80GB**: $2.99/hour when active
- **Idle time**: Significantly cheaper

### Cost per Video (720p, 5-second video)
- **Processing time**: ~45 seconds
- **Cost**: ~$0.06 per video (H100)
- **Monthly estimate**: 1000 videos = ~$60

## 🔒 Security Notes

### Data Handling
- Videos processed in memory only
- No persistent storage of user data
- Automatic cleanup after processing
- Base64 encoding for secure transfer

### API Security
- RunPod API key required
- HTTPS-only endpoints
- No external dependencies during inference

## 📈 Scaling Configuration

### Auto-scaling Settings
```json
{
  "min_workers": 0,
  "max_workers": 5,
  "scaling_up_delay": 30,
  "scaling_down_delay": 300,
  "requests_per_worker": 1
}
```

### Load Balancing
- RunPod handles load balancing automatically
- Multiple workers spawn based on demand
- Cold start optimization through keep-alive

## 🚀 Advanced Features

### Custom Resolutions
```json
{
  "input": {
    "action": "restore",
    "video": "...",
    "res_h": 2160,  // 4K height
    "res_w": 3840   // 4K width
  }
}
```

### Batch Processing
Process multiple videos by making parallel API calls:
```python
import asyncio
import aiohttp

async def process_video(session, video_path):
    # Process single video
    pass

async def process_batch(video_paths):
    async with aiohttp.ClientSession() as session:
        tasks = [process_video(session, path) for path in video_paths]
        results = await asyncio.gather(*tasks)
    return results
```

## 📞 Support

### Documentation
- [RunPod Serverless Docs](https://docs.runpod.io/serverless)
- [SeedVR Paper](https://arxiv.org/abs/2506.05301)
- [API Documentation](./API_DOCUMENTATION.md)

### Testing
```bash
# Comprehensive test
python3 test_seedvr_endpoint.py \
  --endpoint-id YOUR_ID \
  --api-key YOUR_KEY \
  --action comprehensive

# Quick health check
python3 test_seedvr_endpoint.py \
  --endpoint-id YOUR_ID \
  --api-key YOUR_KEY \
  --action health
```

### Performance Monitoring
Monitor your endpoint performance:
1. RunPod Dashboard → Serverless → Your Endpoint
2. Check metrics: requests/minute, avg processing time, errors
3. Adjust worker count based on demand patterns

---

## 🎉 Ready to Deploy!

Your SeedVR serverless endpoint is now ready for deployment. The system will automatically:
- Download and initialize the 7B model
- Generate text embeddings
- Handle video restoration requests
- Scale based on demand
- Provide high-quality video enhancement

**Estimated Total Build Time**: 45-60 minutes (model download + dependencies)
**Ready for Production**: ✅