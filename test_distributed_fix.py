#!/usr/bin/env python3
"""
Test script to verify distributed initialization fix works in serverless mode
"""

import os
import sys

# Note: serverless mode is now default, no need to set environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['RUNPOD_SERVERLESS'] = '1'  # Now defaults to '1'

print("🧪 Testing SeedVR RunPod Serverless (DEFAULT MODE)...")
print(f"RUNPOD_SERVERLESS: {os.environ.get('RUNPOD_SERVERLESS', '1')} (default)")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

try:
    # Import handler and test SeedVRManager initialization
    from handler import SeedVRManager
    
    print("✅ Handler import successful")
    
    # Test manager creation
    manager = SeedVRManager()
    print("✅ SeedVRManager creation successful")
    
    # Test device getter
    device = manager.get_device()
    print(f"✅ Device getter successful: {device}")
    
    # Test model initialization (this is where distributed error would occur)
    print("🔄 Testing model initialization...")
    success = manager.initialize_model(
        model_type="seedvr2_7b", 
        model_variant="normal", 
        sp_size=1
    )
    
    if success:
        print("✅ Model initialization successful - distributed fix working!")
        
        # Test a simple health check
        print("🔄 Testing handler health check...")
        from handler import handler
        
        result = handler({
            "input": {
                "action": "health"
            }
        })
        
        print(f"✅ Health check result: {result}")
        
        if result.get("status") == "healthy":
            print("🎉 ALL TESTS PASSED - Distributed fix is working correctly!")
        else:
            print("⚠️ Health check returned unexpected result")
            
    else:
        print("❌ Model initialization failed")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This is expected if not all dependencies are available")
    
except Exception as e:
    if "MASTER_ADDR" in str(e):
        print(f"❌ DISTRIBUTED ERROR STILL PRESENT: {e}")
        print("The fix needs more work!")
    else:
        print(f"❌ Other error (not distributed): {e}")
        import traceback
        traceback.print_exc()

print("\n📊 Test Summary:")
print("- If you see 'DISTRIBUTED ERROR STILL PRESENT', the fix needs more work")
print("- If you see other errors, they might be due to missing dependencies (OK for testing)")
print("- If you see 'ALL TESTS PASSED', the distributed fix is working!")