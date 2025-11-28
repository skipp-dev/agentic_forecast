#!/usr/bin/env python3
"""
WSL2 GPU Verification Script for RTX 5070 Ti
Run this after following the WSL2_GPU_SETUP_CHECKLIST.md
"""

import sys
import torch
import tensorflow as tf
import numpy as np

def check_pytorch_gpu():
    """Check PyTorch GPU availability"""
    print("=== PyTorch GPU Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {memory_gb:.1f} GB")
        # Test GPU computation
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        y = torch.matmul(x, x)
        memory_used = torch.cuda.memory_allocated(device) / 1024**2
        print(f"GPU memory used: {memory_used:.1f} MB")
        print("✅ PyTorch GPU test PASSED")
        return True
    else:
        print("❌ PyTorch GPU test FAILED")
        return False

def check_tensorflow_gpu():
    """Check TensorFlow GPU availability"""
    print("\n=== TensorFlow GPU Check ===")
    print(f"TensorFlow version: {tf.__version__}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU devices found: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
        print("✅ TensorFlow GPU test PASSED")
        return True
    else:
        print("❌ TensorFlow GPU test FAILED")
        return False

def test_gpu_performance():
    """Basic GPU performance test"""
    print("\n=== GPU Performance Test ===")
    if not torch.cuda.is_available():
        print("❌ Cannot test performance - no GPU available")
        return False

    import time
    device = torch.device('cuda')

    # Matrix multiplication test
    sizes = [1000, 2000, 5000]
    for size in sizes:
        x = torch.randn(size, size).to(device)
        start_time = time.time()
        y = torch.matmul(x, x)
        torch.cuda.synchronize()  # Wait for GPU to finish
        end_time = time.time()

        elapsed = end_time - start_time
        print(f"Matrix size {size}x{size}: {elapsed:.4f} seconds")
    print("✅ GPU performance test completed")
    return True

def main():
    """Main verification function"""
    print("🚀 WSL2 GPU Verification for RTX 5070 Ti")
    print("=" * 50)

    # Check NVIDIA drivers
    print("\n=== NVIDIA Driver Check ===")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_name = result.stdout.strip()
            print(f"✅ NVIDIA GPU detected: {gpu_name}")
        else:
            print("❌ nvidia-smi failed - check driver installation")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"❌ nvidia-smi check failed: {e}")

    # PyTorch checks
    pytorch_ok = check_pytorch_gpu()

    # TensorFlow checks
    tf_ok = check_tensorflow_gpu()

    # Performance test
    if pytorch_ok:
        test_gpu_performance()

    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)

    if pytorch_ok and tf_ok:
        print("🎉 SUCCESS: WSL2 GPU setup is working correctly!")
        print("Your RTX 5070 Ti is ready for ML workloads.")
        return 0
    else:
        print("⚠️  ISSUES DETECTED: Check WSL2_GPU_SETUP_CHECKLIST.md")
        print("Common issues:")
        print("  - Wrong NVIDIA driver (need WSL-enabled version)")
        print("  - WSL2 not properly configured")
        print("  - CUDA toolkit not installed in WSL2")
        return 1

if __name__ == "__main__":
    sys.exit(main())