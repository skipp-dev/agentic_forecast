#!/usr/bin/env python3
"""
Quick GPU Test for AGENTIC_FORECAST Container
Run with: docker-compose run --rm agentic-forecast python scripts/gpu_quick_test.py
"""

import torch
import tensorflow as tf
import sys

def main():
    print("🚀 AGENTIC_FORECAST GPU Quick Test")
    print("=" * 30)

    # PyTorch test
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Quick computation test
        x = torch.randn(100, 100, device='cuda')
        y = torch.randn(100, 100, device='cuda')
        z = x @ y
        print("✅ PyTorch GPU computation successful")
    else:
        print("❌ PyTorch GPU not available")

    # TensorFlow test
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow GPUs: {len(gpus)}")
    if gpus:
        print("✅ TensorFlow GPU available")
    else:
        print("❌ TensorFlow GPU not available")

    # Overall status
    gpu_ready = torch.cuda.is_available() and len(gpus) > 0
    print(f"\n🎯 GPU Ready: {'✅ YES' if gpu_ready else '❌ NO'}")

    return 0 if gpu_ready else 1

if __name__ == "__main__":
    sys.exit(main())
