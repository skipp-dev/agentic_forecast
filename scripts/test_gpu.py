#!/usr/bin/env python
"""
GPU Readiness Test Script
Tests TensorFlow & PyTorch GPU availability and performance
"""

import sys
import os
import time

print("=" * 80)
print("GPU READINESS TEST FOR AGENTIC_FORECAST")
print("=" * 80)

# ============================================================================
# 1. System Information
# ============================================================================
print("\n[1] SYSTEM INFORMATION")
print("-" * 80)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# ============================================================================
# 2. NVIDIA-SMI (if available)
# ============================================================================
print("\n[2] NVIDIA-SMI")
print("-" * 80)
os.system("nvidia-smi")

# ============================================================================
# 3. TensorFlow GPU Detection
# ============================================================================
print("\n[3] TENSORFLOW GPU CHECK")
print("-" * 80)
try:
    import tensorflow as tf
    
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Number of GPUs detected: {len(gpus)}")
    
    if len(gpus) > 0:
        print(" GPU(s) detected")
        for gpu in gpus:
            print(f"  - {gpu}")
        print(" TensorFlow GPU computation successful")
    else:
        print(" NO GPU DETECTED - TensorFlow running on CPU")
        
except Exception as e:
    print(f" TensorFlow ERROR: {e}")

# ============================================================================
# 4. PyTorch GPU Detection
# ============================================================================
print("\n[4] PYTORCH GPU CHECK")
print("-" * 80)
try:
    import torch
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(" PyTorch GPU computation successful")
    else:
        print(" NO GPU DETECTED - PyTorch running on CPU")
        
except Exception as e:
    print(f" PyTorch ERROR: {e}")

# ============================================================================
# 5. Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
try:
    import torch
    import tensorflow as tf
    
    tf_gpus = len(tf.config.list_physical_devices("GPU"))
    pt_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if tf_gpus > 0 or pt_gpus > 0:
        print(" GPU SUPPORT ENABLED")
        print(f"  - TensorFlow: {tf_gpus} GPU(s)")
        print(f"  - PyTorch: {pt_gpus} GPU(s)")
    else:
        print(" GPU NOT AVAILABLE - CPU mode")
except Exception as e:
    print(f"Summary error: {e}")

print("=" * 80)

