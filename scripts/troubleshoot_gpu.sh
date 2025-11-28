#!/bin/bash
# Quick troubleshooting script for WSL2 GPU issues
# Run this if GPU is not being detected

echo "🔧 WSL2 GPU Troubleshooting Guide"
echo "=================================="

# Check 1: Windows NVIDIA Driver
echo ""
echo "CHECK 1: Windows NVIDIA Driver with WSL Support"
echo "---"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi is available"
    nvidia-smi --version
    echo ""
    echo "GPU Info:"
    nvidia-smi
else
    echo "❌ NVIDIA driver not found on Windows"
    echo "   SOLUTION: Download WSL-enabled driver from https://developer.nvidia.com/cuda/wsl"
    exit 1
fi

# Check 2: WSL2 Configuration
echo ""
echo "CHECK 2: WSL2 Configuration (.wslconfig)"
echo "---"
if [ -f "$USERPROFILE/.wslconfig" ]; then
    echo "✅ .wslconfig exists:"
    cat "$USERPROFILE/.wslconfig"
else
    echo "⚠️  .wslconfig not found"
    echo "   SOLUTION: Create C:\\Users\\<username>\\.wslconfig with:"
    echo "   [wsl2]"
    echo "   memory=24GB"
    echo "   processors=8"
    echo "   swap=8GB"
fi

# Check 3: Docker Desktop WSL Integration
echo ""
echo "CHECK 3: Docker Desktop WSL2 Integration"
echo "---"
if docker info 2>&1 | grep -q "WSL"; then
    echo "✅ Docker is using WSL2"
    docker info | grep -A 2 "OSType"
else
    echo "⚠️  Docker may not be using WSL2"
    echo "   SOLUTION: In Docker Desktop Settings:"
    echo "   - Check 'Use WSL 2 based engine'"
    echo "   - Enable Ubuntu in 'Resources > WSL Integration'"
fi

# Check 4: CUDA in WSL
echo ""
echo "CHECK 4: CUDA in WSL2"
echo "---"
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA toolkit is installed:"
    nvcc --version
else
    echo "⚠️  CUDA toolkit not found in WSL2"
    echo "   This is OK if using Docker containers with CUDA"
fi

# Check 5: Test Docker GPU Access
echo ""
echo "CHECK 5: Docker GPU Access"
echo "---"
if docker run --rm --gpus all ubuntu nvidia-smi 2>&1 | grep -q "NVIDIA"; then
    echo "✅ Docker has GPU access"
else
    echo "⚠️  Docker GPU access test inconclusive"
    echo "   SOLUTION: Ensure docker daemon is running and restart Docker"
fi

# Recommendations
echo ""
echo "=================================="
echo "🎯 RECOMMENDED NEXT STEPS:"
echo ""
echo "1. If NVIDIA driver is missing:"
echo "   → Download from: https://developer.nvidia.com/cuda/wsl"
echo ""
echo "2. If .wslconfig is missing:"
echo "   → Create file and increase WSL2 memory"
echo "   → Run: wsl --shutdown && wsl"
echo ""
echo "3. If Docker WSL integration is broken:"
echo "   → Restart Docker Desktop"
echo "   → Check Settings > Resources > WSL Integration"
echo ""
echo "4. After fixes, verify with:"
echo "   docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi"
echo ""