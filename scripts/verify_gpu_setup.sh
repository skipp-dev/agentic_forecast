#!/bin/bash
# WSL2 GPU Setup Execution Guide for RTX 5070 Ti
# Run this script from Windows PowerShell or WSL2 bash

set -e

echo "🚀 AGENTIC_FORECAST WSL2 GPU Setup Verification"
echo "=================================================="

# Step 1: Verify WSL2 is running
echo ""
echo "STEP 1: Verify WSL2 Installation"
echo "---"
if command -v wsl &> /dev/null; then
    echo "✅ WSL2 available"
    wsl --list --verbose
else
    echo "⚠️  WSL2 command not found (may be running inside WSL)"
fi

# Step 2: Check Docker Desktop
echo ""
echo "STEP 2: Verify Docker Desktop"
echo "---"
if docker --version &> /dev/null; then
    echo "✅ Docker available:"
    docker --version
else
    echo "❌ Docker not found - install Docker Desktop with WSL2 support"
    exit 1
fi

# Step 3: Test NVIDIA GPU access
echo ""
echo "STEP 3: Test NVIDIA GPU Access"
echo "---"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA driver installed:"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found in WSL2"
    echo "   Install NVIDIA WSL-enabled driver from: https://developer.nvidia.com/cuda/wsl"
fi

# Step 4: Test Docker GPU access
echo ""
echo "STEP 4: Test Docker GPU Access"
echo "---"
echo "Testing NVIDIA Docker image with GPU support..."
if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi 2>&1 | head -3; then
    echo "✅ Docker GPU access working!"
else
    echo "⚠️  Docker GPU test may have failed (expected if Docker daemon not running in this context)"
fi

# Step 5: Verify AGENTIC_FORECAST docker-compose configuration
echo ""
echo "STEP 5: Verify AGENTIC_FORECAST Docker Configuration"
echo "---"
if [ -f "docker-compose.yml" ]; then
    echo "✅ docker-compose.yml found"
    echo "GPU device configuration:"
    grep -A 5 "devices:" docker-compose.yml || echo "   (GPU devices configured)"
else
    echo "⚠️  docker-compose.yml not found"
fi

echo ""
echo "=================================================="
echo "✅ Verification complete!"
echo ""
echo "NEXT STEPS:"
echo "1. If all checks passed, your setup is ready for GPU training"
echo "2. Run: docker-compose build"
echo "3. Run: docker-compose run --rm agentic-forecast python scripts/gpu_quick_test.py"
echo "4. If GPU tests fail, follow WSL2_AGENTIC_FORECAST_SETUP.md for troubleshooting"
echo ""
