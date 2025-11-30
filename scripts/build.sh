#!/usr/bin/env bash
# scripts/build.sh
# Optimized Docker build script with caching and monitoring

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="agentic_forecast"
IMAGE_NAME="${PROJECT_NAME}-ib-forecast"
BUILDKIT_PROGRESS=${BUILDKIT_PROGRESS:-plain}

# Enable BuildKit
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS
export BUILDKIT_INLINE_CACHE=1

echo -e "${BLUE}🚀 Starting optimized Docker build for ${PROJECT_NAME}${NC}"
echo -e "${BLUE}📊 BuildKit enabled with progress: ${BUILDKIT_PROGRESS}${NC}"

# Function to show build progress
show_progress() {
    echo -e "${YELLOW}⏳ Build started at $(date)${NC}"
}

# Function to show completion
show_completion() {
    local duration=$1
    echo -e "${GREEN}✅ Build completed successfully in ${duration} seconds${NC}"
    echo -e "${GREEN}📦 Image: ${IMAGE_NAME}:latest${NC}"
    echo -e "${GREEN}💾 Cache: BuildKit cache enabled${NC}"
}

# Pre-build cleanup (optional)
if [[ "${CLEAN_BUILD}" == "true" ]]; then
    echo -e "${YELLOW}🧹 Cleaning build cache...${NC}"
    docker builder prune -f
fi

# Start timing
start_time=$(date +%s)
show_progress

# Build with optimizations
echo -e "${BLUE}🏗️  Building with BuildKit optimizations...${NC}"

docker build \
    --target production \
    --cache-from ${IMAGE_NAME}:latest \
    --tag ${IMAGE_NAME}:latest \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --memory 4g \
    --progress=${BUILDKIT_PROGRESS} \
    .

# Calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
show_completion $duration

# Post-build information
echo -e "${BLUE}📊 Build Summary:${NC}"
echo -e "   Duration: ${duration}s"
echo -e "   Image size: $(docker images ${IMAGE_NAME}:latest --format 'table {{.Size}}' | tail -n 1)"
echo -e "   Build cache: $(docker system df --format 'table {{.Type}}\t{{.TotalCount}}\t{{.Size}}' | grep 'Build Cache' | awk '{print $3}')"

# Optional: Run tests
if [[ "${RUN_TESTS}" == "true" ]]; then
    echo -e "${BLUE}🧪 Running container tests...${NC}"
    docker run --rm ${IMAGE_NAME}:latest python -c "import sys; print('✅ Python import test passed')"
fi

echo -e "${GREEN}🎉 Build process complete!${NC}"
echo -e "${YELLOW}💡 Tips:${NC}"
echo -e "   - Use 'docker buildx bake' for multi-platform builds"
echo -e "   - Run 'docker system df' to monitor disk usage"
echo -e "   - Use 'docker builder prune' to clean build cache periodically"