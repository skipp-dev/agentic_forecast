#!/usr/bin/env bash
# scripts/pre-download.sh
# Pre-download large packages to speed up builds

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📦 Pre-downloading Large Packages${NC}"
echo -e "${BLUE}=================================${NC}"

# Create cache directory
CACHE_DIR="${HOME}/.cache/docker-prebuild"
mkdir -p "${CACHE_DIR}"

# Large packages that take time to download
LARGE_PACKAGES=(
    "neuralforecast>=1.6.0"
    "statsforecast>=1.5.0"
    "shap>=0.41.0"
    "scipy>=1.10.0"
    "statsmodels>=0.14.0"
    "scikit-learn>=1.3.0"
    "pandas>=1.5.0"
    "numpy<2.0"
    "matplotlib>=3.7.0"
)

echo -e "${YELLOW}📥 Downloading large packages to cache...${NC}"

# Create a temporary requirements file for large packages
TEMP_REQ="${CACHE_DIR}/large_packages.txt"
printf '%s\n' "${LARGE_PACKAGES[@]}" > "${TEMP_REQ}"

echo -e "${BLUE}📋 Packages to pre-download:${NC}"
cat "${TEMP_REQ}"
echo ""

# Pre-download with pip cache
echo -e "${YELLOW}⏳ Downloading packages (this may take a few minutes)...${NC}"

if pip download \
    --dest "${CACHE_DIR}/wheels" \
    --no-cache-dir \
    --progress-bar off \
    -r "${TEMP_REQ}"; then
    echo -e "${GREEN}✅ Pre-download completed successfully${NC}"

    # Show downloaded packages
    echo -e "${BLUE}📦 Downloaded packages:${NC}"
    ls -la "${CACHE_DIR}/wheels/" | wc -l | xargs echo "Total files:"
    du -sh "${CACHE_DIR}/wheels/"

else
    echo -e "${RED}❌ Pre-download failed${NC}"
    exit 1
fi

# Create a cached requirements file for Docker builds
echo -e "${YELLOW}📝 Creating cached requirements file...${NC}"
cp requirements.txt "${CACHE_DIR}/requirements.cached.txt"

echo -e "${GREEN}🎉 Pre-download setup complete!${NC}"
echo -e "${BLUE}💡 Usage:${NC}"
echo -e "   - Cache location: ${CACHE_DIR}"
echo -e "   - Run this script before building to speed up Docker builds"
echo -e "   - The cache will be used automatically by Docker BuildKit"

# Optional: Clean old cache
echo -e "${YELLOW}🧹 Cleaning old cache files (7+ days old)...${NC}"
find "${CACHE_DIR}/wheels" -name "*.whl" -mtime +7 -delete 2>/dev/null || true

echo -e "${GREEN}✅ Maintenance completed${NC}"