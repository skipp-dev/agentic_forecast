#!/usr/bin/env bash
# scripts/cleanup.sh
# Docker cache cleanup and optimization script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧹 Docker Cache Cleanup & Optimization${NC}"
echo -e "${BLUE}=====================================${NC}"

# Show current disk usage
echo -e "${YELLOW}📊 Current Docker disk usage:${NC}"
docker system df

echo ""

# Function to safely remove resources
safe_cleanup() {
    local command="$1"
    local description="$2"

    echo -e "${YELLOW}🗑️  ${description}...${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✅ ${description} completed${NC}"
    else
        echo -e "${RED}❌ ${description} failed${NC}"
    fi
    echo ""
}

# Stop all containers (optional)
read -p "Stop all running containers? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    safe_cleanup "docker stop \$(docker ps -q)" "Stopping all containers"
fi

# Remove stopped containers
safe_cleanup "docker container prune -f" "Removing stopped containers"

# Remove unused images
safe_cleanup "docker image prune -f" "Removing unused images"

# Clean build cache
safe_cleanup "docker builder prune -f" "Cleaning build cache"

# Remove unused volumes
safe_cleanup "docker volume prune -f" "Removing unused volumes"

# Remove unused networks
safe_cleanup "docker network prune -f" "Removing unused networks"

# Deep cleanup (optional)
read -p "Perform deep cleanup (remove all unused data)? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    safe_cleanup "docker system prune -a -f" "Deep system cleanup"
fi

# Show final disk usage
echo -e "${YELLOW}📊 Final Docker disk usage:${NC}"
docker system df

echo ""
echo -e "${GREEN}🎉 Cleanup completed!${NC}"
echo -e "${BLUE}💡 Regular maintenance tips:${NC}"
echo -e "   - Run this script weekly to prevent disk space issues"
echo -e "   - Monitor with: docker system df"
echo -e "   - Use .dockerignore to exclude unnecessary files"
echo -e "   - Enable BuildKit for better caching: export DOCKER_BUILDKIT=1"