#!/usr/bin/env bash
# scripts/optimize-build.sh
# Comprehensive Docker build optimization script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="agentic-forecast"
DOCKER_COMPOSE_FILE="docker-compose.yml"
DOCKER_COMPOSE_OVERRIDE="docker-compose.override.yml"
BUILDKIT_INLINE_CACHE=1
COMPRESSION_LEVEL=6

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "${PURPLE}🚀 $1${NC}"
    echo -e "${PURPLE}$(printf '%.0s=' {1..50})${NC}"
}

check_dependencies() {
    log_info "Checking dependencies..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi

    # Check Docker version for BuildKit support
    DOCKER_VERSION=$(docker --version | grep -oP 'Docker version \K[\d.]+')
    if [[ "$(printf '%s\n' "$DOCKER_VERSION" "18.06" | sort -V | head -n1)" == "18.06" ]]; then
        log_warning "Docker version $DOCKER_VERSION detected. BuildKit requires Docker 18.06+"
    fi

    log_success "Dependencies check passed"
}

setup_buildkit() {
    log_info "Setting up Docker BuildKit..."

    # Enable BuildKit
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1

    # Set BuildKit progress to plain for CI/CD
    if [[ -n "$CI" ]]; then
        export BUILDKIT_PROGRESS=plain
    fi

    log_success "BuildKit enabled"
}

optimize_docker_daemon() {
    log_info "Optimizing Docker daemon settings..."

    # Check if daemon.json exists
    if [[ ! -f "/etc/docker/daemon.json" ]]; then
        log_warning "Docker daemon.json not found. Creating optimized config..."

        sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "builder": {
        "gc": {
            "defaultKeepStorage": "20GB",
            "enabled": true
        }
    },
    "experimental": false,
    "features": {
        "buildkit": true
    },
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2"
}
EOF

        log_info "Restarting Docker daemon..."
        sudo systemctl restart docker

        log_success "Docker daemon optimized"
    else
        log_info "Docker daemon.json already exists"
    fi
}

pre_build_cleanup() {
    log_info "Performing pre-build cleanup..."

    # Remove dangling images
    docker image prune -f

    # Remove stopped containers
    docker container prune -f

    # Clean build cache (keep last 3 days)
    docker builder prune -f --filter type=exec.cachemount --filter unused-for=72h

    log_success "Pre-build cleanup completed"
}

build_with_optimization() {
    log_header "Building Docker Images with Optimizations"

    local start_time=$(date +%s)

    # Build arguments for optimization
    BUILD_ARGS=(
        --build-arg BUILDKIT_INLINE_CACHE="${BUILDKIT_INLINE_CACHE}"
        --build-arg COMPRESSION_LEVEL="${COMPRESSION_LEVEL}"
        --progress=auto
        --no-cache=false
    )

    # Add cache mounts for pip
    BUILD_ARGS+=(
        --mount=type=cache,target=/root/.cache/pip
        --mount=type=cache,target=/opt/venv
    )

    log_info "Starting optimized build..."
    log_info "Build arguments: ${BUILD_ARGS[*]}"

    if docker-compose -f "${DOCKER_COMPOSE_FILE}" -f "${DOCKER_COMPOSE_OVERRIDE}" build "${BUILD_ARGS[@]}"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        log_success "Build completed successfully in ${duration} seconds"

        # Show build cache info
        log_info "Build cache information:"
        docker builder df

        return 0
    else
        log_error "Build failed"
        return 1
    fi
}

post_build_analysis() {
    log_info "Performing post-build analysis..."

    # Show image sizes
    echo -e "${BLUE}📊 Image Sizes:${NC}"
    docker images "${PROJECT_NAME}*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

    # Show build cache usage
    echo -e "\n${BLUE}📈 Build Cache Usage:${NC}"
    docker builder df

    # Check for large layers
    echo -e "\n${BLUE}🔍 Large Layers Check:${NC}"
    docker history "${PROJECT_NAME}:latest" --format "{{.Size}}\t{{.CreatedBy}}" | head -10

    log_success "Analysis completed"
}

run_health_checks() {
    log_info "Running health checks..."

    if docker-compose -f "${DOCKER_COMPOSE_FILE}" -f "${DOCKER_COMPOSE_OVERRIDE}" ps | grep -q "Up"; then
        log_success "All containers are running"

        # Test basic connectivity
        if curl -f http://localhost:8000/health 2>/dev/null; then
            log_success "Main service health check passed"
        else
            log_warning "Main service health check failed"
        fi

        if curl -f http://localhost:9091/-/healthy 2>/dev/null; then
            log_success "Prometheus health check passed"
        else
            log_warning "Prometheus health check failed"
        fi

        if curl -f http://localhost:3001/api/health 2>/dev/null; then
            log_success "Grafana health check passed"
        else
            log_warning "Grafana health check failed"
        fi

    else
        log_error "Some containers are not running"
        docker-compose -f "${DOCKER_COMPOSE_FILE}" -f "${DOCKER_COMPOSE_OVERRIDE}" ps
        return 1
    fi
}

main() {
    log_header "Docker Build Optimization Script"
    echo -e "${BLUE}Project: ${PROJECT_NAME}${NC}"
    echo -e "${BLUE}Timestamp: $(date)${NC}"
    echo ""

    check_dependencies
    setup_buildkit
    optimize_docker_daemon
    pre_build_cleanup

    if build_with_optimization; then
        post_build_analysis
        run_health_checks

        log_success "🎉 Build optimization completed successfully!"
        echo -e "${GREEN}💡 Tips for maintaining fast builds:${NC}"
        echo -e "   • Run this script regularly to keep builds optimized"
        echo -e "   • Use 'docker system df' to monitor disk usage"
        echo -e "   • Clean build cache monthly: 'docker builder prune -a'"
        echo -e "   • Pre-download large packages with scripts/pre-download.sh"

    else
        log_error "Build optimization failed"
        exit 1
    fi
}

# Run main function
main "$@"