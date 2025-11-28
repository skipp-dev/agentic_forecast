#!/bin/bash

# IB Forecast Production Deployment Script
# This script sets up the complete production environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOYMENT_DIR="$SCRIPT_DIR"

echo "🚀 Starting IB Forecast Production Deployment"
echo "Project root: $PROJECT_ROOT"
echo "Deployment dir: $DEPLOYMENT_DIR"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Generate SSL certificates
echo "🔐 Generating SSL certificates..."
cd "$DEPLOYMENT_DIR/ssl"
if [ ! -f cert.pem ] || [ ! -f key.pem ]; then
    ./generate-certificates.sh
else
    echo "SSL certificates already exist, skipping generation"
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/reports"
mkdir -p "$PROJECT_ROOT/logs"

# Set environment variables
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-AGENTIC_FORECAST_2024}"
export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"

echo "🔧 Setting up environment variables..."
cat > "$PROJECT_ROOT/.env" << EOF
# Database
POSTGRES_PASSWORD=$POSTGRES_PASSWORD

# Grafana
GRAFANA_PASSWORD=$GRAFANA_PASSWORD

# Redis
REDIS_PASSWORD=

# Application
ENV=production
LOG_LEVEL=INFO
EOF

# Build and start services
echo "🐳 Building and starting Docker services..."
cd "$DEPLOYMENT_DIR"

# Stop any existing containers
docker-compose down || true

# Build services
echo "Building services..."
docker-compose build --parallel

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
MAX_ATTEMPTS=30
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "Health check attempt $ATTEMPT/$MAX_ATTEMPTS..."

    # Check if all services are running
    RUNNING_SERVICES=$(docker-compose ps --services --filter "status=running" | wc -l)
    TOTAL_SERVICES=$(docker-compose ps --services | wc -l)

    if [ "$RUNNING_SERVICES" -eq "$TOTAL_SERVICES" ]; then
        echo "✅ All services are running!"
        break
    else
        echo "⚠️  $RUNNING_SERVICES/$TOTAL_SERVICES services running, waiting..."
        sleep 10
        ATTEMPT=$((ATTEMPT + 1))
    fi
done

if [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
    echo "❌ Services failed to start properly"
    docker-compose logs
    exit 1
fi

# Run database migrations/initialization
echo "🗄️  Initializing database..."
sleep 10  # Wait for PostgreSQL to be ready

# Check if database is accessible
docker-compose exec -T postgres pg_isready -U ib_user -d AGENTIC_FORECAST || {
    echo "❌ Database not ready"
    exit 1
}

echo "✅ Database is ready"

# Display service information
echo ""
echo "🎉 IB Forecast Production Deployment Complete!"
echo ""
echo "📊 Service Endpoints:"
echo "  • API Gateway: https://localhost/api/"
echo "  • Grafana Dashboard: https://localhost/grafana/ (admin/$GRAFANA_PASSWORD)"
echo "  • Kibana Logs: https://localhost/kibana/"
echo "  • Prometheus Metrics: http://localhost:8080/prometheus/"
echo ""
echo "🔧 Management Commands:"
echo "  • View logs: docker-compose logs -f [service-name]"
echo "  • Stop services: docker-compose down"
echo "  • Restart service: docker-compose restart [service-name]"
echo "  • Scale service: docker-compose up -d --scale [service-name]=N"
echo ""
echo "📈 Monitoring:"
echo "  • System Health: https://localhost/health"
echo "  • Metrics: https://localhost/metrics"
echo ""
echo "⚠️  Remember to:"
echo "  • Update SSL certificates for production use"
echo "  • Configure proper authentication and authorization"
echo "  • Set up backups for databases and models"
echo "  • Configure log rotation and retention"
echo "  • Set up monitoring alerts"
echo ""
echo "Happy forecasting! 📈"
