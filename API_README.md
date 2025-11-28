# Agentic Forecast API

FastAPI-based REST API for the agentic forecasting system with comprehensive metrics and health monitoring.

## Features

- **Health Monitoring**: `/health` endpoint with quality assessment and guardrail status
- **Metrics Export**: `/metrics` endpoint for Prometheus-compatible monitoring
- **Quality Integration**: Real-time metrics quality assessment
- **Guardrail Monitoring**: Risk assessment decision tracking
- **Production Ready**: Error handling, logging, and graceful degradation

## Endpoints

### GET /
Basic API information and available endpoints.

**Response:**
```json
{
  "message": "Agentic Forecast API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "metrics": "/metrics (Prometheus format)"
  }
}
```

### GET /health
Comprehensive health check including service status and metrics quality assessment.

**Response:**
```json
{
  "status": "ok|warning|error",
  "service": {
    "name": "agentic_forecast",
    "version": "1.0.0"
  },
  "metrics_quality": {
    "status": "ok|warning|error",
    "last_updated": "2024-01-01T12:00:00Z",
    "issues": []
  },
  "guardrail_status": {
    "can_rotate_models": true,
    "can_update_champions": true,
    "last_decision": "2024-01-01T12:00:00Z"
  }
}
```

### GET /metrics
Prometheus-compatible metrics export for monitoring and alerting.

**Response:** Plain text in Prometheus format
```
# HELP metrics_health_status Overall health status of metrics system
# TYPE metrics_health_status gauge
metrics_health_status 1

# HELP quality_score Overall quality score
# TYPE quality_score gauge
quality_score 0.95
```

## Testing

### Using TestClient (Recommended)
```python
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

# Test endpoints
response = client.get("/")
print(response.json())

response = client.get("/health")
print(response.json())

response = client.get("/metrics")
print(response.text)
```

### Using Python Requests
```python
import requests

# Assuming server is running on localhost:8000
base_url = "http://localhost:8000"

response = requests.get(f"{base_url}/")
print(response.json())

response = requests.get(f"{base_url}/health")
print(response.json())

response = requests.get(f"{base_url}/metrics")
print(response.text)
```

### Using cURL
```bash
# Root endpoint
curl http://localhost:8000/

# Health endpoint
curl http://localhost:8000/health

# Metrics endpoint
curl http://localhost:8000/metrics
```

## Configuration

The API uses environment variables for configuration:

- `QUALITY_REPORT_PATH`: Path to quality report JSON (default: `data/metrics/quality_report_latest.json`)
- `GUARDRAIL_DECISION_PATH`: Path to guardrail decision JSON (default: `data/metrics/guardrail_decision_latest.json`)
- `HOST`: Server host (default: `127.0.0.1`)
- `PORT`: Server port (default: `8000`)

## Running the Server

### Development
```bash
# Using uvicorn directly
python -m uvicorn api:app --reload

# Using the production script
python run_api.py
```

### Production
```bash
# Set environment variables
export HOST=0.0.0.0
export PORT=8000

# Run the server
python run_api.py
```

## Integration with LangGraph

The API integrates with the LangGraph system through the `HealthAgent` which:

1. Calls the `/health` endpoint to assess system health
2. Provides health summaries for decision routing
3. Enables automated health monitoring in the forecasting pipeline

## Monitoring Setup

### Prometheus Configuration
Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'agentic_forecast'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Health Check Integration
Configure your load balancer or monitoring system to check the `/health` endpoint:

- `200 OK` with `status: "ok"` = Healthy
- `200 OK` with `status: "warning"` = Degraded but operational
- `503 Service Unavailable` with `status: "error"` = Unhealthy

## Dependencies

- fastapi
- uvicorn
- prometheus-client
- requests (for HealthAgent)

## Error Handling

The API implements graceful error handling:

- Missing JSON files return empty objects (graceful degradation)
- Invalid JSON is handled safely
- Metrics endpoint errors return fallback Prometheus metrics
- All endpoints return appropriate HTTP status codes

## Files

- `api.py`: Main FastAPI application
- `services/metrics_exporter.py`: Prometheus metrics generation
- `agents/health_agent.py`: LangGraph health monitoring node
- `run_api.py`: Production server runner script