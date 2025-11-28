# api.py - FastAPI application for the agentic forecast system
"""
FastAPI application providing REST endpoints for the agentic forecasting system.
Includes metrics export for Prometheus monitoring and health checks with quality assessment.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="Agentic Forecast API",
    description="REST API for the agentic forecasting system with metrics and health monitoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
QUALITY_REPORT_PATH = os.getenv(
    "QUALITY_REPORT_PATH", "data/metrics/quality_report_latest.json"
)
GUARDRAIL_DECISION_PATH = os.getenv(
    "GUARDRAIL_DECISION_PATH", "data/metrics/guardrail_decision_latest.json"
)


def _load_json_safely(path: str) -> Dict[str, Any]:
    """Load JSON file safely; return {} if missing or invalid."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


@app.get("/")
def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Agentic Forecast API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics (Prometheus format)"
        }
    }


@app.get("/health")
def health():
    """
    Simple health check.
    """
    return {
        "status": "ok",
        "service": {
            "name": "agentic_forecast",
            "version": "1.0.0"
        }
    }


@app.get("/metrics")
def metrics():
    """
    Prometheus /metrics endpoint.
    """
    try:
        from services.metrics_exporter import generate_metrics_text
        output = generate_metrics_text()
        return Response(content=output, media_type="text/plain; charset=utf-8")
    except Exception as e:
        error_output = f"""# Error generating metrics: {str(e)}
metrics_health_status 0
metrics_issue_count{{type="metrics_endpoint_error"}} 1
"""
        return Response(content=error_output, media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )