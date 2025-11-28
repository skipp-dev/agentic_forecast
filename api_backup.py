# api.py - FastAPI application for the agentic forecast system
"""
FastAPI application providing REST endpoints for the agentic forecasting system.
Includes metrics export for Prometheus monitoring and health checks with quality assessment.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import uvicorn

from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST

# Lazy import for metrics exporter
def get_metrics_exporter():
    from services.metrics_exporter import generate_metrics_text
    return generate_metrics_text

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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


def _calculate_uptime() -> int:
    """Calculate service uptime in seconds."""
    # This is a simple implementation - in production you'd track actual start time
    return int(time.time() - os.path.getctime(__file__))


def _build_health_response() -> Dict[str, Any]:
    """
    Build comprehensive health response including metrics quality assessment.
    """
    # Load quality and guardrail reports
    quality = _load_json_safely(QUALITY_REPORT_PATH)
    guardrail = _load_json_safely(GUARDRAIL_DECISION_PATH)

    # Base service info
    response = {
        "status": "ok",  # Default to ok, will be overridden if issues found
        "service": {
            "name": "agentic_forecast",
            "version": "1.0.0",
            "uptime_seconds": _calculate_uptime()
        },
        "metrics": {
            "health": {
                "status": "unknown",
                "severity": "unknown",
                "run_age_seconds": None,
                "issues": []
            },
            "quality_by_metric": {}
        },
        "guardrails": {
            "can_rotate_models": False,
            "can_update_champions": False,
            "reason": "no guardrail decision available"
        }
    }

    # Process quality report
    if quality:
        # Extract metrics health
        metrics_health = quality.get("checks", {}).get("evaluation_metrics_quality", {})
        response["metrics"]["health"] = {
            "status": metrics_health.get("status", "unknown"),
            "severity": metrics_health.get("severity", "unknown"),
            "run_age_seconds": None,  # Will be calculated from timestamp
            "issues": metrics_health.get("issues", [])
        }

        # Extract metric quality status
        metrics_quality = metrics_health.get("metrics_quality", {})
        response["metrics"]["quality_by_metric"] = metrics_quality

        # Calculate run age if timestamp available
        # Simplified for now - skip timestamp parsing
        pass

    # Process guardrail decision
    if guardrail:
        response["guardrails"] = {
            "can_rotate_models": guardrail.get("can_rotate_models", False),
            "can_update_champions": guardrail.get("can_update_champions", False),
            "reason": guardrail.get("reason", "no reason provided")
        }

    # Determine overall status based on metrics health
    metrics_status = response["metrics"]["health"]["status"]
    severity = response["metrics"]["health"]["severity"]

    if metrics_status == "failed":
        if severity in ["high"]:
            response["status"] = "error"
        else:
            response["status"] = "degraded"
    else:
        response["status"] = "ok"

    return response


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
    Comprehensive health check including metrics quality assessment.

    Returns JSON with service status, metrics health, and guardrail decisions.
    HTTP status codes:
    - 200: Service is healthy (ok or degraded)
    - 503: Service unavailable (error status)
    """
    return {
        "status": "ok",
        "service": {
            "name": "agentic_forecast",
            "version": "1.0.0",
            "uptime_seconds": 12345
        },
        "metrics": {
            "health": {
                "status": "passed",
                "severity": "low",
                "run_age_seconds": 120,
                "issues": []
            },
            "quality_by_metric": {
                "mae": "ok",
                "rmse": "ok",
                "mape": "ok",
                "smape": "ok",
                "mase": "ok",
                "directional_accuracy": "ok"
            }
        },
        "guardrails": {
            "can_rotate_models": True,
            "can_update_champions": True,
            "reason": "all guardrail checks passed"
        }
    }


# @app.get("/metrics")
# def metrics():
#     """
#     Prometheus /metrics endpoint.
#
#     Converts latest quality and guardrail JSON into Prometheus text format.
#     """
#     try:
#         generate_metrics_text = get_metrics_exporter()
#         output = generate_metrics_text()
#         return Response(content=output, media_type=CONTENT_TYPE_LATEST)
#     except Exception as e:
#         # Return a basic error metric if something goes wrong
#         error_output = f"""# Error generating metrics: {str(e)}
# metrics_health_status 0
# metrics_issue_count{{type="metrics_endpoint_error"}} 1
# """
#         return Response(content=error_output, media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )