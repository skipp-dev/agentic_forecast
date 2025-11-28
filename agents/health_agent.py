# agents/health_agent.py
"""
HealthAgent - Monitors system health and metrics quality.

Checks the /health endpoint and provides summaries for decision-making.
Integrates with LangGraph for automated health-based routing.
"""

import time
from typing import Any, Dict, Optional

import requests

from src.graphs.state import GraphState


def _fetch_health_json(health_url: str, timeout_seconds: float = 5.0) -> Dict[str, Any]:
    """
    Call the /health endpoint and return parsed JSON.
    On failure, return an empty dict to signal a hard problem.
    """
    try:
        resp = requests.get(health_url, timeout=timeout_seconds)
        # We accept both 200 and 503: 503 means "service says it's in ERROR"
        if resp.status_code not in (200, 503):
            # Unexpected status → treat as serious problem
            return {}
        return resp.json()
    except Exception:
        # Network / parsing error → treat as serious problem
        return {}


def _derive_health_status_from_body(body: Dict[str, Any], severity_threshold_for_error: str) -> str:
    """
    Map the /health JSON into 'ok' | 'degraded' | 'error'.
    severity_threshold_for_error defines how strict we are:
      - "high": only HIGH severity becomes 'error'
      - "medium": MEDIUM or HIGH becomes 'error'
    """
    # If body is empty we consider this a hard error
    if not body:
        return "error"

    top_status = body.get("status", "error")  # e.g. "ok", "degraded", "error"
    metrics_health = body.get("metrics", {}).get("health", {})
    severity = metrics_health.get("severity", "high")  # "low" | "medium" | "high"

    if top_status == "ok":
        return "ok"

    if top_status == "degraded":
        # 'degraded' from the service can still be soft or hard based on severity
        if severity_threshold_for_error == "medium" and severity in ("medium", "high"):
            return "error"
        if severity_threshold_for_error == "high" and severity == "high":
            return "error"
        return "degraded"

    # top_status == "error" or anything unknown
    return "error"


def _build_plain_summary(body: Dict[str, Any], health_status: str) -> str:
    """
    Build a simple, human-readable summary string using only the JSON.
    No LLM involved here – this is the safe fallback and baseline.
    """
    if not body:
        return (
            "Status: ERROR – /health endpoint unavailable or returned invalid data. "
            "Guardrails should assume metrics are unsafe until this is resolved."
        )

    service_name = body.get("service", {}).get("name", "forecast_service")
    status = body.get("status", "unknown").upper()

    metrics_health = body.get("metrics", {}).get("health", {})
    severity = metrics_health.get("severity", "unknown").upper()
    run_age_seconds = metrics_health.get("run_age_seconds", None)
    issues_list = metrics_health.get("issues", [])

    guardrails = body.get("guardrails", {})
    can_rotate = guardrails.get("can_rotate_models", False)
    can_update = guardrails.get("can_update_champions", False)
    reason = guardrails.get("reason", "no reason provided")

    # First line: status + severity
    parts = [
        f"Service: {service_name}. Status: {status} (derived: {health_status.upper()}). Severity: {severity}."
    ]

    # Second line: run age, if available
    if run_age_seconds is not None:
        parts.append(f"Latest evaluation run age: {int(run_age_seconds)} seconds.")

    # Third: guardrails
    parts.append(
        f"Guardrails: can_rotate_models={str(can_rotate).lower()}, "
        f"can_update_champions={str(can_update).lower()}. Reason: {reason}."
    )

    # Issues
    if issues_list:
        parts.append("Issues:")
        for issue in issues_list[:5]:  # limit to first 5
            parts.append(f"- {issue}")
    else:
        parts.append("Issues: none reported by QualityAgent.")

    return " ".join(parts)


def health_check_node(state: GraphState) -> GraphState:
    """
    LangGraph node:
    - Calls /health endpoint
    - Derives health_status ("ok" | "degraded" | "error")
    - Stores raw JSON + summary + guardrail flags into state
    """
    monitoring_cfg = state.config.get("monitoring", {})
    health_url = monitoring_cfg.get("health_endpoint_url", "http://forecast-api:8000/health")
    severity_threshold = monitoring_cfg.get("severity_threshold_for_error", "high")

    # 1. Fetch /health JSON
    body = _fetch_health_json(health_url)

    # 2. Derive internal health_status
    health_status = _derive_health_status_from_body(body, severity_threshold)

    # 3. Extract guardrail flags
    guardrails = body.get("guardrails", {}) if body else {}
    can_rotate = guardrails.get("can_rotate_models", False)
    can_update = guardrails.get("can_update_champions", False)

    # 4. Build a simple summary (we can later replace/augment this via LLM)
    summary = _build_plain_summary(body, health_status)

    # 5. Write into state
    state.health_raw = body
    state.health_status = health_status
    state.health_can_rotate_models = bool(can_rotate)
    state.health_can_update_champions = bool(can_update)
    state.health_summary = summary

    return state