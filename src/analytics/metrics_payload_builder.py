# src/analytics/metrics_payload_builder.py
import json
from pathlib import Path
from typing import Dict, Any


def load_health_report(path: str) -> Dict[str, Any]:
    """Load the existing health/metrics JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_metrics_payload_from_health(health: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt the existing health report into the metrics_payload format
    expected by LLMAnalyticsExplainerAgent.
    """
    run_meta = health.get("run_metadata", {})
    model_perf = health.get("model_performance", {})
    cross_asset = health.get("cross_asset_v2", {})
    metric_sanity = health.get("metric_sanity", {})

    # Map to expected structure
    metrics_payload = {
        "run_metadata": {
            "run_type": "DAILY",
            "date": run_meta.get("evaluated_at", "unknown").split("T")[0] if "evaluated_at" in run_meta else "unknown",
            "universe_size": len(run_meta.get("symbols", [])),
            "cross_asset_v2_enabled": cross_asset.get("status") == "enabled",
        },
        "metrics_global": model_perf.get("metrics", {}),
        "per_symbol_metrics": [],  # Not available in current health report
        "regime_metrics": {
            "peer_shock_flag": cross_asset.get("by_regime", {})
        },
        "feature_importance": {
            "overall": [{"name": f, "importance": 0} for f in cross_asset.get("top_features", [])],
            "shock_regime": [],  # Not specified
        },
        "guardrail_summary": {
            "status": metric_sanity.get("severity", "unknown"),
            "issues": metric_sanity.get("key_findings", []),
        },
    }
    return metrics_payload