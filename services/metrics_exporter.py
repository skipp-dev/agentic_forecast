# services/metrics_exporter.py
"""
Metrics exporter service for Prometheus integration.

Converts quality and guardrail JSON reports into Prometheus-compatible metrics.
Also exports forecast performance metrics from evaluation results.
"""

import json
import os
import time
from typing import Dict, Any

import numpy as np
from prometheus_client import CollectorRegistry, Gauge, generate_latest


QUALITY_REPORT_PATH = os.getenv(
    "QUALITY_REPORT_PATH", "data/metrics/quality_report_latest.json"
)
GUARDRAIL_DECISION_PATH = os.getenv(
    "GUARDRAIL_DECISION_PATH", "data/metrics/guardrail_decision_latest.json"
)
EVALUATION_RESULTS_PATH = os.getenv(
    "EVALUATION_RESULTS_PATH", "data/metrics/evaluation_results_baseline_latest.csv"
)
FORECAST_AGENT_OUTPUT_PATH = os.getenv(
    "FORECAST_AGENT_OUTPUT_PATH", "data/metrics/forecast_agent_output_latest.json"
)


def _load_json_safely(path: str) -> Dict[str, Any]:
    """Load JSON file safely; return {} if missing or invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def build_metrics_registry() -> CollectorRegistry:
    """
    Build a fresh Prometheus registry from the latest quality/guardrail JSONs.
    This is called once per /metrics request.
    """
    registry = CollectorRegistry()

    # Define gauges
    health_status = Gauge(
        "metrics_health_status",
        "Overall health of the latest evaluation run (1=passed, 0=failed)",
        registry=registry,
    )

    severity_level = Gauge(
        "metrics_health_severity_level",
        "Severity of metrics health (one-hot per level)",
        ["level"],
        registry=registry,
    )

    metric_quality = Gauge(
        "metric_quality_status",
        "Quality status per metric (ok/suspect/unreliable)",
        ["metric", "status"],
        registry=registry,
    )

    # Metric sanity checks
    metric_sanity_ok = Gauge(
        "metric_sanity_ok",
        "Overall sanity check status (1=passed, 0=failed)",
        registry=registry,
    )

    metric_sanity_issue_count = Gauge(
        "metric_sanity_issue_count",
        "Number of sanity issues per metric",
        ["metric"],
        registry=registry,
    )

    metric_sanity_severity = Gauge(
        "metric_sanity_severity_level",
        "Severity level of metric sanity issues (one-hot per level)",
        ["level"],
        registry=registry,
    )

    issue_count = Gauge(
        "metrics_issue_count",
        "Number of occurrences per issue type in latest run",
        ["type"],
        registry=registry,
    )

    can_rotate = Gauge(
        "guardrail_can_rotate_models",
        "Whether automatic model rotation is allowed (1=yes, 0=no)",
        registry=registry,
    )

    can_update_champions = Gauge(
        "guardrail_can_update_champions",
        "Whether champion promotion is allowed (1=yes, 0=no)",
        registry=registry,
    )

    guardrail_info = Gauge(
        "guardrail_decision_info",
        "Info gauge with guardrail reason encoded as label",
        ["reason"],
        registry=registry,
    )

    run_timestamp_seconds = Gauge(
        "metrics_run_timestamp_seconds",
        "Unix timestamp when latest evaluation run finished",
        registry=registry,
    )

    # Forecast performance metrics
    forecast_mae = Gauge(
        "forecast_mae",
        "Mean Absolute Error per symbol/horizon/model",
        ["symbol", "horizon", "model_family"],
        registry=registry,
    )

    forecast_mape = Gauge(
        "forecast_mape",
        "Mean Absolute Percentage Error per symbol/horizon/model",
        ["symbol", "horizon", "model_family"],
        registry=registry,
    )

    forecast_smape = Gauge(
        "forecast_smape",
        "Symmetric Mean Absolute Percentage Error per symbol/horizon/model",
        ["symbol", "horizon", "model_family"],
        registry=registry,
    )

    forecast_swase = Gauge(
        "forecast_swase",
        "Shock-Weighted Absolute Scaled Error per symbol/horizon/model",
        ["symbol", "horizon", "model_family"],
        registry=registry,
    )

    forecast_directional_accuracy = Gauge(
        "forecast_directional_accuracy",
        "Directional Accuracy per symbol/horizon/model",
        ["symbol", "horizon", "model_family"],
        registry=registry,
    )

    forecast_metrics_timestamp = Gauge(
        "forecast_metrics_run_timestamp_seconds",
        "Unix timestamp when latest forecast evaluation finished",
        registry=registry,
    )

    # Forecast Agent output metrics
    forecast_predicted_return = Gauge(
        "forecast_predicted_return",
        "Predicted return per symbol and horizon from forecast agent",
        ["symbol", "horizon"],
        registry=registry,
    )

    forecast_confidence_level = Gauge(
        "forecast_confidence_level",
        "Confidence level per symbol (0=low, 1=medium, 2=high)",
        ["symbol"],
        registry=registry,
    )

    forecast_guardrail_flag = Gauge(
        "forecast_guardrail_flag",
        "Guardrail flags per symbol (1=active, 0=inactive)",
        ["symbol", "flag"],
        registry=registry,
    )

    forecast_risk_model_confidence_comment = Gauge(
        "forecast_risk_model_confidence_comment",
        "Model confidence comment per symbol",
        ["symbol", "comment"],
        registry=registry,
    )

    forecast_risk_regime_comment = Gauge(
        "forecast_risk_regime_comment",
        "Regime comment per symbol",
        ["symbol", "comment"],
        registry=registry,
    )

    forecast_agent_timestamp = Gauge(
        "forecast_agent_run_timestamp_seconds",
        "Unix timestamp when latest forecast agent run finished",
        ["symbol"],
        registry=registry,
    )

    # Load JSONs
    quality = _load_json_safely(QUALITY_REPORT_PATH)
    guardrail = _load_json_safely(GUARDRAIL_DECISION_PATH)

    # Default behavior: assume failure if no quality report
    if not quality:
        # No file / broken JSON → treat as high severity failure
        health_status.set(0)
        for level in ("low", "medium", "high"):
            severity_level.labels(level=level).set(1 if level == "high" else 0)
        issue_count.labels(type="missing_quality_report").set(1)
        # run timestamp = now (best we can do)
        run_timestamp_seconds.set(time.time())
        # Guardrail defaults
        can_rotate.set(0)
        can_update_champions.set(0)
        guardrail_info.labels(reason="no_quality_report").set(1)
        return registry

    # Extract data from quality report
    checks = quality.get("checks", {})
    eval_metrics_check = checks.get("evaluation_metrics_quality", {})

    # 1) Health status & severity
    status_str = eval_metrics_check.get("status", "failed")
    severity_str = eval_metrics_check.get("severity", "high")

    health_status.set(1 if status_str == "passed" else 0)

    for level in ("low", "medium", "high"):
        severity_level.labels(level=level).set(1 if level == severity_str else 0)

    # 2) Metric quality
    metrics_quality = eval_metrics_check.get("metrics_quality", {})
    for metric_name, status in metrics_quality.items():
        for s in ("ok", "suspect", "unreliable"):
            metric_quality.labels(metric=metric_name, status=s).set(1 if s == status else 0)

    # 3) Issues
    issues = eval_metrics_check.get("issues", [])
    # Aggregate by type
    counts: Dict[str, int] = {}
    for issue in issues:
        issue_type = issue.get("type", "unknown")
        counts[issue_type] = counts.get(issue_type, 0) + 1

    for issue_type, count in counts.items():
        issue_count.labels(type=issue_type).set(count)

    # 3.5) Metric sanity checks
    # Aggregate sanity issues by metric
    sanity_issues_by_metric = {}
    sanity_severity = "low"

    for issue in issues:
        issue_type = issue.get("type", "")
        # Check if this is a sanity-related issue
        if any(keyword in issue_type for keyword in ["smape_", "swase_", "cross_metric"]):
            # Extract metric from issue type
            if "smape" in issue_type:
                metric = "smape"
            elif "swase" in issue_type:
                metric = "swase"
            else:
                metric = "cross_metric"

            if metric not in sanity_issues_by_metric:
                sanity_issues_by_metric[metric] = 0
            sanity_issues_by_metric[metric] += 1

            # Update severity based on issue type
            if "invalid_range" in issue_type or "negative" in issue_type or "invalid_values" in issue_type:
                sanity_severity = "high"
            elif sanity_severity != "high" and ("extreme" in issue_type or "low_variability" in issue_type):
                sanity_severity = "medium"

    # Set sanity gauges
    has_sanity_issues = len(sanity_issues_by_metric) > 0
    metric_sanity_ok.set(1 if not has_sanity_issues else 0)

    for metric in ["smape", "swase", "cross_metric"]:
        issue_count = sanity_issues_by_metric.get(metric, 0)
        metric_sanity_issue_count.labels(metric=metric).set(issue_count)

    for level in ("low", "medium", "high"):
        metric_sanity_severity.labels(level=level).set(1 if level == sanity_severity else 0)

    # 4) Guardrails
    if not guardrail:
        can_rotate.set(0)
        can_update_champions.set(0)
        guardrail_info.labels(reason="no_guardrail_decision").set(1)
    else:
        can_rotate.set(1 if guardrail.get("can_rotate_models", False) else 0)
        can_update_champions.set(1 if guardrail.get("can_update_champions", False) else 0)

        reason = guardrail.get("reason", "no_reason_provided")
        # Keep reason reasonably short and sanitized
        guardrail_info.labels(reason=reason[:200]).set(1)

    # 5) Run timestamp
    # If provided as ISO8601, parse it; if not, just use time.time()
    run_ts_str = quality.get("timestamp")
    if run_ts_str:
        try:
            # Very rough: in production, use datetime.fromisoformat or dateutil
            import datetime

            dt = datetime.datetime.fromisoformat(run_ts_str.replace("Z", "+00:00"))
            run_timestamp_seconds.set(dt.timestamp())
        except Exception:
            run_timestamp_seconds.set(time.time())
    else:
        run_timestamp_seconds.set(time.time())

    # 6) Load and populate forecast performance metrics
    try:
        import pandas as pd
        eval_df = pd.read_csv(EVALUATION_RESULTS_PATH)

        # Ensure we have SWASE (add if missing)
        if 'swase' not in eval_df.columns:
            eval_df['swase'] = eval_df.get('mape', 1.0)  # Placeholder

        # Populate forecast metrics
        for _, row in eval_df.iterrows():
            symbol = str(row.get("symbol", "UNKNOWN"))
            horizon = str(row.get("target_horizon", "UNKNOWN"))
            model_family = str(row.get("model_type", "UNKNOWN"))

            # Set each metric
            mae_val = float(row.get("mae", float("nan")))
            mape_val = float(row.get("mape", float("nan")))
            smape_val = float(row.get("smape", float("nan")))
            swase_val = float(row.get("swase", float("nan")))
            da_val = float(row.get("directional_accuracy", float("nan")))

            if not np.isnan(mae_val):
                forecast_mae.labels(symbol=symbol, horizon=horizon, model_family=model_family).set(mae_val)
            if not np.isnan(mape_val):
                forecast_mape.labels(symbol=symbol, horizon=horizon, model_family=model_family).set(mape_val)
            if not np.isnan(smape_val):
                forecast_smape.labels(symbol=symbol, horizon=horizon, model_family=model_family).set(smape_val)
            if not np.isnan(swase_val):
                forecast_swase.labels(symbol=symbol, horizon=horizon, model_family=model_family).set(swase_val)
            if not np.isnan(da_val):
                forecast_directional_accuracy.labels(symbol=symbol, horizon=horizon, model_family=model_family).set(da_val)

        # Set forecast metrics timestamp
        if "evaluation_timestamp" in eval_df.columns and not eval_df.empty:
            # Try to parse the latest timestamp
            try:
                latest_ts = pd.to_datetime(eval_df["evaluation_timestamp"]).max()
                forecast_metrics_timestamp.set(latest_ts.timestamp())
            except Exception:
                forecast_metrics_timestamp.set(time.time())
        else:
            forecast_metrics_timestamp.set(time.time())

    except Exception as e:
        # If evaluation file doesn't exist or can't be read, just skip forecast metrics
        print(f"Warning: Could not load forecast metrics: {e}")
        pass

    # 7) Load and populate forecast agent output metrics
    try:
        forecast_agent_output = _load_json_safely(FORECAST_AGENT_OUTPUT_PATH)
        
        if forecast_agent_output:
            symbol = forecast_agent_output.get("symbol", "UNKNOWN")
            
            # Set confidence level (convert string to numeric)
            confidence_str = forecast_agent_output.get("risk_assessment", {}).get("model_confidence_comment", "")
            confidence_level = 0  # default to low
            if "high" in confidence_str.lower():
                confidence_level = 2
            elif "medium" in confidence_str.lower():
                confidence_level = 1
            # else remains 0 for low
            
            forecast_confidence_level.labels(symbol=symbol).set(confidence_level)
            
            # Set predicted returns per horizon
            horizon_forecasts = forecast_agent_output.get("horizon_forecasts", [])
            for forecast in horizon_forecasts:
                horizon = str(forecast.get("horizon", "unknown"))
                predicted_return = forecast.get("predicted_return", 0.0)
                forecast_predicted_return.labels(symbol=symbol, horizon=horizon).set(predicted_return)
            
            # Set guardrail flags
            guardrail_flags = forecast_agent_output.get("risk_assessment", {}).get("guardrail_flags", [])
            all_possible_flags = {
                "high_error_recently", "shock_regime_active", "data_drift_suspected",
                "missing_feature_values", "pipeline_failure_recently", "news_shock_active"
            }
            
            for flag in all_possible_flags:
                is_active = 1 if flag in guardrail_flags else 0
                forecast_guardrail_flag.labels(symbol=symbol, flag=flag).set(is_active)
            
            # Set risk comments (truncated to reasonable length for labels)
            model_comment = forecast_agent_output.get("risk_assessment", {}).get("model_confidence_comment", "")
            regime_comment = forecast_agent_output.get("risk_assessment", {}).get("regime_comment", "")
            
            # Truncate comments to prevent label bloat (Prometheus has limits)
            model_comment_truncated = model_comment[:200] + "..." if len(model_comment) > 200 else model_comment
            regime_comment_truncated = regime_comment[:200] + "..." if len(regime_comment) > 200 else regime_comment
            
            forecast_risk_model_confidence_comment.labels(
                symbol=symbol, comment=model_comment_truncated
            ).set(1)
            
            forecast_risk_regime_comment.labels(
                symbol=symbol, comment=regime_comment_truncated
            ).set(1)
            
            # Set timestamp (use current time if not available)
            forecast_agent_timestamp.labels(symbol=symbol).set(time.time())
            
    except Exception as e:
        print(f"Warning: Could not load forecast agent output metrics: {e}")
        pass
    
    # 8) Add LLM usage metrics
    try:
        from src.llm.llm_factory import get_llm_usage_stats
        llm_stats = get_llm_usage_stats()
        
        # Create dynamic gauges for LLM usage
        llm_calls = Gauge(
            "llm_role_calls_total",
            "Total number of LLM calls per role",
            ["role"],
            registry=registry,
        )
        
        llm_tokens = Gauge(
            "llm_role_tokens_total",
            "Total tokens used per LLM role",
            ["role"],
            registry=registry,
        )
        
        llm_cost = Gauge(
            "llm_role_cost_total",
            "Total estimated cost per LLM role",
            ["role"],
            registry=registry,
        )
        
        llm_errors = Gauge(
            "llm_role_errors_total",
            "Total errors per LLM role",
            ["role"],
            registry=registry,
        )
        
        llm_last_used = Gauge(
            "llm_role_last_used_timestamp",
            "Last used timestamp per LLM role",
            ["role"],
            registry=registry,
        )
        
        # Populate LLM metrics
        for role, stats in llm_stats.items():
            llm_calls.labels(role=role).set(stats['calls'])
            llm_tokens.labels(role=role).set(stats['total_tokens'])
            llm_cost.labels(role=role).set(stats['total_cost'])
            llm_errors.labels(role=role).set(stats['errors'])
            if stats['last_used']:
                llm_last_used.labels(role=role).set(stats['last_used'])
                
    except Exception as e:
        # If LLM stats are unavailable, just skip
        print(f"Warning: Could not load LLM usage metrics: {e}")
        pass

    return registry


def generate_metrics_text() -> bytes:
    """Build registry and return Prometheus text output."""
    registry = build_metrics_registry()
    return generate_latest(registry)


def save_forecast_agent_output(forecast_agent_output: Dict[str, Any]) -> None:
    """
    Save forecast agent output to JSON file for Prometheus metrics export.
    
    Args:
        forecast_agent_output: The output dictionary from ForecastAgent.interpret_forecasts()
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(FORECAST_AGENT_OUTPUT_PATH), exist_ok=True)
        
        # Add timestamp if not present
        if "timestamp" not in forecast_agent_output:
            forecast_agent_output["timestamp"] = time.time()
        
        # Save to file
        with open(FORECAST_AGENT_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(forecast_agent_output, f, indent=2, default=str)
            
        print(f"Forecast agent output saved to {FORECAST_AGENT_OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Error saving forecast agent output: {e}")
        raise