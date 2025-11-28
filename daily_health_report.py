# Daily Health Report Node for LangGraph

from typing import TypedDict, Optional, Dict, Any
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

from llm_prompts import METRIC_SANITY_EXEC_STRUCTURED_PROMPT
from metric_sanity_explainer import call_reporting_llm


class GraphState(TypedDict, total=False):
    run_id: Optional[str]
    performance_summary_path: str
    metric_sanity_summary_path: str
    cross_asset_v2_analysis_path: str
    daily_report_path_json: str
    daily_report_path_md: str
    can_auto_promote_models: bool
    symbols: list
    horizons: list
    config: Dict[str, Any]


def _load_json_safely(path: str, default: Any = {}) -> Dict[str, Any]:
    """Load JSON file safely; return default if missing or invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _aggregate_performance_metrics(eval_csv_path: str) -> Dict[str, Any]:
    """Aggregate performance metrics from evaluation CSV."""
    try:
        df = pd.read_csv(eval_csv_path)

        # Calculate overall metrics
        metrics = {}
        for metric in ["mae", "rmse", "mape", "smape", "swase", "directional_accuracy"]:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    metrics[metric] = {
                        "mean": round(float(values.mean()), 3),
                        "std": round(float(values.std()), 3),
                        "min": round(float(values.min()), 3),
                        "max": round(float(values.max()), 3)
                    }

        # Calculate overall score (simple average of normalized metrics)
        # Lower is better for error metrics, higher is better for directional_accuracy
        if metrics:
            # Normalize each metric to 0-1 scale (higher = better)
            normalized_scores = []
            for metric, stats in metrics.items():
                if metric in ["mae", "rmse", "mape", "smape", "swase"]:
                    # For error metrics: lower is better, so invert
                    if stats["max"] > stats["min"]:
                        score = 1 - (stats["mean"] - stats["min"]) / (stats["max"] - stats["min"])
                        normalized_scores.append(score)
                elif metric == "directional_accuracy":
                    # Already 0-1 scale, higher is better
                    normalized_scores.append(stats["mean"])

            overall_score = round(float(np.mean(normalized_scores)), 3) if normalized_scores else 0.5
        else:
            overall_score = 0.5

        # Aggregate by model family
        per_model = []
        if "model_type" in df.columns:
            model_groups = df.groupby("model_type")
            for model_family, group in model_groups:
                model_metrics = {}
                for metric in ["mape", "smape", "directional_accuracy"]:
                    if metric in group.columns:
                        model_metrics[f"mean_{metric}"] = round(float(group[metric].mean()), 3)

                if model_metrics:
                    # Determine status based on performance
                    best_mape = df["mape"].min() if "mape" in df.columns else float('inf')
                    model_mape = model_metrics.get("mean_mape", float('inf'))

                    status = "candidate"
                    if model_mape <= best_mape * 1.05:  # Within 5% of best
                        status = "champion"

                    per_model.append({
                        "model_family": model_family,
                        **model_metrics,
                        "status": status
                    })

            # Sort by MAPE performance
            per_model.sort(key=lambda x: x.get("mean_mape", float('inf')))

        return {
            "overall_score": overall_score,
            "metrics": metrics,
            "per_model": per_model,
            "file_ref": eval_csv_path
        }

    except Exception as e:
        return {
            "overall_score": 0.0,
            "metrics": {},
            "per_model": [],
            "error": str(e),
            "file_ref": eval_csv_path
        }


def _analyze_cross_asset_v2(v2_analysis_path: str) -> Dict[str, Any]:
    """Analyze cross-asset V2 experiment results."""
    analysis = _load_json_safely(v2_analysis_path)

    if not analysis:
        return {
            "status": "unknown",
            "decision": "investigate",
            "summary": "V2 analysis not available",
            "overall_lift": {},
            "by_regime": {},
            "top_features": [],
            "file_ref": v2_analysis_path
        }

    # Extract metrics
    v2_off = analysis.get("metrics", {}).get("v2_off", {})
    v2_on = analysis.get("metrics", {}).get("v2_on", {})

    # Calculate overall lift
    overall_lift = {}
    if v2_off and v2_on:
        for metric in ["mae", "mape", "directional_accuracy"]:
            if metric in v2_off and metric in v2_on:
                off_val = v2_off[metric]
                on_val = v2_on[metric]

                if metric in ["mae", "mape"]:  # Lower is better
                    delta = off_val - on_val  # Positive = improvement
                else:  # directional_accuracy: higher is better
                    delta = on_val - off_val  # Positive = improvement

                overall_lift[f"delta_{metric}"] = round(delta, 4)

    # Extract regime-specific metrics
    regime_metrics = analysis.get("regime_metrics", {})
    by_regime = {}

    v2_off_regimes = regime_metrics.get("v2_off", {})
    v2_on_regimes = regime_metrics.get("v2_on", {})

    for regime in set(v2_off_regimes.keys()) | set(v2_on_regimes.keys()):
        regime_lift = {}
        off_metrics = v2_off_regimes.get(regime, {})
        on_metrics = v2_on_regimes.get(regime, {})

        for metric in ["mae", "mape", "directional_accuracy"]:
            if metric in off_metrics and metric in on_metrics:
                off_val = off_metrics[metric]
                on_val = on_metrics[metric]

                if metric in ["mae", "mape"]:  # Lower is better
                    delta = off_val - on_val
                else:  # Higher is better
                    delta = on_val - off_val

                regime_lift[f"delta_{metric}"] = round(delta, 4)

        if regime_lift:
            by_regime[f"peer_shock_flag={regime}"] = regime_lift

    # Determine decision
    has_improvement = any(lift > 0 for lift in overall_lift.values() if isinstance(lift, (int, float)))
    decision = "keep_enabled" if has_improvement else "rollback"

    # Extract top features
    v2_coverage = analysis.get("v2_coverage", {})
    top_features = [feature for feature, coverage in v2_coverage.items()
                   if isinstance(coverage, (int, float)) and coverage > 0.8][:4]

    return {
        "status": "enabled" if analysis else "disabled",
        "decision": decision,
        "summary": f"V2 {'improves' if has_improvement else 'does not improve'} overall performance.",
        "overall_lift": overall_lift,
        "by_regime": by_regime,
        "top_features": top_features,
        "file_ref": v2_analysis_path
    }


def _make_auto_promotion_decision(metric_sanity: Dict[str, Any], performance: Dict[str, Any]) -> Dict[str, Any]:
    """Make decision about auto-promotion based on sanity and performance."""
    sanity_status = metric_sanity.get("status", "unknown")
    sanity_severity = metric_sanity.get("severity", "low")

    # Don't auto-promote if sanity checks fail with medium/high severity
    if sanity_status == "failed" and sanity_severity in ["medium", "high"]:
        return {
            "can_auto_promote_models": False,
            "reason": f"Metric sanity check failed ({sanity_severity} severity); metrics may be unreliable.",
            "recommended_actions": [
                "Fix metric calculation issues before enabling auto-promotion.",
                "Review evaluation pipeline for horizon-specific bugs.",
                "Re-run sanity checks after fixes."
            ]
        }

    # Check if we have reasonable performance data
    overall_score = performance.get("overall_score", 0)
    if overall_score < 0.3:  # Very poor performance
        return {
            "can_auto_promote_models": False,
            "reason": f"Overall performance score ({overall_score}) is too low for auto-promotion.",
            "recommended_actions": [
                "Investigate why model performance is poor.",
                "Check data quality and feature engineering.",
                "Consider retraining models with different configurations."
            ]
        }

    # All good
    return {
        "can_auto_promote_models": True,
        "reason": "Metric sanity passed and performance meets thresholds.",
        "recommended_actions": [
            "Proceed with automated model promotion.",
            "Monitor performance in production.",
            "Continue regular evaluation cycles."
        ]
    }


def _render_daily_health_markdown(report: Dict[str, Any]) -> str:
    """Render the daily health report as Markdown."""
    md = []

    # Header
    md.append("# Daily Forecast Health Report")
    md.append("")

    meta = report["run_metadata"]
    md.append(f"**Run ID:** {meta['run_id']}")
    md.append(f"**Evaluated at:** {meta['evaluated_at']}")
    md.append(f"**Symbols:** {', '.join(meta['symbols'])}")
    md.append(f"**Horizons:** {', '.join(meta['horizons'])}")
    md.append("")

    # Metric Sanity
    sanity = report["metric_sanity"]
    status_icon = "❌ FAILED" if sanity["status"] == "failed" else "✅ PASSED"
    severity_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(sanity["severity"], "⚪")

    md.append("---")
    md.append("")
    md.append("## 1. Metric Sanity")
    md.append("")
    md.append(f"- **Status:** {status_icon} ({severity_icon} {sanity['severity'].title()})")
    md.append(f"- **Summary:** {sanity['summary']}")

    if sanity.get("key_findings"):
        md.append("- **Key Findings:**")
        for finding in sanity["key_findings"]:
            md.append(f"  - {finding}")

    md.append("")

    # Model Performance
    perf = report["model_performance"]
    md.append("---")
    md.append("")
    md.append("## 2. Model Performance (High Level)")
    md.append("")
    md.append(f"- **Overall score:** {perf['overall_score']} (0–1 scale, higher is better)")

    if perf.get("metrics"):
        md.append("- **Headline metrics (average across symbols & horizons):**")
        for metric, stats in perf["metrics"].items():
            md.append(f"  - {metric.upper()}: {stats['mean']}")
        md.append("")

    if perf.get("per_model"):
        md.append("- **Model comparison:**")
        for model in perf["per_model"][:3]:  # Top 3
            status_badge = "**champion**" if model["status"] == "champion" else "candidate"
            md.append(f"  - **{model['model_family']}**: {status_badge}")
        md.append("")

    # Cross-asset V2
    v2 = report["cross_asset_v2"]
    md.append("---")
    md.append("")
    md.append("## 3. Cross-Asset Features V2")
    md.append("")
    md.append(f"- **Status:** {'✅ Enabled' if v2['status'] == 'enabled' else '❌ Disabled'}")
    md.append(f"- **Overall effect:** {v2['summary']}")

    if v2.get("overall_lift"):
        md.append("- **Performance lift:**")
        for horizon, lift in v2["overall_lift"].items():
            md.append(f"  - {horizon}: {lift:+.4f}")
        md.append("")

    if v2.get("by_regime"):
        md.append("- **By regime:**")
        for regime, lifts in v2["by_regime"].items():
            md.append(f"  - {regime}: {lifts}")
        md.append("")

    if v2.get("top_features"):
        md.append("- **Most influential new features:**")
        for feature in v2["top_features"]:
            md.append(f"  - `{feature}`")
        md.append("")

    md.append(f"**Decision:** → Keep V2 **{v2['decision'].replace('_', ' ')}**")
    md.append("")

    # Alerts & Decisions
    alerts = report["alerts_and_decisions"]
    md.append("---")
    md.append("")
    md.append("## 4. Alerts & Automated Decisions")
    md.append("")

    promote_icon = "❌ **Disabled**" if not alerts["can_auto_promote_models"] else "✅ **Enabled**"
    md.append(f"- **Auto-promotion of models:** {promote_icon}")
    md.append(f"  - Reason: {alerts['reason']}")

    if alerts.get("recommended_actions"):
        md.append("- **Recommended engineering actions:**")
        for i, action in enumerate(alerts["recommended_actions"], 1):
            md.append(f"  {i}. {action}")

    md.append("")

    # Executive Summary
    md.append("---")
    md.append("")
    md.append("## 5. Short Executive Summary")
    md.append("")

    sanity_ok = sanity["status"] == "passed"
    v2_good = v2["decision"] == "keep_enabled"
    can_promote = alerts["can_auto_promote_models"]

    summary_parts = []
    if sanity_ok:
        summary_parts.append("metric sanity checks passed")
    else:
        summary_parts.append(f"metric sanity issues detected ({sanity['severity']} severity)")

    if perf["overall_score"] > 0.7:
        summary_parts.append("strong model performance")
    elif perf["overall_score"] > 0.5:
        summary_parts.append("adequate model performance")
    else:
        summary_parts.append("concerning model performance")

    if v2_good:
        summary_parts.append("V2 features providing benefits")
    else:
        summary_parts.append("V2 features showing limited impact")

    summary = f"The forecasting system shows {', '.join(summary_parts)}."

    if not can_promote:
        summary += " Automated model promotion has been paused due to quality concerns."
    else:
        summary += " The system is ready for automated operations."

    md.append(f"> {summary}")
    md.append("")

    return "\n".join(md)


def daily_health_report_node(state: GraphState) -> GraphState:
    """
    LangGraph node that combines metric sanity, performance, and V2 analysis
    into a unified daily health report.
    """
    # Resolve paths - handle both dict and dataclass state
    if hasattr(state, 'performance_summary_path'):
        performance_path = state.performance_summary_path
        sanity_path = state.metric_sanity_summary_path
        v2_path = state.cross_asset_v2_analysis_path
        report_json_path = state.daily_report_path_json
        report_md_path = state.daily_report_path_md
        run_id = state.run_id
        symbols = state.symbols
        horizons = state.horizons
        config = state.config
    else:
        # Fallback for dict-like state
        performance_path = state.get("performance_summary_path", "data/metrics/evaluation_results_baseline_latest.csv")
        sanity_path = state.get("metric_sanity_summary_path", "results/quality/metric_sanity_summary.json")
        v2_path = state.get("cross_asset_v2_analysis_path", "results/hpo/v2_ab_analysis.json")
        report_json_path = state.get("daily_report_path_json", "results/reports/daily_forecast_health_latest.json")
        report_md_path = state.get("daily_report_path_md", "results/reports/daily_forecast_health_latest.md")
        run_id = state.get("run_id")
        symbols = state.get("symbols", ["UNKNOWN"])
        horizons = state.get("horizons", ["UNKNOWN"])
        config = state.get("config", {})

    # Load components
    metric_sanity = _load_json_safely(sanity_path)
    model_performance = _aggregate_performance_metrics(performance_path)
    cross_asset_v2 = _analyze_cross_asset_v2(v2_path)

    # Build run metadata
    run_type = state.get("run_type", "DAILY")
    run_metadata = {
        "run_id": run_id or datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ"),
        "run_type": run_type,
        "evaluated_at": datetime.now().isoformat() + "Z",
        "symbols": symbols,
        "horizons": horizons,
        "config": config
    }

    # Make auto-promotion decision
    alerts_decisions = _make_auto_promotion_decision(metric_sanity, model_performance)

    # Build unified report
    report = {
        "run_metadata": run_metadata,
        "metric_sanity": {
            "status": metric_sanity.get("raw_overall_status", {}).get("status", "unknown"),
            "severity": metric_sanity.get("raw_overall_status", {}).get("severity", "low"),
            "issue_count": metric_sanity.get("raw_overall_status", {}).get("issue_count", 0),
            "summary": metric_sanity.get("status_summary", "Sanity check completed"),
            "key_findings": metric_sanity.get("key_findings", []),
            "file_ref": sanity_path
        },
        "model_performance": model_performance,
        "cross_asset_v2": cross_asset_v2,
        "alerts_and_decisions": alerts_decisions
    }

    # Create reports directory
    Path(report_json_path).parent.mkdir(parents=True, exist_ok=True)

    # Write JSON report
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write Markdown report
    markdown_content = _render_daily_health_markdown(report)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # Update state
    if hasattr(state, 'daily_report_path_json'):
        state.daily_report_path_json = report_json_path
        state.daily_report_path_md = report_md_path
        state.can_auto_promote_models = alerts_decisions["can_auto_promote_models"]
    else:
        # Fallback for dict-like state
        state["daily_report_path_json"] = report_json_path
        state["daily_report_path_md"] = report_md_path
        state["can_auto_promote_models"] = alerts_decisions["can_auto_promote_models"]

    return state


# Example usage
if __name__ == "__main__":
    # Example state
    state = GraphState(
        run_id="2025-11-28T21-30-00Z",
        symbols=["AAPL", "MSFT", "NVDA"],
        horizons=["1", "5"],
        config={"feature_store_v2_enabled": True}
    )

    # Run the node
    result_state = daily_health_report_node(state)

    print("Daily health report generated!")
    print(f"JSON: {result_state.get('daily_report_path_json')}")
    print(f"Markdown: {result_state.get('daily_report_path_md')}")
    print(f"Can auto-promote: {result_state.get('can_auto_promote_models')}")