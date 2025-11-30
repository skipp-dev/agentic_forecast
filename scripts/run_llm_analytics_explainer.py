# scripts/run_llm_analytics_explainer.py
import sys
from pathlib import Path

# Add src and root to path
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import json
from pathlib import Path

from analytics.metrics_payload_builder import (
    load_health_report,
    build_metrics_payload_from_health,
)
from agents.llm_analytics_agent import LLMAnalyticsExplainerAgent


HEALTH_PATH = "results/reports/daily_forecast_health_latest.json"
OUTPUT_JSON = "results/reports/analytics_explainer_latest.json"
OUTPUT_MD = "results/reports/analytics_explainer_latest.md"


def main() -> None:
    # 1) Load existing health report
    health = load_health_report(HEALTH_PATH)

    # 2) Build metrics_payload
    metrics_payload = build_metrics_payload_from_health(health)

    # 3) Call the LLMAnalyticsExplainerAgent
    agent = LLMAnalyticsExplainerAgent()
    explanation = agent.explain_metrics(metrics_payload)

    # 4) Save raw JSON explanation
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_JSON).write_text(
        json.dumps(explanation, indent=2),
        encoding="utf-8",
    )

    # 5) Build a human-friendly Markdown report
    md = build_markdown_from_explanation(explanation, metrics_payload)
    Path(OUTPUT_MD).write_text(md, encoding="utf-8")

    print(f"Wrote LLM analytics JSON to: {OUTPUT_JSON}")
    print(f"Wrote LLM analytics markdown to: {OUTPUT_MD}")


def build_markdown_from_explanation(expl: dict, metrics_payload: dict) -> str:
    """Turn the LLM JSON into a readable .md for you / Grafana."""
    run_meta = metrics_payload.get("run_metadata", {})
    date = run_meta.get("date", "unknown")
    run_type = run_meta.get("run_type", "unknown")

    lines = []
    lines.append(f"# LLM Analytics Summary – {date} ({run_type})")
    lines.append("")

    # Global summary
    global_summary = expl.get("global_summary", "No summary provided.")
    lines.append("## Global Summary")
    lines.append("")
    lines.append(global_summary)
    lines.append("")

    # Metric explanations
    metric_expl = expl.get("metric_explanations", {})
    if metric_expl:
        lines.append("## Metric Explanations")
        lines.append("")
        for name, text in metric_expl.items():
            lines.append(f"**{name.upper()}** – {text}")
        lines.append("")

    # Regime insights
    regime_insights = expl.get("regime_insights", [])
    if regime_insights:
        lines.append("## Regime Insights")
        lines.append("")
        for r in regime_insights:
            regime = r.get("regime", "unknown")
            perf = r.get("performance_comment", "")
            risk = r.get("risk_comment", "")
            lines.append(f"### Regime: {regime}")
            if perf:
                lines.append(f"- Performance: {perf}")
            if risk:
                lines.append(f"- Risk: {risk}")
            lines.append("")
    
    # Symbol outliers
    outliers = expl.get("symbol_outliers", [])
    if outliers:
        lines.append("## Symbol Outliers")
        lines.append("")
        for o in outliers:
            sym = o.get("symbol", "UNKNOWN")
            horizon = o.get("horizon", "n/a")
            issue = o.get("issue", "")
            comment = o.get("comment", "")
            lines.append(f"- **{sym}** (horizon {horizon}): {issue}")
            if comment:
                lines.append(f"  - {comment}")
        lines.append("")

    # Feature insights
    feat = expl.get("feature_insights", {})
    overall = feat.get("overall_top_features", [])
    shock = feat.get("shock_regime_top_features", [])

    if overall or shock:
        lines.append("## Feature Insights")
        lines.append("")
        if overall:
            lines.append("### Overall Top Features")
            for f in overall:
                name = f.get("name", "unknown")
                comment = f.get("importance_comment", "")
                lines.append(f"- **{name}** – {comment}")
            lines.append("")
        if shock:
            lines.append("### Shock Regime Top Features")
            for f in shock:
                name = f.get("name", "unknown")
                comment = f.get("importance_comment", "")
                lines.append(f"- **{name}** – {comment}")
            lines.append("")
    
    # Recommendations
    recs = expl.get("recommendations", [])
    if recs:
        lines.append("## Recommendations")
        lines.append("")
        for r in recs:
            cat = r.get("category", "general")
            action = r.get("action", "")
            reason = r.get("reason", "")
            lines.append(f"- **[{cat}]** {action}")
            if reason:
                lines.append(f"  - Reason: {reason}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()