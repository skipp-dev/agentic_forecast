# src/analytics/llm_report_builder.py
from typing import Dict, Any, List


def build_markdown_from_explanation(expl: Dict[str, Any], metrics_payload: Dict[str, Any]) -> str:
    """Turn the LLM JSON into a readable .md for you / dashboards."""
    run_meta = metrics_payload.get("run_metadata", {})
    date = run_meta.get("date", "unknown")
    run_type = run_meta.get("run_type", "unknown")

    lines: List[str] = []
    lines.append(f"# LLM Analytics Summary – {date} ({run_type})")
    lines.append("")

    global_summary = expl.get("global_summary", "No summary provided.")
    lines.append("## Global Summary")
    lines.append("")
    lines.append(global_summary)
    lines.append("")

    metric_expl = expl.get("metric_explanations", {})
    if metric_expl:
        lines.append("## Metric Explanations")
        lines.append("")
        for name, text in metric_expl.items():
            lines.append(f"**{name.upper()}** – {text}")
        lines.append("")

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