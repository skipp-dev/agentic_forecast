# src/analytics/llm_analytics_orchestrator.py
import json
from pathlib import Path
from typing import Dict, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from analytics.metrics_payload_builder import (
    load_health_report,
    build_metrics_payload_from_health,
)
from src.analytics.llm_report_builder import build_markdown_from_explanation
from agents.llm_analytics_agent import LLMAnalyticsExplainerAgent


DEFAULT_HEALTH_PATH = "results/reports/daily_forecast_health_latest.json"
OUTPUT_JSON = "results/reports/analytics_explainer_latest.json"
OUTPUT_MD = "results/reports/analytics_explainer_latest.md"


def run_llm_analytics_explainer(
    health_path: str = DEFAULT_HEALTH_PATH,
    output_json: str = OUTPUT_JSON,
    output_md: str = OUTPUT_MD,
) -> Dict[str, Any]:
    """
    End-to-end: load health → build metrics_payload → call LLMAnalyticsExplainerAgent →
    save JSON + Markdown → return explanation dict.
    """
    health = load_health_report(health_path)
    metrics_payload = build_metrics_payload_from_health(health)

    agent = LLMAnalyticsExplainerAgent()
    explanation = agent.explain_metrics(metrics_payload)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(
        json.dumps(explanation, indent=2),
        encoding="utf-8",
    )

    md = build_markdown_from_explanation(explanation, metrics_payload)
    Path(output_md).write_text(md, encoding="utf-8")

    return explanation