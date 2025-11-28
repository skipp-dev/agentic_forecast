# Metric Sanity Explainer Node for LangGraph

from typing import TypedDict, Optional, Dict, Any
import json
from pathlib import Path
from datetime import datetime

from llm_prompts import METRIC_SANITY_EXEC_STRUCTURED_PROMPT


class GraphState(TypedDict, total=False):
    run_id: Optional[str]
    metric_sanity_report_path: str
    metric_sanity_summary_path: str
    metric_sanity_summary: Dict[str, Any]
    metric_sanity_status: str
    metric_sanity_issue_count: int


def call_reporting_llm(prompt: str, model: str = "reporting-llm") -> str:
    """
    Generic wrapper to call your ReportingLLM.
    Returns the raw text response.

    TODO: Replace with your actual LLM client (OpenAI, LM Studio, etc.)
    """
    # Placeholder implementation - replace with your actual LLM call
    # For now, return a mock response
    return json.dumps({
        "status_summary": "Metric sanity check passed with low severity.",
        "key_findings": [
            "All metrics show reasonable ranges and variability.",
            "SMAPE and SWASE have appropriate unique value counts.",
            "No critical calculation errors detected."
        ],
        "recommended_actions": [
            "Continue monitoring metric quality in production.",
            "Consider adding more sophisticated outlier detection."
        ],
        "risk_assessment": "Metrics appear trustworthy for automated model selection and guardrails.",
        "raw_overall_status": {
            "status": "passed",
            "severity": "low",
            "issue_count": 0,
            "summary": "All checks passed."
        }
    })


def metric_sanity_explainer_node(state: GraphState) -> GraphState:
    """
    LangGraph node that:
    - reads metric_sanity_latest.json
    - calls the ReportingLLM with a structured prompt
    - writes metric_sanity_summary.json
    - updates the graph state
    """
    # 1. Resolve paths
    report_path = Path(
        state.get("metric_sanity_report_path", "results/quality/metric_sanity_latest.json")
    )
    summary_path = Path(
        state.get("metric_sanity_summary_path", "results/quality/metric_sanity_summary.json")
    )

    if not report_path.exists():
        # Nothing to explain; record status and bail
        state["metric_sanity_status"] = "missing_report"
        state["metric_sanity_issue_count"] = 0
        state["metric_sanity_summary"] = {
            "status_summary": "No metric sanity report found; nothing to analyze.",
            "key_findings": [],
            "recommended_actions": ["Run the QualityAgent metric sanity check first."],
            "risk_assessment": "Forecast metrics cannot be assessed until the sanity report exists.",
            "raw_overall_status": None,
        }
        return state

    # 2. Load JSON
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    metric_sanity_json_str = json.dumps(report, indent=2)

    # 3. Build prompt
    prompt = METRIC_SANITY_EXEC_STRUCTURED_PROMPT.replace(
        "{{ metric_sanity_json }}", metric_sanity_json_str
    )

    # 4. Call LLM
    raw_response = call_reporting_llm(prompt, model="reporting-llm")

    # 5. Parse LLM JSON
    try:
        summary = json.loads(raw_response)
    except json.JSONDecodeError:
        # Fallback if the model misbehaves
        summary = {
            "status_summary": "LLM failed to return valid JSON summary.",
            "key_findings": ["Failed to parse ReportingLLM output."],
            "recommended_actions": [
                "Check the ReportingLLM prompt or logs.",
                "Temporarily rely on the raw metric_sanity_latest.json."
            ],
            "risk_assessment": "Automatically generated explanations are unavailable for this run.",
            "raw_overall_status": report.get("overall_status", {}),
        }

    # 6. Write summary to disk
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 7. Update state
    raw_overall = summary.get("raw_overall_status") or report.get("overall_status") or {}
    status = raw_overall.get("status", "unknown")
    issue_count = raw_overall.get("issue_count", 0)

    state["metric_sanity_summary"] = summary
    state["metric_sanity_status"] = status
    state["metric_sanity_issue_count"] = int(issue_count)

    return state


# Example usage in a LangGraph workflow
if __name__ == "__main__":
    # Example state
    state = GraphState(
        run_id="2025-02-03T21-15-00Z",
        metric_sanity_report_path="results/quality/metric_sanity_latest.json",
        metric_sanity_summary_path="results/quality/metric_sanity_summary.json"
    )

    # Run the node
    result_state = metric_sanity_explainer_node(state)

    print("Metric sanity status:", result_state.get("metric_sanity_status"))
    print("Issue count:", result_state.get("metric_sanity_issue_count"))
    print("Summary keys:", list(result_state.get("metric_sanity_summary", {}).keys()))