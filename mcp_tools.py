# MCP Tools for Interactive Analyst

import json
from pathlib import Path
from typing import Dict, Any

from llm_prompts import (
    METRIC_SANITY_EXEC_STRUCTURED_PROMPT,
    METRIC_SANITY_EXEC_SUMMARY_PROMPT
)
from metric_sanity_explainer import call_reporting_llm


def handle_metric_sanity_explainer(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool handler for metric sanity explainer.

    Args:
        params: Tool parameters with 'mode' field ("executive" or "structured")

    Returns:
        Tool response with content
    """
    mode = params.get("mode", "structured")
    report_path = Path("results/quality/metric_sanity_latest.json")

    if not report_path.exists():
        return {
            "content": {
                "status": "missing_report",
                "message": "No metric_sanity_latest.json found. Run quality checks first.",
            }
        }

    # Load the report
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    metric_sanity_json_str = json.dumps(report, indent=2)

    if mode == "structured":
        # Use structured JSON prompt
        prompt = METRIC_SANITY_EXEC_STRUCTURED_PROMPT.replace(
            "{{ metric_sanity_json }}", metric_sanity_json_str
        )
        raw_response = call_reporting_llm(prompt, model="reporting-llm")

        try:
            summary = json.loads(raw_response)
            return {"content": summary}
        except json.JSONDecodeError:
            return {
                "content": {
                    "status": "error",
                    "message": "LLM failed to return valid JSON response.",
                    "raw_response": raw_response[:500]  # Truncate for safety
                }
            }

    elif mode == "executive":
        # Use executive summary prompt
        prompt = METRIC_SANITY_EXEC_SUMMARY_PROMPT.replace(
            "{{ metric_sanity_json }}", metric_sanity_json_str
        )
        text_response = call_reporting_llm(prompt, model="reporting-llm")

        return {
            "content": {
                "markdown": text_response,
                "mode": "executive"
            }
        }

    else:
        return {
            "content": {
                "status": "error",
                "message": f"Unknown mode: {mode}. Use 'structured' or 'executive'."
            }
        }


# MCP Tool Specification
METRIC_SANITY_EXPLAINER_TOOL_SPEC = {
    "name": "metric_sanity_explainer",
    "description": "Reads the latest metric sanity JSON, calls the ReportingLLM, and returns a structured summary about whether evaluation metrics are trustworthy.",
    "input_schema": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["executive", "structured"],
                "description": "Format of the response. 'structured' returns JSON fields; 'executive' returns Markdown.",
                "default": "structured"
            }
        },
        "required": ["mode"]
    }
}


# Example usage
if __name__ == "__main__":
    # Test the tool
    result = handle_metric_sanity_explainer({"mode": "structured"})
    print("Tool result:")
    print(json.dumps(result, indent=2))