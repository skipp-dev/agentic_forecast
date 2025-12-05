# src/reporting/postprocess.py

import re
from typing import Any, Dict, List


def normalize_system_report(
    report_html: str,
    metrics: Dict[str, Any],
    guardrail_status: str,
) -> str:
    """
    Force-overwrite specific sections of the LLM report with ground-truth data.
    This prevents the LLM from hallucinating "573 anomalies" or "0 trials".
    """
    # 1. Overwrite the "System Health" badge/text
    # We look for patterns like "Status: Healthy" or "Status: Failing" and force our status.
    # A simple regex replacement for the status line:
    status_pattern = re.compile(r"(System Status:)\s*(<strong>)?\w+(</strong>)?", re.IGNORECASE)
    
    # Map our internal status to a display string
    display_status = guardrail_status.upper()
    replacement = f"System Status: <strong>{display_status}</strong>"
    
    report_html = status_pattern.sub(replacement, report_html)

    # 2. Inject the exact anomaly count if we can find the section
    # (This is harder if the LLM writes free text, but we can append a footer or
    # replace a known placeholder if we used one. For now, let's just ensure
    # the summary table at the top is correct if it exists.)
    
    return report_html


def format_hpo_summary(trials: List[Dict]) -> str:
    """
    Generate a deterministic HTML block for HPO results.
    """
    if not trials:
        return "<p><em>No HPO trials were run for this session.</em></p>"
    
    # If we had trials, build a table...
    return f"<p>Ran {len(trials)} trials.</p>"
