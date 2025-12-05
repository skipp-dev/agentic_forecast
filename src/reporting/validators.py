# src/reporting/validators.py

import re
from typing import Dict, Any


class ReportConsistencyError(Exception):
    """Raised when the generated report contradicts the ground truth metrics."""
    pass


def validate_report_consistency(report_html: str, metrics: Dict[str, Any]):
    """
    Scan the report for numbers and ensure they match the metrics.
    """
    # Example: Check anomaly count
    # If the report says "573 anomalies" but metrics say 0, raise Error.
    
    # 1. Extract anomaly count from text like "found 573 anomalies" or "573 anomalies detected"
    # This is a heuristic check.
    anomaly_matches = re.findall(r"(\d+)\s+anomalies", report_html, re.IGNORECASE)
    
    true_anomalies = metrics.get("anomalies", {}).get("count", 0)
    
    for match in anomaly_matches:
        claimed_count = int(match)
        if claimed_count != true_anomalies:
            # We allow some fuzziness if the text is "0 anomalies" vs "no anomalies",
            # but if there's a number, it must match.
            raise ReportConsistencyError(
                f"Report claims {claimed_count} anomalies, but metrics show {true_anomalies}."
            )

    # 2. Check for "Healthy" status if we are actually "Failing"
    # (This is partially covered by post-processing, but good to verify).
    if metrics.get("guardrails", {}).get("status") == "failing":
        if "System Status: Healthy" in report_html:
             raise ReportConsistencyError("Report claims Healthy status but guardrails are failing.")
