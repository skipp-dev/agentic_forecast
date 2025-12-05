# src/reporting/validators.py

import re
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ReportConsistencyError(Exception):
    """Raised when the generated report contradicts the ground truth metrics."""
    pass

def validate_report_consistency(report_html: str, metrics: Dict[str, Any]):
    """
    Scan the report for numbers and ensure they match the metrics.
    This is a heuristic check to catch hallucinations.
    """
    report_text = report_html.lower() # Simple normalization for searching

    # --- 1. Anomaly Count Check ---
    # Extract anomaly count from text like "found 573 anomalies" or "573 anomalies detected"
    # We look for patterns like: "X anomalies", "anomalies: X"
    anomaly_matches = re.findall(r"(\d+)\s+anomalies", report_text)
    anomaly_matches += re.findall(r"anomalies:\s*(\d+)", report_text)
    
    # Get truth from metrics (handle different structures)
    true_anomalies = metrics.get("num_anomalies", 0)
    if isinstance(true_anomalies, dict):
        true_anomalies = true_anomalies.get("count", 0)
    
    for match in anomaly_matches:
        claimed_count = int(match)
        if claimed_count != true_anomalies:
            msg = f"Report claims {claimed_count} anomalies, but metrics show {true_anomalies}."
            logger.error(msg)
            raise ReportConsistencyError(msg)

    # --- 2. Total Symbols Check ---
    # "X symbols", "total symbols: X"
    # We want to be strict about "Total Symbols: X" but loose about "X symbols" (could be "5 symbols failed")
    
    # Specific check for "Total Symbols: X"
    total_symbol_matches = re.findall(r"total symbols:?\s*(\d+)", report_text)
    true_symbols = metrics.get("total_symbols", 0)
    
    for match in total_symbol_matches:
        claimed_count = int(match)
        if claimed_count != true_symbols:
            msg = f"Report claims Total Symbols: {claimed_count}, but metrics show {true_symbols}."
            logger.error(msg)
            raise ReportConsistencyError(msg)

    # --- 3. MAPE Check (Approximate) ---
    # If report says "Average MAPE: 5.2%", truth should be close.
    
    true_avg_mape = metrics.get("avg_mape", 0.0)
    true_median_mape = metrics.get("median_mape", 0.0)
    
    # Convert truth to percentage for comparison
    true_avg_pct = true_avg_mape * 100
    true_median_pct = true_median_mape * 100
    
    # Look for "Average MAPE ... X%"
    # This regex allows for "is", ":", "of", etc. in between
    avg_mape_matches = re.findall(r"average mape\D{0,20}(\d+\.?\d*)%", report_text)
    
    for match in avg_mape_matches:
        try:
            claimed_val = float(match)
            # Check if it matches avg within tolerance
            if not (abs(claimed_val - true_avg_pct) < 0.5):
                 msg = f"Report claims Average MAPE {claimed_val}%, but truth is {true_avg_pct:.2f}%."
                 logger.error(msg)
                 raise ReportConsistencyError(msg)
        except ValueError:
            continue
            
    # Look for "Median MAPE ... X%"
    median_mape_matches = re.findall(r"median mape\D{0,20}(\d+\.?\d*)%", report_text)
    
    for match in median_mape_matches:
        try:
            claimed_val = float(match)
            # Check if it matches median within tolerance
            if not (abs(claimed_val - true_median_pct) < 0.5):
                 msg = f"Report claims Median MAPE {claimed_val}%, but truth is {true_median_pct:.2f}%."
                 logger.error(msg)
                 raise ReportConsistencyError(msg)
        except ValueError:
            continue

    # --- 4. Risk Event Check ---
    # If we have a portfolio blocked event, the report MUST mention "blocked" or "rejected"
    risk_events = metrics.get("risk_events", [])
    blocked_events = [e for e in risk_events if e.get("type") in ["portfolio_blocked", "portfolio_rejected"]]
    
    if blocked_events:
        if not any(w in report_text for w in ["blocked", "rejected", "prevented", "stopped"]):
            msg = "Risk events occurred (portfolio blocked), but report does not mention 'blocked' or 'rejected'."
            logger.error(msg)
            raise ReportConsistencyError(msg)

    # --- 5. Guardrail Status Check ---
    guardrails = metrics.get("guardrails", {})
    status = guardrails.get("status", "unknown")
    
    if status == "failing":
        if "system status: healthy" in report_text:
             msg = "Report claims Healthy status but guardrails are failing."
             logger.error(msg)
             raise ReportConsistencyError(msg)

    logger.info("Report consistency check passed.")
