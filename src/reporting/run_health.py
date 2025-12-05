from typing import List, Dict, Any, Union

def compute_run_health(risk_events: List[Any], guardrail_log: List[str]) -> Dict[str, Any]:
    """
    Compute the health of the run based on risk events and guardrail logs.
    
    Returns:
        Dict with 'status' (GREEN, YELLOW, RED) and 'reasons' (List[str])
    """
    reasons = []
    status = "GREEN"
    
    # Check for critical risk events
    critical_events = [e for e in risk_events if getattr(e, 'severity', 'high') == 'critical']
    if critical_events:
        status = "RED"
        reasons.append(f"{len(critical_events)} critical risk events detected")
        
    # Check for portfolio rejection (which is critical)
    rejections = [e for e in risk_events if getattr(e, 'event_type', '') == 'portfolio_rejection']
    if rejections:
        status = "RED"
        reasons.append("Portfolio construction rejected by risk agent")
        
    # Check guardrails
    # Guardrail log entries are strings like "Guardrail blocked..." or "Guardrail warning..."
    blocked_actions = [g for g in guardrail_log if "blocked" in g.lower()]
    warnings = [g for g in guardrail_log if "warning" in g.lower()]
    
    if blocked_actions:
        # Blocking actions is usually a sign of risk, but system is working.
        # However, if too many are blocked, it might be an issue.
        # Let's say if > 50% actions blocked? We don't know total here easily.
        # For now, treat blocks as YELLOW unless critical.
        if status != "RED":
            status = "YELLOW"
            reasons.append(f"{len(blocked_actions)} actions blocked by guardrails")
            
    if warnings:
        if status == "GREEN":
            status = "YELLOW"
            reasons.append(f"{len(warnings)} guardrail warnings")
            
    if not reasons:
        reasons.append("System healthy")
        
    return {
        "status": status,
        "reasons": reasons
    }
