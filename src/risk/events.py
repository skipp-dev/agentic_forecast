# src/risk/events.py

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Literal, Optional, List


RiskSeverity = Literal["info", "warning", "critical"]
RiskSource = Literal["risk_management", "portfolio", "guardrails", "unknown"]


@dataclass
class RiskEvent:
    """
    Canonical representation of a risk event in the agentic_forecast system.

    This is intentionally minimal and JSON-friendly.
    """
    type: str
    severity: RiskSeverity
    source: RiskSource
    message: str
    symbol: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


def add_risk_event(state: Dict[str, Any], event: RiskEvent) -> None:
    """
    Append a risk event to the shared run_state in a consistent way.
    """
    events = list(state.get("risk_events", []))
    # Convert dataclass to dict for state storage
    events.append(asdict(event))
    state["risk_events"] = events


def portfolio_rejection_event(
    *,
    reason: str,
    var_value: Optional[float] = None,
    var_limit: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
) -> RiskEvent:
    """
    Convenience helper for portfolio rejection events.
    """
    event_details: Dict[str, Any] = {}
    if var_value is not None:
        event_details["var_value"] = var_value
    if var_limit is not None:
        event_details["var_limit"] = var_limit
    
    if details:
        event_details.update(details)

    return RiskEvent(
        type="portfolio_rejected",
        severity="critical",
        source="portfolio",
        message=reason,
        symbol=None,
        details=event_details,
    )
