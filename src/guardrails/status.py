# src/guardrails/status.py

from __future__ import annotations


def compute_guardrail_status(
    *,
    total_checks: int,
    passed: int,
    warnings: int,
    critical: int,
    engine_errors: int = 0,
    skipped: int = 0,
) -> str:
    """
    Derive a simple normalized guardrail status from raw counts.

    Semantics:
    - "offline": engine errors OR we expected checks but none show as passed/warn/critical.
    - "failing": at least one critical guardrail fired.
    - "degraded": only warnings.
    - "inactive": total_checks == 0 (feature not enabled).
    - "healthy": all checks passed, no warnings/criticals.
    """
    if engine_errors > 0:
        return "offline"

    if total_checks > 0 and passed == 0 and warnings == 0 and critical == 0:
        # Something is wrong: we think there are checks, but none have a status.
        return "offline"

    if critical > 0:
        return "failing"

    if warnings > 0:
        return "degraded"

    if total_checks == 0:
        return "inactive"

    return "healthy"
