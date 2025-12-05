import pytest
from src.guardrails.status import compute_guardrail_status
from src.reporting.postprocess import normalize_system_report
from src.reporting.validators import validate_report_consistency, ReportConsistencyError

def test_compute_guardrail_status():
    assert compute_guardrail_status(total_checks=10, passed=10, warnings=0, critical=0) == "healthy"
    assert compute_guardrail_status(total_checks=10, passed=9, warnings=1, critical=0) == "degraded"
    assert compute_guardrail_status(total_checks=10, passed=9, warnings=0, critical=1) == "failing"
    assert compute_guardrail_status(total_checks=0, passed=0, warnings=0, critical=0) == "inactive"
    assert compute_guardrail_status(total_checks=10, passed=0, warnings=0, critical=0) == "offline"
    assert compute_guardrail_status(total_checks=10, passed=10, warnings=0, critical=0, engine_errors=1) == "offline"

def test_normalize_system_report():
    html = "<html><body>System Status: Healthy</body></html>"
    metrics = {}
    normalized = normalize_system_report(html, metrics, "failing")
    assert "System Status: <strong>FAILING</strong>" in normalized

def test_validate_report_consistency():
    html = "Found 573 anomalies in the data."
    metrics = {"anomalies": {"count": 0}}
    
    with pytest.raises(ReportConsistencyError):
        validate_report_consistency(html, metrics)

    html_ok = "Found 0 anomalies in the data."
    validate_report_consistency(html_ok, metrics)
