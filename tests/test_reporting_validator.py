import pytest
from src.reporting.validators import validate_report_consistency, ReportConsistencyError

class TestReportValidator:
    
    def test_valid_report_consistency(self):
        """Test that a consistent report passes validation."""
        metrics = {
            "num_anomalies": 5,
            "total_symbols": 100,
            "avg_mape": 0.125,  # 12.5%
            "risk_events": []
        }
        
        report_html = """
        <html>
        <body>
            <p>Total Symbols: 100</p>
            <p>We found 5 anomalies in the dataset.</p>
            <p>The Average MAPE was 12.5%.</p>
            <p>No risk events occurred.</p>
        </body>
        </html>
        """
        
        # Should not raise exception
        validate_report_consistency(report_html, metrics)

    def test_anomaly_hallucination(self):
        """Test that inventing anomalies raises an error."""
        metrics = {
            "num_anomalies": 0,
            "total_symbols": 100
        }
        
        report_html = """
        <p>We detected 5 anomalies in the data.</p>
        """
        
        with pytest.raises(ReportConsistencyError) as excinfo:
            validate_report_consistency(report_html, metrics)
        
        assert "Report claims 5 anomalies, but metrics show 0" in str(excinfo.value)

    def test_symbol_count_mismatch(self):
        """Test that getting the symbol count wrong raises an error."""
        metrics = {
            "total_symbols": 50
        }
        
        report_html = """
        <p>Total Symbols: 100</p>
        """
        
        with pytest.raises(ReportConsistencyError):
            validate_report_consistency(report_html, metrics)

    def test_mape_hallucination(self):
        """Test that inventing a MAPE score raises an error."""
        metrics = {
            "avg_mape": 0.10, # 10%
            "median_mape": 0.08
        }
        
        report_html = """
        <p>The Average MAPE is 5.0%.</p>
        """
        
        with pytest.raises(ReportConsistencyError) as excinfo:
            validate_report_consistency(report_html, metrics)
            
        assert "Report claims Average MAPE 5.0%, but truth is 10.00%" in str(excinfo.value)

    def test_risk_event_omission(self):
        """Test that failing to mention a blocked portfolio raises an error."""
        metrics = {
            "risk_events": [
                {"type": "portfolio_blocked", "reason": "volatility"}
            ]
        }
        
        report_html = """
        <p>Everything is fine. No issues.</p>
        """
        
        with pytest.raises(ReportConsistencyError) as excinfo:
            validate_report_consistency(report_html, metrics)
            
        assert "Risk events occurred (portfolio blocked), but report does not mention" in str(excinfo.value)

    def test_risk_event_mention_passes(self):
        """Test that mentioning the blocked portfolio passes."""
        metrics = {
            "risk_events": [
                {"type": "portfolio_blocked", "reason": "volatility"}
            ]
        }
        
        report_html = """
        <p>The portfolio was blocked due to high volatility.</p>
        """
        
        validate_report_consistency(report_html, metrics)

    def test_healthy_status_lie(self):
        """Test that claiming healthy when failing raises error."""
        metrics = {
            "guardrails": {"status": "failing"}
        }
        
        report_html = """
        <p>System Status: Healthy</p>
        """
        
        with pytest.raises(ReportConsistencyError) as excinfo:
            validate_report_consistency(report_html, metrics)
            
        assert "Report claims Healthy status but guardrails are failing" in str(excinfo.value)
