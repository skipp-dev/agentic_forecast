import json
import re
import pytest
from unittest.mock import MagicMock

from src.agents.reporting_agent import LLMReportingAgent as ReportingAgent, SystemReport, ReportingInput

class RecordingFakeLLM:
    """
    Minimal fake LLM that:
    - records the last prompt & system prompt
    - returns a valid, but mostly empty, SystemReport JSON
    """
    def __init__(self):
        self.last_prompt = None
        self.last_system = None

    def complete(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        self.last_prompt = prompt
        self.last_system = system

        # Minimal valid JSON according to your new reporting_agent schema
        return json.dumps(
            {
                "executive_summary": "stub",
                "sections": [],
                "key_risks": [],
                "key_opportunities": [],
                "actions_for_quants": [],
                "actions_for_ops": [],
                "performance_overview": {
                    "headline": "",
                    "metrics": {
                        "total_symbols": 0,
                        "models_trained": 0,
                        "models_promoted": 0,
                        "avg_mape": 0.0,
                        "median_mape": 0.0,
                        "num_anomalies": 0,
                    },
                    "model_comparison_comment": "",
                },
                "risk_assessment": {
                    "guardrails": {
                        "summary": "",
                        "raw_counts": {
                            "total_checks": 0,
                            "passed": 0,
                            "warnings": 0,
                            "critical": 0,
                        },
                    },
                    "risk_events": [],
                    "interpretation": "",
                    "open_issues": [],
                },
                "optimization_recommendations": {
                    "hpo": [],
                    "models": [],
                    "features": [],
                },
                "research_insights": {
                    "summary": "",
                    "hypotheses": [],
                    "data_suggestions": [],
                },
                "operational_notes": {
                    "system_health": "",
                    "data_quality": "",
                    "maintenance_needs": [],
                },
                "priority_actions": [],
            }
        )


class EchoRiskEventsFakeLLM:
    """
    Fake LLM that:
    - parses risk_events from the user prompt JSON
    - returns them back inside risk_assessment.risk_events
    - includes a short executive_summary mentioning the portfolio rejection
    This simulates a 'good' LLM behaviour and tests the wiring end-to-end.
    """
    def __init__(self):
        self.last_prompt = None
        self.last_system = None

    def complete(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        self.last_prompt = prompt
        self.last_system = system

        # Simplified logic: if we see the portfolio rejection in the prompt text,
        # we assume the LLM sees it and we return it in the response.
        # This avoids fragile regex parsing of nested JSON in the prompt.
        
        risk_events = []
        if "portfolio_rejected" in prompt and "volatility_limit" in prompt:
            risk_events = [
                {
                    "type": "portfolio_rejected",
                    "reason": "volatility_limit",
                    "details": {
                        "portfolio_volatility": 0.3084,
                        "volatility_limit": 0.30,
                    },
                }
            ]

        # Build a response that echoes risk events into risk_assessment
        if any(ev.get("type") == "portfolio_rejected" for ev in risk_events):
            exec_summary = (
                "The run included a portfolio_rejected risk event; "
                "the portfolio was blocked by risk rails before execution."
            )
        else:
            exec_summary = "No portfolio-related risk events detected."

        return json.dumps(
            {
                "executive_summary": exec_summary,
                "sections": [],
                "key_risks": [],
                "key_opportunities": [],
                "actions_for_quants": [],
                "actions_for_ops": [],
                "performance_overview": {
                    "headline": "",
                    "metrics": {
                        "total_symbols": 0,
                        "models_trained": 0,
                        "models_promoted": 0,
                        "avg_mape": 0.0,
                        "median_mape": 0.0,
                        "num_anomalies": 0,
                    },
                    "model_comparison_comment": "",
                },
                "risk_assessment": {
                    "guardrails": {
                        "summary": "",
                        "raw_counts": {
                            "total_checks": 0,
                            "passed": 0,
                            "warnings": 0,
                            "critical": 0,
                        },
                    },
                    "risk_events": risk_events,
                    "interpretation": "Synthetic test response.",
                    "open_issues": [],
                },
                "optimization_recommendations": {
                    "hpo": [],
                    "models": [],
                    "features": [],
                },
                "research_insights": {
                    "summary": "",
                    "hypotheses": [],
                    "data_suggestions": [],
                },
                "operational_notes": {
                    "system_health": "",
                    "data_quality": "",
                    "maintenance_needs": [],
                },
                "priority_actions": [],
            }
        )


@pytest.fixture
def synthetic_metrics_overview_with_portfolio_rejection():
    """
    Synthetic metrics_overview matching your real structure, but minimized.
    """
    return {
        "total_symbols": 10,
        "models_trained": 10,
        "models_promoted": 0,
        "avg_mape": 0.04,
        "median_mape": 0.03,
        "num_anomalies": 0,
        "hpo": {
            "run_type": "WEEKEND_HPO",
            "total_trials": 3,
            "families": {},
        },
        "guardrails": {
            "total_checks": 5,
            "passed": 5,
            "warnings": 0,
            "critical": [],
        },
        "risk_events": [
            {
                "type": "portfolio_rejected",
                "reason": "volatility_limit",
                "details": {
                    "portfolio_volatility": 0.3084,
                    "volatility_limit": 0.30,
                },
            }
        ],
        "model_comparison": {
            "leaderboard": {
                "AAPL": "BaselineLinear"
            },
            "promotions": [],
            "baseline_wins": 1,
            "challenger_wins": 0,
        },
    }


@pytest.fixture
def minimal_llm_inputs(synthetic_metrics_overview_with_portfolio_rejection):
    """
    Build a minimal set of inputs that your ReportingAgent expects.
    """
    return ReportingInput(
        run_metadata={
            "run_type": "WEEKEND_HPO",
            "run_id": "test_run_123",
            "timestamp": "2025-12-04T18:41:39Z",
        },
        analytics_summary={"dummy": "analytics"},
        hpo_plan={"dummy": "hpo"},
        research_insights={"dummy": "research"},
        guardrail_status={"dummy": "guardrails", "risk_events": synthetic_metrics_overview_with_portfolio_rejection["risk_events"]}, # Pass risk events here too as ReportingAgent might look here
    )


def test_risk_events_are_present_in_llm_prompt(
    minimal_llm_inputs,
    synthetic_metrics_overview_with_portfolio_rejection,
):
    """
    Ensure metrics_overview.risk_events actually appear in the user prompt sent to the LLM.
    This catches wiring / formatting bugs in the ReportingAgent.
    """
    fake_llm = RecordingFakeLLM()
    agent = ReportingAgent(llm_client=fake_llm)

    # Mock _build_metrics_overview to return our synthetic metrics
    # This is necessary because ReportingAgent builds metrics_overview internally
    agent._build_metrics_overview = MagicMock(return_value=synthetic_metrics_overview_with_portfolio_rejection)
    
    # Mock _enhance_input_with_evaluation_metrics to just return input
    agent._enhance_input_with_evaluation_metrics = MagicMock(return_value=minimal_llm_inputs)

    # --- call the method that triggers the LLM ---
    report = agent.generate_report(minimal_llm_inputs)

    # Ensure we actually called the LLM
    assert fake_llm.last_prompt is not None

    # The serialized risk_events should appear in the prompt
    # We just check for signature strings that indicate it's there.
    assert "portfolio_rejected" in fake_llm.last_prompt
    assert "volatility_limit" in fake_llm.last_prompt
    assert '"risk_events"' in fake_llm.last_prompt


def test_risk_events_echoed_into_system_report(
    minimal_llm_inputs,
    synthetic_metrics_overview_with_portfolio_rejection
):
    """
    End-to-end test (with a fake LLM) that:
    - risk_events in metrics_overview are visible in the final SystemReport.risk_assessment
    This simulates a 'good' LLM that follows the contract.
    """
    fake_llm = EchoRiskEventsFakeLLM()
    agent = ReportingAgent(llm_client=fake_llm)
    
    # Mock _build_metrics_overview to return our synthetic metrics
    agent._build_metrics_overview = MagicMock(return_value=synthetic_metrics_overview_with_portfolio_rejection)
    
    # Mock _enhance_input_with_evaluation_metrics to just return input
    agent._enhance_input_with_evaluation_metrics = MagicMock(return_value=minimal_llm_inputs)

    full_report = agent.generate_report(minimal_llm_inputs)
    report = full_report.system_report

    # Basic sanity: we got a SystemReport object
    assert isinstance(report, SystemReport)

    # The risk_assessment should contain the portfolio_rejected event
    risk_assessment = report.risk_assessment
    assert "risk_events" in risk_assessment

    risk_events = risk_assessment["risk_events"]
    assert isinstance(risk_events, list)
    assert any(ev.get("type") == "portfolio_rejected" for ev in risk_events)

    # Check that volatility details survived
    event = next(ev for ev in risk_events if ev.get("type") == "portfolio_rejected")
    details = event.get("details", {})
    assert details.get("portfolio_volatility") == pytest.approx(0.3084)
    assert details.get("volatility_limit") == pytest.approx(0.30)


def test_executive_summary_mentions_portfolio_block(
    minimal_llm_inputs,
    synthetic_metrics_overview_with_portfolio_rejection
):
    """
    Ensure the executive summary explicitly mentions that a portfolio was blocked by risk rails.
    This is important for CIO / management transparency.
    """
    fake_llm = EchoRiskEventsFakeLLM()
    agent = ReportingAgent(llm_client=fake_llm)
    
    # Mock _build_metrics_overview to return our synthetic metrics
    agent._build_metrics_overview = MagicMock(return_value=synthetic_metrics_overview_with_portfolio_rejection)
    
    # Mock _enhance_input_with_evaluation_metrics to just return input
    agent._enhance_input_with_evaluation_metrics = MagicMock(return_value=minimal_llm_inputs)

    full_report = agent.generate_report(minimal_llm_inputs)
    report = full_report.system_report

    summary = report.executive_summary.lower()
    assert "portfolio" in summary
    assert "blocked" in summary or "rejected" in summary
    assert "volatility" in summary or "risk rails" in summary
