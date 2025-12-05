import sys
import os
import json
import logging
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.getcwd())

from src.agents.reporting_agent import LLMReportingAgent, ReportingInput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_flow")

def test_reporting_flow():
    print("--- Starting End-to-End Reporting Verification ---")

    # 1. Setup Mock Data
    mock_metrics = {
        "total_symbols": 10,
        "avg_mape": 0.05,
        "num_anomalies": 0,
        "risk_events": []
    }
    
    input_data = ReportingInput(
        analytics_summary={"anomalies_detected": 0},
        hpo_plan={"total_trials": 0},
        research_insights={},
        guardrail_status={"total_checks": 10, "passed_checks": 10, "warnings": 0, "critical_issues": 0},
        run_metadata={"run_type": "TEST"}
    )

    # 2. Mock the LLM Client
    mock_llm = MagicMock()
    
    # Case A: Consistent Report
    # The LLM returns numbers that match the input (0 anomalies, etc.)
    good_json = json.dumps({
        "executive_summary": "The system performed well with 0 anomalies detected.",
        "sections": [],
        "key_risks": [],
        "key_opportunities": []
    })
    mock_llm.complete.return_value = good_json

    # Initialize Agent with mock LLM
    agent = LLMReportingAgent(llm_client=mock_llm)
    
    # Run Agent (Good Case)
    print("\n[Test 1] Running with CONSISTENT LLM output...")
    report = agent.generate_report(input_data)
    
    if "[AUTOMATED WARNING" in report.system_report.executive_summary:
        print("FAILED: Found warning in good report!")
        sys.exit(1)
    else:
        print("PASSED: Good report accepted without warnings.")

    # Case B: Hallucinated Report
    # The LLM claims 5 anomalies when input has 0
    bad_json = json.dumps({
        "executive_summary": "We found 5 anomalies in the system.",
        "sections": [],
        "key_risks": [],
        "key_opportunities": []
    })
    mock_llm.complete.return_value = bad_json
    
    # Run Agent (Bad Case)
    print("\n[Test 2] Running with HALLUCINATED LLM output...")
    report = agent.generate_report(input_data)
    
    if "[AUTOMATED WARNING" in report.system_report.executive_summary:
        print("PASSED: Warning correctly appended to hallucinated report.")
        print(f"Warning text: {report.system_report.executive_summary.split('[AUTOMATED WARNING')[1]}")
    else:
        print("FAILED: No warning found in hallucinated report!")
        print(f"Summary was: {report.system_report.executive_summary}")
        sys.exit(1)

    print("\n--- Verification Complete: SUCCESS ---")

if __name__ == "__main__":
    test_reporting_flow()
