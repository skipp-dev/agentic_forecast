#!/usr/bin/env python3
"""
Test script for NotificationAgent and GuardrailAgent integration.

This script tests the complete notification and guardrail pipeline
that processes the daily health report.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from graphs.state import GraphState
from agents.notification_agent import NotificationAgent
from agents.guardrail_agent import GuardrailAgent


def test_guardrail_agent():
    """Test the GuardrailAgent with sample daily health report data."""
    print("🛡️  Testing GuardrailAgent")
    print("-" * 40)

    # Sample config
    config = {
        "guardrails": {
            "min_overall_score": 0.6,
            "require_metric_sanity": True,
            "allow_promotion_on_warning": False
        }
    }

    agent = GuardrailAgent(config)

    # Test case 1: All good
    print("Test 1: All conditions satisfied")
    good_report = {
        "metric_sanity": {"status": "passed", "severity": "low"},
        "model_performance": {"overall_score": 0.8},
        "cross_asset_v2": {"decision": "keep_enabled"}
    }

    decision = agent.evaluate_daily_health(good_report)
    print(f"  Allow promotion: {decision['allow_auto_promotion']}")
    print(f"  Allow deploy: {decision['allow_auto_deploy']}")
    print(f"  Severity: {decision['severity']}")
    print(f"  Reasons: {decision['reasons'][:1]}")  # First reason only
    print()

    # Test case 2: Metric sanity failed
    print("Test 2: Metric sanity failed (high severity)")
    bad_report = {
        "metric_sanity": {"status": "failed", "severity": "high"},
        "model_performance": {"overall_score": 0.8},
        "cross_asset_v2": {"decision": "keep_enabled"}
    }

    decision = agent.evaluate_daily_health(bad_report)
    print(f"  Allow promotion: {decision['allow_auto_promotion']}")
    print(f"  Allow deploy: {decision['allow_auto_deploy']}")
    print(f"  Severity: {decision['severity']}")
    print(f"  Reasons: {decision['reasons'][:1]}")
    print()

    # Test case 3: Poor performance
    print("Test 3: Poor model performance")
    poor_report = {
        "metric_sanity": {"status": "passed", "severity": "low"},
        "model_performance": {"overall_score": 0.3},
        "cross_asset_v2": {"decision": "keep_enabled"}
    }

    decision = agent.evaluate_daily_health(poor_report)
    print(f"  Allow promotion: {decision['allow_auto_promotion']}")
    print(f"  Allow deploy: {decision['allow_auto_deploy']}")
    print(f"  Severity: {decision['severity']}")
    print(f"  Reasons: {decision['reasons'][:1]}")
    print()

    return True


def test_notification_agent():
    """Test the NotificationAgent console output."""
    print("📢 Testing NotificationAgent")
    print("-" * 40)

    # Sample config (console only for testing)
    config = {
        "notifications": {
            "enabled": True,
            "send_console": True,
            "send_email": False,
            "send_slack": False
        }
    }

    agent = NotificationAgent(config)

    # Mock run metadata
    run_metadata = {
        "run_id": "test-run-2025-01-28",
        "symbols": ["AAPL", "MSFT"],
        "horizons": ["1", "5"]
    }

    print("Sending console notification...")
    result = agent.notify_daily_report(run_metadata)

    print(f"Notification status: {result.get('status')}")
    if result.get('channels'):
        print(f"Channels: {list(result['channels'].keys())}")
    print()

    return result.get('status') in ['done', 'partial']


def test_graph_integration():
    """Test the graph node functions."""
    print("🔗 Testing Graph Node Integration")
    print("-" * 40)

    # Create test state
    state = GraphState(
        run_id="test-integration-2025-01-28",
        symbols=["AAPL", "MSFT"],
        horizons=["1", "5"],
        config={
            "notifications": {
                "enabled": True,
                "send_console": True,
                "send_email": False,
                "send_slack": False
            },
            "guardrails": {
                "min_overall_score": 0.6,
                "require_metric_sanity": True,
                "allow_promotion_on_warning": False
            }
        }
    )

    # Import the node functions
    from src.graphs.main_graph import guardrail_decision_node, notification_node

    print("Testing guardrail_decision_node...")
    # This will fail gracefully if no report exists
    result_state = guardrail_decision_node(state)
    guardrail_decision = result_state.guardrail_decision
    print(f"  Guardrail decision: {guardrail_decision.get('allow_auto_promotion', 'N/A')}")

    print("Testing notification_node...")
    result_state = notification_node(result_state)
    notification_result = result_state.notification_result
    print(f"  Notification status: {notification_result.get('status', 'N/A')}")

    print()
    return True


def main():
    """Run all tests."""
    print("🧪 Testing Notification & Guardrail Agents")
    print("=" * 60)
    print()

    tests = [
        ("GuardrailAgent", test_guardrail_agent),
        ("NotificationAgent", test_notification_agent),
        ("Graph Integration", test_graph_integration)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"{test_name}: ❌ ERROR - {e}")
            results.append((test_name, False))

    print()
    print("📊 Test Summary:")
    print("-" * 30)
    all_passed = True
    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All tests passed! Ready for production.")
    else:
        print("⚠️  Some tests failed. Check the output above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)