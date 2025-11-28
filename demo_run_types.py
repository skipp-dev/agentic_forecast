#!/usr/bin/env python3
"""
Demo: Run Type Labels in Forecasting System

This script demonstrates the run type labeling system that provides
clear identification of what kind of run just happened.

Shows how run_type flows from CLI → GraphState → daily report → emails.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from graphs.state import GraphState
from agents.notification_agent import NotificationAgent


def demo_run_types():
    """Demonstrate different run types and their email subjects."""

    print("🏷️  Demo: Run Type Labels in Forecasting System")
    print("=" * 60)
    print()

    # Step 1: Show CLI usage
    print("📋 Step 1: CLI Usage Examples")
    print("-" * 40)

    cli_examples = [
        ("DAILY", "Normal end-of-day run with data refresh and evaluation"),
        ("WEEKEND_HPO", "Expensive weekend run with extra HPO trials"),
        ("BACKTEST", "Offline historical validation (no auto-deployment)")
    ]

    print("Command examples:")
    for run_type, description in cli_examples:
        print(f"  python main.py --task daily_run --run_type {run_type}")
        print(f"    → {description}")
    print()

    # Step 2: Show how run_type flows through the system
    print("🔄 Step 2: Data Flow Through System")
    print("-" * 40)

    flow_steps = [
        "1. CLI: --run_type DAILY → args.run_type",
        "2. main.py: build_initial_state(..., run_type) → GraphState.run_type",
        "3. daily_health_report_node: state.run_type → run_metadata.run_type",
        "4. guardrail_decision_node: run_metadata.run_type → guardrail logic",
        "5. notification_node: run_metadata.run_type → email subject + header"
    ]

    for step in flow_steps:
        print(f"  {step}")
    print()

    # Step 3: Show email subjects for different scenarios
    print("📧 Step 3: Email Subjects by Run Type + Status")
    print("-" * 40)

    scenarios = [
        ("DAILY", "OK", "Normal healthy run"),
        ("DAILY", "WARN", "Minor issues detected"),
        ("DAILY", "FAIL", "Critical problems found"),
        ("WEEKEND_HPO", "OK", "Expensive tuning successful"),
        ("WEEKEND_HPO", "WARN", "Tuning found issues"),
        ("WEEKEND_HPO", "FAIL", "Tuning revealed problems"),
        ("BACKTEST", "OK", "Historical validation passed"),
        ("BACKTEST", "WARN", "Historical validation with warnings"),
        ("BACKTEST", "FAIL", "Historical validation failed")
    ]

    print("Subject format: [Forecast Daily Report][RUN_TYPE][STATUS] run=...")
    print()
    for run_type, status, description in scenarios:
        subject = f"[Forecast Daily Report][{run_type}][{status}] run=2025-01-28T09-00-00Z"
        print(f"  {subject}")
        print(f"    → {description}")
        print()

    # Step 4: Show guardrail behavior for BACKTEST
    print("🛡️  Step 4: Guardrail Behavior for BACKTEST")
    print("-" * 40)

    print("BACKTEST runs have special guardrail rules:")
    print("  • Auto-promotion: ALWAYS DISABLED")
    print("  • Auto-deployment: ALWAYS DISABLED")
    print("  • Severity: At least MEDIUM (prevents accidents)")
    print("  • Reason: 'run_type=BACKTEST: auto-promotion/deploy disabled by design'")
    print()

    backtest_decision = {
        "allow_auto_promotion": False,
        "allow_auto_deploy": False,
        "severity": "medium",
        "reasons": [
            "run_type=BACKTEST: auto-promotion/deploy disabled by design.",
            "All guardrail conditions satisfied - safe for automation"
        ]
    }

    print("Example BACKTEST guardrail decision:")
    print(f"  allow_auto_promotion: {backtest_decision['allow_auto_promotion']}")
    print(f"  allow_auto_deploy: {backtest_decision['allow_auto_deploy']}")
    print(f"  severity: {backtest_decision['severity'].upper()}")
    print("  reasons:")
    for reason in backtest_decision['reasons']:
        print(f"    - {reason}")
    print()

    # Step 5: Show email header examples
    print("📬 Step 5: Email Header Examples")
    print("-" * 40)

    headers = {
        "DAILY": {
            "run_type": "DAILY",
            "severity": "LOW",
            "allow_promo": True,
            "allow_deploy": True
        },
        "WEEKEND_HPO": {
            "run_type": "WEEKEND_HPO",
            "severity": "MEDIUM",
            "allow_promo": True,
            "allow_deploy": False
        },
        "BACKTEST": {
            "run_type": "BACKTEST",
            "severity": "MEDIUM",
            "allow_promo": False,
            "allow_deploy": False
        }
    }

    for run_type, data in headers.items():
        print(f"{run_type} Run Email Header:")
        print("  Daily Forecast Health Report")
        print()
        print(f"  Run ID: 2025-01-28T09-00-00Z")
        print(f"  Run type: {data['run_type']}")
        print(f"  Guardrail severity: {data['severity']}")
        print(f"  Auto-promotion allowed: {'YES' if data['allow_promo'] else 'NO'}")
        print(f"  Auto-deployment allowed: {'YES' if data['allow_deploy'] else 'NO'}")
        print()
        print("  Reasons:")
        if run_type == "BACKTEST":
            print("    - run_type=BACKTEST: auto-promotion/deploy disabled by design.")
        else:
            print("    - All guardrail conditions satisfied - safe for automation")
        print("  ... (full report follows)")
        print()

    # Step 6: Show daily report JSON structure
    print("📄 Step 6: Daily Report JSON Structure")
    print("-" * 40)

    sample_metadata = {
        "run_id": "2025-01-28T09-00-00Z",
        "run_type": "DAILY",
        "evaluated_at": "2025-01-28T09-02-15Z",
        "symbols": ["AAPL", "MSFT", "NVDA"],
        "horizons": ["1", "5", "10"],
        "config": {
            "feature_store_v2_enabled": True,
            "models_trained": ["AutoNHITS", "AutoTFT"],
            "hpo_trials_per_symbol": 20
        }
    }

    print("run_metadata section in daily_forecast_health_latest.json:")
    print("  {")
    for key, value in sample_metadata.items():
        if isinstance(value, list):
            print(f'    "{key}": {value},')
        elif isinstance(value, dict):
            print(f'    "{key}": {{...}},')
        else:
            print(f'    "{key}": "{value}",')
    print("  }")
    print()

    # Step 7: Testing instructions
    print("🧪 Step 7: Testing the Implementation")
    print("-" * 40)

    test_commands = [
        "# Test DAILY run (default)",
        "python main.py --task full --run_type DAILY",
        "",
        "# Test WEEKEND_HPO run",
        "python main.py --task full --run_type WEEKEND_HPO",
        "",
        "# Test BACKTEST run (should block auto-deploy)",
        "python main.py --task full --run_type BACKTEST",
        "",
        "# Check the daily report JSON",
        'cat results/reports/daily_forecast_health_latest.json | jq ".run_metadata"',
        "",
        "# Check email subject in logs",
        'grep -i "subject" logs/*.log'
    ]

    print("Testing commands:")
    for cmd in test_commands:
        if cmd.startswith("#"):
            print(f"  {cmd}")
        elif cmd.strip():
            print(f"  $ {cmd}")
        else:
            print()

    print("\n🎯 Result: Every run now clearly identifies WHAT it was, not just whether it passed!")


if __name__ == "__main__":
    demo_run_types()