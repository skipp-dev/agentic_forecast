#!/usr/bin/env python3
"""
Demo: Complete Daily Run with Notifications & Guardrails

This script demonstrates the complete daily forecasting workflow:
1. Generate unified health report
2. Evaluate guardrail conditions
3. Send notifications
4. Show final status

This represents the "last mile" - turning powerful AI into comfortable daily use.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from graphs.state import GraphState
from daily_health_report import daily_health_report_node
from src.graphs.main_graph import guardrail_decision_node, notification_node


def demo_complete_workflow():
    """Demonstrate the complete daily workflow with notifications and guardrails."""

    print("🚀 Demo: Complete Daily Forecasting Workflow")
    print("=" * 70)
    print()

    # Step 1: Initialize state
    print("📋 Step 1: Initializing workflow state")
    state = GraphState(
        run_id="demo-daily-run-2025-01-28",
        symbols=["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"],
        horizons=["1", "5", "10"],
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
    print(f"  Run ID: {state.run_id}")
    print(f"  Symbols: {len(state.symbols)} assets")
    print(f"  Horizons: {state.horizons}")
    print()

    # Step 2: Generate daily health report
    print("📊 Step 2: Generating unified daily health report")
    print("  (This would normally follow evaluation, sanity checks, and V2 analysis)")
    state = daily_health_report_node(state)

    json_path = getattr(state, 'daily_report_path_json', 'N/A')
    md_path = getattr(state, 'daily_report_path_md', 'N/A')
    print(f"  ✅ JSON Report: {json_path}")
    print(f"  ✅ Markdown Report: {md_path}")
    print()

    # Step 3: Evaluate guardrails
    print("🛡️  Step 3: Evaluating guardrail conditions")
    state = guardrail_decision_node(state)

    guardrail_decision = getattr(state, 'guardrail_decision', {})
    allow_promotion = guardrail_decision.get('allow_auto_promotion', False)
    allow_deploy = guardrail_decision.get('allow_auto_deploy', False)
    severity = guardrail_decision.get('severity', 'unknown')
    reasons = guardrail_decision.get('reasons', [])

    print(f"  Auto-promotion: {'✅ ALLOWED' if allow_promotion else '❌ BLOCKED'}")
    print(f"  Auto-deployment: {'✅ ALLOWED' if allow_deploy else '❌ BLOCKED'}")
    print(f"  Severity: {severity.upper()}")
    print("  Reasons:")
    for reason in reasons[:2]:  # Show first 2 reasons
        print(f"    • {reason}")
    print()

    # Step 4: Send notifications
    print("📢 Step 4: Sending notifications")
    state = notification_node(state)

    notification_result = getattr(state, 'notification_result', {})
    status = notification_result.get('status', 'unknown')
    channels = notification_result.get('channels', {})

    print(f"  Status: {status.upper()}")
    print("  Channels:")
    for channel, result in channels.items():
        print(f"    • {channel}: {result}")
    print()

    # Step 5: Final summary
    print("🎯 Step 5: Workflow Complete - Final Status")
    print("-" * 50)

    # Check if files exist
    json_exists = Path(json_path).exists() if json_path != 'N/A' else False
    md_exists = Path(md_path).exists() if md_path != 'N/A' else False

    print("📄 Generated Files:")
    print(f"  JSON Report: {'✅' if json_exists else '❌'} {json_path}")
    print(f"  Markdown Report: {'✅' if md_exists else '❌'} {md_path}")
    print()

    print("🤖 Automation Status:")
    print(f"  Can Auto-Promote Models: {'✅ YES' if allow_promotion else '❌ NO'}")
    print(f"  Can Auto-Deploy: {'✅ YES' if allow_deploy else '❌ NO'}")
    print()

    print("💼 Business Impact:")
    if allow_promotion and allow_deploy:
        print("  ✅ System is healthy - proceeding with automated operations")
        print("  ✅ Models can be promoted and deployed automatically")
        print("  ✅ Stakeholders notified via configured channels")
    elif allow_promotion:
        print("  ⚠️  Models can be promoted but deployment requires manual review")
        print("  ⚠️  Check guardrail reasons for deployment blocking")
    else:
        print("  ❌ Automation blocked - manual intervention required")
        print("  ❌ Check guardrail reasons and resolve issues before proceeding")
    print()

    print("🔧 Next Steps:")
    print("  • Review detailed reports in results/reports/")
    print("  • Check logs for any warnings or errors")
    print("  • Configure email/Slack notifications for production")
    print("  • Set up automated scheduling (cron, Airflow, etc.)")
    print()

    return allow_promotion and allow_deploy


def show_configuration_guide():
    """Show how to configure notifications and guardrails."""

    print("\n⚙️  Configuration Guide")
    print("-" * 30)

    print("1. Enable Console Notifications (Development):")
    print("   config.yaml -> notifications.send_console: true")
    print()

    print("2. Enable Email Notifications (Production):")
    print("   config.yaml -> notifications.send_email: true")
    print("   Set environment variable: FORECAST_SMTP_PASSWORD=your_password")
    print("   Update SMTP settings in config.yaml")
    print()

    print("3. Enable Slack Notifications:")
    print("   config.yaml -> notifications.send_slack: true")
    print("   Set environment variable: FORECAST_SLACK_WEBHOOK_URL=https://...")
    print()

    print("4. Tune Guardrail Sensitivity:")
    print("   config.yaml -> guardrails.min_overall_score: 0.6  # Lower = more permissive")
    print("   config.yaml -> guardrails.allow_promotion_on_warning: false  # Strict mode")
    print()

    print("5. Environment Variables Needed:")
    print("   FORECAST_SMTP_PASSWORD=your_smtp_password")
    print("   FORECAST_SLACK_WEBHOOK_URL=https://hooks.slack.com/...")

    print()


if __name__ == "__main__":
    success = demo_complete_workflow()
    show_configuration_guide()

    print("🎉 Demo completed!")
    if success:
        print("✅ The system is ready for automated daily operations.")
    else:
        print("⚠️  The system requires attention before automated operations.")

    sys.exit(0 if success else 1)