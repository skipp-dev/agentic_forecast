#!/usr/bin/env python3
"""
Demo: Email-Only Notifications with Guardrail Integration

This script demonstrates the email notification system that includes
guardrail decisions in the subject line and email body.

Shows how the system provides instant visibility into:
- Pipeline health status (OK/WARN/FAIL)
- Auto-promotion/auto-deployment decisions
- Detailed reasons for guardrail decisions
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from graphs.state import GraphState
from agents.notification_agent import NotificationAgent


def demo_email_notifications():
    """Demonstrate email notifications with guardrail integration."""

    print("📧 Demo: Email-Only Notifications with Guardrail Integration")
    print("=" * 70)
    print()

    # Step 1: Show configuration
    print("📋 Step 1: Email Configuration")
    print("-" * 40)

    config = {
        "notifications": {
            "enabled": True,
            "send_console": True,  # For demo - shows what would be emailed
            "send_email": False,   # Set to True when SMTP is configured
            "send_slack": False,
            "email": {
                "smtp_host": "smtp.gmail.com",
                "smtp_port": 587,
                "use_tls": True,
                "username": "forecast-bot@example.com",
                "password_env_var": "FORECAST_SMTP_PASSWORD",
                "from_addr": "Forecast Bot <forecast-bot@example.com>",
                "to_addrs": ["you@yourcompany.com", "team@yourcompany.com"],
                "subject_prefix": "[Forecast Daily Report]"
            }
        }
    }

    print("Config structure:")
    print("  notifications:")
    print("    enabled: true")
    print("    send_console: true  # For development")
    print("    send_email: false   # Set to true for production")
    print("    email:")
    print("      smtp_host: smtp.gmail.com")
    print("      username: forecast-bot@example.com")
    print("      password_env_var: FORECAST_SMTP_PASSWORD")
    print("      to_addrs: [you@yourcompany.com, team@yourcompany.com]")
    print()

    # Step 2: Show different guardrail scenarios
    print("🛡️  Step 2: Guardrail Decision Examples")
    print("-" * 40)

    scenarios = [
        {
            "name": "✅ HEALTHY SYSTEM (OK)",
            "decision": {
                "allow_auto_promotion": True,
                "allow_auto_deploy": True,
                "severity": "low",
                "reasons": ["All guardrail conditions satisfied - safe for automation"]
            }
        },
        {
            "name": "⚠️  MINOR ISSUES (WARN)",
            "decision": {
                "allow_auto_promotion": True,
                "allow_auto_deploy": False,
                "severity": "medium",
                "reasons": ["Cross-asset V2 recommends rollback - requires manual review"]
            }
        },
        {
            "name": "❌ CRITICAL ISSUES (FAIL)",
            "decision": {
                "allow_auto_promotion": False,
                "allow_auto_deploy": False,
                "severity": "high",
                "reasons": [
                    "metric_sanity_status=failed, severity=high",
                    "overall_score=0.45 < min_overall_score=0.60"
                ]
            }
        }
    ]

    agent = NotificationAgent(config)

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        decision = scenario['decision']

        # Show what the subject line would be
        status_tag = agent._status_tag_from_guardrails(decision)
        run_id = "demo-run-2025-01-28"
        subject = f"[Forecast Daily Report]{status_tag} run={run_id}"
        print(f"  Subject: {subject}")

        # Show key email header info
        allow_promo = decision.get("allow_auto_promotion", False)
        allow_deploy = decision.get("allow_auto_deploy", False)
        severity = decision.get("severity", "low")

        print(f"  Auto-promotion: {'YES' if allow_promo else 'NO'}")
        print(f"  Auto-deployment: {'YES' if allow_deploy else 'NO'}")
        print(f"  Severity: {severity.upper()}")
        print(f"  Reasons: {len(decision.get('reasons', []))} issue(s)")
    print()

    # Step 3: Simulate notification sending
    print("📧 Step 3: Notification Simulation")
    print("-" * 40)

    # Create test state with guardrail decision
    state = GraphState(
        run_id="demo-email-test-2025-01-28",
        symbols=["AAPL", "MSFT", "NVDA"],
        horizons=["1", "5"],
        guardrail_decision={
            "allow_auto_promotion": True,
            "allow_auto_deploy": True,
            "severity": "low",
            "reasons": ["All guardrail conditions satisfied - safe for automation"]
        }
    )

    # Simulate the notification node
    from src.graphs.main_graph import notification_node

    print("Running notification node...")
    result_state = notification_node(state)

    notification_result = getattr(result_state, 'notification_result', {})
    status = notification_result.get('status', 'unknown')
    channels = notification_result.get('channels', {})

    print(f"Notification status: {status.upper()}")
    print("Channels used:")
    for channel, result in channels.items():
        print(f"  • {channel}: {result}")
    print()

    # Step 4: Show production setup
    print("🚀 Step 4: Production Setup Instructions")
    print("-" * 40)

    print("1. Configure SMTP credentials:")
    print("   $env:FORECAST_SMTP_PASSWORD = 'your-actual-password'")
    print("   # Or set permanently in Windows environment variables")
    print()

    print("2. Update config.yaml with real email addresses:")
    print("   email:")
    print("     username: 'your-bot@yourdomain.com'")
    print("     from_addr: 'Forecast Bot <your-bot@yourdomain.com>'")
    print("     to_addrs:")
    print("       - 'you@yourcompany.com'")
    print("       - 'quant-team@yourcompany.com'")
    print()

    print("3. Enable email notifications:")
    print("   notifications:")
    print("     send_email: true")
    print("     send_console: false  # Optional")
    print()

    print("4. Test with:")
    print("   python demo_complete_workflow.py")
    print()

    # Step 5: Show what you'll see in your inbox
    print("📬 Step 5: What You'll See in Your Inbox")
    print("-" * 40)

    print("✅ GOOD DAY (OK):")
    print("Subject: [Forecast Daily Report][OK] run=2025-01-28T09-00-00Z")
    print("Body preview:")
    print("  Daily Forecast Health Report")
    print("  ")
    print("  Run ID: 2025-01-28T09-00-00Z")
    print("  Guardrail severity: LOW")
    print("  Auto-promotion allowed: YES")
    print("  Auto-deployment allowed: YES")
    print("  ")
    print("  Reasons:")
    print("  - All guardrail conditions satisfied - safe for automation")
    print("  ")
    print("  Full report follows below:")
    print("  ------------------------------------------------------------")
    print("  # Daily Forecast Health Report")
    print("  ... (complete Markdown report)")
    print()

    print("❌ BAD DAY (FAIL):")
    print("Subject: [Forecast Daily Report][FAIL] run=2025-01-28T09-00-00Z")
    print("Body preview:")
    print("  Daily Forecast Health Report")
    print("  ")
    print("  Run ID: 2025-01-28T09-00-00Z")
    print("  Guardrail severity: HIGH")
    print("  Auto-promotion allowed: NO")
    print("  Auto-deployment allowed: NO")
    print("  ")
    print("  Reasons:")
    print("  - metric_sanity_status=failed, severity=high")
    print("  - overall_score=0.45 < min_overall_score=0.60")
    print("  ")
    print("  Full report follows below:")
    print("  ------------------------------------------------------------")
    print("  # Daily Forecast Health Report")
    print("  ... (complete Markdown report with details)")
    print()

    print("🎯 Result: 2-second inbox triage → instant visibility into system health!")


if __name__ == "__main__":
    demo_email_notifications()