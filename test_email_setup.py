#!/usr/bin/env python3
"""
Test Email Notification Setup

This script validates that the email notification system is properly configured
and can send test emails with guardrail status indicators.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from agents.notification_agent import NotificationAgent


def test_email_setup():
    """Test email notification setup and configuration."""

    print("🧪 Testing Email Notification Setup")
    print("=" * 50)
    print()

    # Check if SMTP password is configured
    smtp_password = os.getenv('FORECAST_SMTP_PASSWORD')
    if not smtp_password:
        print("❌ FORECAST_SMTP_PASSWORD environment variable not set")
        print("   To test email functionality, set the password:")
        print("   $env:FORECAST_SMTP_PASSWORD = 'your-smtp-password'")
        print()
        return False

    print("✅ SMTP password is configured")

    # Load config
    try:
        import yaml
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✅ Config file loaded successfully")
        else:
            print("❌ config.yaml not found")
            return False
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

    # Check email configuration
    email_config = config.get('notifications', {}).get('email', {})
    required_fields = ['smtp_host', 'smtp_port', 'username', 'from_addr', 'to_addrs']

    missing_fields = []
    for field in required_fields:
        if not email_config.get(field):
            missing_fields.append(field)

    if missing_fields:
        print(f"❌ Missing email config fields: {', '.join(missing_fields)}")
        print("   Update config.yaml with your email settings")
        return False

    print("✅ Email configuration is complete")

    # Test notification agent initialization
    try:
        agent = NotificationAgent(config)
        print("✅ NotificationAgent initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing NotificationAgent: {e}")
        return False

    # Test guardrail status tag generation
    test_decisions = [
        {"allow_auto_promotion": True, "allow_auto_deploy": True, "severity": "low"},
        {"allow_auto_promotion": True, "allow_auto_deploy": False, "severity": "medium"},
        {"allow_auto_promotion": False, "allow_auto_deploy": False, "severity": "high"}
    ]

    print("\n🛡️  Testing Guardrail Status Tags:")
    for decision in test_decisions:
        tag = agent._status_tag_from_guardrails(decision)
        severity = decision['severity'].upper()
        print(f"   {severity} severity → {tag}")

    # Test email sending (dry run)
    print("\n📧 Testing Email Generation (Dry Run):")

    test_run_metadata = {
        "run_id": "test-email-run-2025-01-28",
        "symbols": ["AAPL", "MSFT"],
        "horizons": ["1", "5"]
    }

    test_guardrail_decision = {
        "allow_auto_promotion": True,
        "allow_auto_deploy": True,
        "severity": "low",
        "reasons": ["All guardrail conditions satisfied - safe for automation"]
    }

    # Temporarily enable email for testing
    test_config = config.copy()
    test_config['notifications']['send_email'] = True
    test_config['notifications']['send_console'] = False

    test_agent = NotificationAgent(test_config)

    try:
        # This will attempt to send email if configured
        result = test_agent.notify_daily_report(
            run_metadata=test_run_metadata,
            guardrail_decision=test_guardrail_decision
        )

        if result.get('status') == 'done':
            channels = result.get('channels', {})
            if 'email' in channels:
                print("✅ Email sent successfully!")
                print(f"   Status: {channels['email']}")
            else:
                print("⚠️  Email channel not used (check config)")
        else:
            print(f"❌ Email sending failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error during email test: {e}")
        print("   This might be expected if SMTP credentials are invalid")
        print("   Check your email provider settings and credentials")

    print("\n📋 Next Steps:")
    print("1. Verify the test email was received")
    print("2. Check spam folder if not found in inbox")
    print("3. Update config.yaml with correct email addresses")
    print("4. Run the full workflow: python demo_complete_workflow.py")

    return True


if __name__ == "__main__":
    success = test_email_setup()
    if success:
        print("\n🎉 Email setup test completed!")
    else:
        print("\n⚠️  Email setup needs configuration before testing")
        sys.exit(1)