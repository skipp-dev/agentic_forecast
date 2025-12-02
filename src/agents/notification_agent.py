from pathlib import Path
from typing import Dict, Any
import os
import smtplib
import requests
import json
import logging

logger = logging.getLogger(__name__)


class NotificationAgent:
    """
    Agent responsible for sending daily forecast health reports via various channels.

    Supports console, email, and Slack notifications with config-driven behavior.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config.get("notifications", {})
        self.enabled = self.cfg.get("enabled", True)

    def notify_daily_report(self, run_metadata: Dict[str, Any], guardrail_decision: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Send the daily health report via configured channels.
        Expects the unified report files to already exist.
        """
        if not self.enabled:
            logger.info("Notifications disabled in config")
            return {"status": "skipped", "reason": "notifications_disabled"}

        json_path = Path("results/reports/daily_forecast_health_latest.json")
        md_path = Path("results/reports/daily_forecast_health_latest.md")

        if not json_path.exists() or not md_path.exists():
            logger.warning("Daily report files missing - cannot send notifications")
            return {
                "status": "failed",
                "reason": "daily report files missing",
                "json_exists": json_path.exists(),
                "md_exists": md_path.exists(),
            }

        try:
            md_text = md_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")
            json_data = json.loads(json_text)
        except Exception as e:
            logger.error(f"Failed to read report files: {e}")
            return {"status": "failed", "reason": f"file_read_error: {str(e)}"}

        results = {}
        errors = []

        # Send via configured channels
        if self.cfg.get("send_console", True):
            try:
                results["console"] = self._send_console(md_text)
            except Exception as e:
                logger.error(f"Console notification failed: {e}")
                errors.append(f"console: {str(e)}")

        if self.cfg.get("send_email", False):
            try:
                results["email"] = self._send_email(md_text, run_metadata, guardrail_decision)
            except Exception as e:
                logger.error(f"Email notification failed: {e}")
                errors.append(f"email: {str(e)}")

        if self.cfg.get("send_slack", False):
            try:
                results["slack"] = self._send_slack(md_text, run_metadata, json_data)
            except Exception as e:
                logger.error(f"Slack notification failed: {e}")
                errors.append(f"slack: {str(e)}")

        status = "done" if not errors else "partial"
        if errors:
            results["errors"] = errors

        logger.info(f"Notification completed with status: {status}")
        return {"status": status, "channels": results}

    def _status_tag_from_guardrails(self, decision: Dict[str, Any] | None) -> str:
        """Derive status tag from guardrail decision."""
        if not decision:
            return "[UNKNOWN]"

        allow_deploy = decision.get("allow_auto_deploy", False)
        severity = decision.get("severity", "low")

        if allow_deploy:
            return "[OK]"
        if severity == "high":
            return "[FAIL]"
        return "[WARN]"

    def _send_console(self, md_text: str) -> str:
        """Send notification to console output."""
        print("\n" + "="*60)
        print("DAILY FORECAST HEALTH REPORT")
        print("="*60)
        print(md_text)
        print("="*60 + "\n")
        return "printed"

    def _send_email(self, md_text: str, run_metadata: Dict[str, Any], guardrail_decision: Dict[str, Any] | None) -> str:
        """Send notification via email."""
        cfg_email = self.cfg.get("email", {})
        password = os.getenv(cfg_email.get("password_env_var", ""), "")

        if not password:
            logger.warning("Email password not found in environment variables")
            return "skipped_missing_password"

        # Build subject with status tag
        subject_prefix = cfg_email.get("subject_prefix", "[Forecast Daily Report]")
        run_id = run_metadata.get("run_id", "unknown-run")
        run_type = run_metadata.get("run_type", "DAILY").upper()
        status_tag = self._status_tag_from_guardrails(guardrail_decision)
        subject = f"{subject_prefix}[{run_type}]{status_tag} run={run_id}"

        # Build header summary with guardrail information
        if guardrail_decision:
            allow_promo = guardrail_decision.get("allow_auto_promotion", False)
            allow_deploy = guardrail_decision.get("allow_auto_deploy", False)
            severity = guardrail_decision.get("severity", "low")
            reasons = guardrail_decision.get("reasons", [])
        else:
            allow_promo = False
            allow_deploy = False
            severity = "unknown"
            reasons = ["No guardrail decision available."]

        header_lines = [
            "Daily Forecast Health Report",
            "",
            f"Run ID: {run_id}",
            f"Run type: {run_type}",
            f"Guardrail severity: {severity.upper()}",
            f"Auto-promotion allowed: {'YES' if allow_promo else 'NO'}",
            f"Auto-deployment allowed: {'YES' if allow_deploy else 'NO'}",
            "",
            "Reasons:"
        ]
        header_lines += [f"- {r}" for r in reasons[:5]]  # Limit to first 5 reasons
        header_lines += [
            "",
            "Full report follows below:",
            "",
            "------------------------------------------------------------",
            ""
        ]

        header_text = "\n".join(header_lines)
        body_text = header_text + md_text

        # Build email message
        from_addr = cfg_email.get("from_addr", "")
        to_addrs = cfg_email.get("to_addrs", [])

        if not from_addr or not to_addrs:
            logger.error("Email configuration incomplete - missing from_addr or to_addrs")
            return "skipped_config_incomplete"

        msg_parts = [
            f"Subject: {subject}",
            f"From: {from_addr}",
            f"To: {', '.join(to_addrs)}",
            "Content-Type: text/plain; charset=utf-8",
            "",
            body_text
        ]
        msg = "\n".join(msg_parts)

        # Send email
        try:
            server = smtplib.SMTP(cfg_email["smtp_host"], cfg_email["smtp_port"])
            if cfg_email.get("use_tls", True):
                server.starttls()
            server.login(cfg_email["username"], password)
            server.sendmail(from_addr, to_addrs, msg.encode("utf-8"))
            server.quit()
            logger.info(f"Email sent successfully to {len(to_addrs)} recipients")
            return "sent"
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            raise

    def _send_slack(self, md_text: str, run_metadata: Dict[str, Any], json_data: Dict[str, Any]) -> str:
        """Send notification via Slack webhook."""
        cfg_slack = self.cfg.get("slack", {})
        webhook_url = os.getenv(cfg_slack.get("webhook_env_var", ""), "")

        if not webhook_url:
            logger.warning("Slack webhook URL not found in environment variables")
            return "skipped_missing_webhook"

        run_id = run_metadata.get("run_id", "unknown-run")
        title = f"Daily Forecast Health Report – {run_id}"

        # Extract key metrics for Slack message
        metric_sanity = json_data.get("metric_sanity", {})
        performance = json_data.get("model_performance", {})
        cross_v2 = json_data.get("cross_asset_v2", {})
        alerts = json_data.get("alerts_and_decisions", {})

        # Build concise Slack message
        sanity_status = "✅ PASSED" if metric_sanity.get("status") == "passed" else "❌ FAILED"
        overall_score = performance.get("overall_score", 0.0)
        v2_decision = cross_v2.get("decision", "unknown").replace("_", " ").title()
        can_promote = "✅ YES" if alerts.get("can_auto_promote_models") else "❌ NO"

        slack_text = f"""*{title}*

• *Metric Sanity:* {sanity_status}
• *Overall Score:* {overall_score:.3f}
• *V2 Decision:* {v2_decision}
• *Auto-Promotion:* {can_promote}

*Key Findings:*
{self._extract_key_findings(json_data)}

*Full Report:* See attached Markdown or JSON files.
"""

        # Truncate if too long for Slack
        if len(slack_text) > 3500:
            slack_text = slack_text[:3497] + "..."

        payload = {
            "username": cfg_slack.get("username", "ForecastBot"),
            "icon_emoji": cfg_slack.get("icon_emoji", ":bar_chart:"),
            "text": slack_text,
            "channel": cfg_slack.get("channel", "#forecast-alerts")
        }

        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Slack notification sent successfully (status: {response.status_code})")
            return f"status_{response.status_code}"
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            raise

    def _extract_key_findings(self, json_data: Dict[str, Any]) -> str:
        """Extract key findings for Slack message."""
        findings = []

        # Metric sanity findings
        sanity = json_data.get("metric_sanity", {})
        if sanity.get("key_findings"):
            findings.extend(sanity["key_findings"][:2])  # Top 2 findings

        # Performance summary
        perf = json_data.get("model_performance", {})
        if perf.get("overall_score", 0) > 0.7:
            findings.append("Strong model performance across metrics")
        elif perf.get("overall_score", 0) < 0.5:
            findings.append("Concerning model performance - review needed")

        # V2 insights
        v2 = json_data.get("cross_asset_v2", {})
        if v2.get("decision") == "keep_enabled":
            findings.append("V2 features providing positive impact")

        # Guardrail decision
        alerts = json_data.get("alerts_and_decisions", {})
        if not alerts.get("can_auto_promote_models"):
            findings.append("Auto-promotion blocked - manual review required")

        if not findings:
            findings.append("All systems operating normally")

        return "\n".join(f"• {finding}" for finding in findings[:3])  # Max 3 findings