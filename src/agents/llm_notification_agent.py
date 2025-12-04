from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import json
from langsmith import traceable
from src.utils.llm_utils import extract_json_from_llm_output

logger = logging.getLogger(__name__)

@dataclass
class NotificationInput:
    alerts_data: Dict[str, Any]
    guardrail_context: Dict[str, Any]
    recipient_context: Dict[str, Any]
    channel: str

@dataclass
class NotificationOutput:
    channel_messages: List[Dict[str, Any]]
    meta: Dict[str, Any]

class LLMNotificationAgent:
    """
    Agent that converts technical alerts and metrics into human-friendly notifications.
    """
    def __init__(self, settings=None):
        # Use the new LLM factory
        from src.llm.llm_factory import create_notification_agent_llm
        self.llm = create_notification_agent_llm()
        self.settings = settings or {}

    @traceable(
        name="notification_agent_generate_notifications",
        tags=["notification", "llm", "alerts"],
        metadata={"role": "notification_agent"}
    )
    def generate_notifications(self, notification_input: NotificationInput) -> NotificationOutput:
        """
        Generate human-friendly notifications from technical alerts.
        This call is traced to LangSmith.
        """
        # Correct import path for prompts
        from src.configs.llm_prompts import PROMPTS, build_notification_agent_user_prompt

        system_prompt = PROMPTS["notification_agent"]
        user_prompt = build_notification_agent_user_prompt(
            alerts_data=notification_input.alerts_data,
            guardrail_context=notification_input.guardrail_context,
            recipient_context=notification_input.recipient_context,
            channel=notification_input.channel
        )

        logger.info("Calling LLM for notification generation (LangSmith tracing enabled)")

        raw = self.llm.complete(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.1,  # Low temperature for consistent alert formatting
            max_tokens=1500,
        )

        logger.info(f"Raw LLM response (first 500 chars): {raw[:500]}")

        try:
            json_str = extract_json_from_llm_output(raw)
            data = json.loads(json_str)
            logger.info("Successfully parsed LLM response as JSON")
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}. Raw response: {raw}")
            # Fallback: create basic notification structure matching prompt schema
            data = {
                "channel_messages": [
                    {
                        "channel": notification_input.channel,
                        "text": f"System Alert: Alert processing encountered an error. Raw response: {raw[:100]}..."
                    }
                ],
                "meta": {
                    "symbol": "UNKNOWN",
                    "horizon": 0,
                    "alert_name": "ParseError",
                    "severity": "warning"
                }
            }

        # Validate and filter
        valid_keys = NotificationOutput.__annotations__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        
        # Ensure required fields
        if "channel_messages" not in filtered:
            filtered["channel_messages"] = []
        if "meta" not in filtered:
            filtered["meta"] = {}

        return NotificationOutput(**filtered)

    def format_channel_message(self, message_data: Dict[str, Any]) -> str:
        """
        Format a notification message for the specified channel.
        """
        channel = message_data.get("channel", "console")
        # Map 'text' from prompt to 'message' for internal formatting if needed, 
        # or just use 'text' directly.
        message = message_data.get("text", message_data.get("message", ""))
        
        # Optional fields that might not be in the simple prompt schema but could be added later
        priority = message_data.get("priority", "info")
        title = message_data.get("title", "System Notification")
        actions = message_data.get("actions", [])
        metadata = message_data.get("metadata", {})

        if channel.lower() == "slack":
            return self._format_slack_message(priority, title, message, actions, metadata)
        elif channel.lower() == "email":
            return self._format_email_message(priority, title, message, actions, metadata)
        else:
            return self._format_console_message(priority, title, message, actions, metadata)

    def _format_slack_message(self, priority: str, title: str, message: str,
                            actions: list, metadata: Dict[str, Any]) -> str:
        """Format message for Slack."""
        emoji_map = {
            "critical": "🚨",
            "warning": "⚠️",
            "info": "ℹ️"
        }
        emoji = emoji_map.get(priority, "📢")

        slack_msg = f"{emoji} *{title}*\n\n{message}"

        if actions:
            slack_msg += "\n\n*Recommended Actions:*\n" + "\n".join(f"• {action}" for action in actions)

        return slack_msg

    def _format_email_message(self, priority: str, title: str, message: str,
                            actions: list, metadata: Dict[str, Any]) -> str:
        """Format message for email."""
        priority_indicator = {
            "critical": "[CRITICAL]",
            "warning": "[WARNING]",
            "info": "[INFO]"
        }.get(priority, "[ALERT]")

        email_msg = f"Subject: {priority_indicator} {title}\n\n"
        email_msg += f"Priority: {priority.upper()}\n\n"
        email_msg += f"{message}\n\n"

        if actions:
            email_msg += "Recommended Actions:\n"
            for i, action in enumerate(actions, 1):
                email_msg += f"{i}. {action}\n"
            email_msg += "\n"

        return email_msg

    def _format_console_message(self, priority: str, title: str, message: str,
                              actions: list, metadata: Dict[str, Any]) -> str:
        """Format message for console output."""
        border = "=" * 60
        priority_banner = {
            "critical": "🚨 CRITICAL ALERT 🚨",
            "warning": "⚠️  WARNING  ⚠️",
            "info": "ℹ️  INFORMATION  ℹ️"
        }.get(priority, "📢 NOTIFICATION 📢")

        console_msg = f"\n{border}\n{priority_banner}\n{border}\n\n"
        console_msg += f"Title: {title}\n\n"
        console_msg += f"{message}\n\n"

        if actions:
            console_msg += "Recommended Actions:\n"
            for i, action in enumerate(actions, 1):
                console_msg += f"  {i}. {action}\n"
            console_msg += "\n"

        console_msg += f"{border}\n"
        return console_msg