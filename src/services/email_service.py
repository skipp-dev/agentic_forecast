# services/email_service.py
"""
Email service for sending reports and notifications.

Provides functionality to send formatted reports via email.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from datetime import datetime


class EmailService:
    """Service for sending emails with reports and notifications."""

    def __init__(self,
                 smtp_server: Optional[str] = None,
                 smtp_port: int = 587,
                 smtp_username: Optional[str] = None,
                 smtp_password: Optional[str] = None,
                 from_email: Optional[str] = None):
        """
        Initialize email service.

        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            smtp_username: SMTP username
            smtp_password: SMTP password
            from_email: From email address
        """
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'localhost')
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username or os.getenv('SMTP_USERNAME')
        self.smtp_password = smtp_password or os.getenv('SMTP_PASSWORD')
        self.from_email = from_email or os.getenv('FROM_EMAIL', 'reports@agenticforecast.com')

        # Check if we have valid SMTP configuration
        self.smtp_available = all([
            self.smtp_server,
            self.smtp_username,
            self.smtp_password
        ])

    def send_email_report(self,
                         subject: str,
                         html_content: str,
                         markdown_content: str,
                         recipients: List[str],
                         attachments: Optional[List[dict]] = None) -> bool:
        """
        Send a report via email.

        Args:
            subject: Email subject line
            html_content: HTML version of the report
            markdown_content: Markdown version of the report
            recipients: List of recipient email addresses
            attachments: Optional list of attachment dictionaries with 'filename' and 'content'

        Returns:
            True if email was sent successfully, False otherwise
        """
        if not self.smtp_available:
            print(f"SMTP not configured. Would send email to {recipients} with subject: {subject}")
            print("Email content (first 500 chars):")
            print(html_content[:500] + "..." if len(html_content) > 500 else html_content)
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)

            # Add timestamp
            msg['Date'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')

            # Attach HTML version
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # For now, we'll send HTML only. In a full implementation,
            # you might want to attach the markdown as a text alternative
            # or as an attachment.

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.from_email, recipients, msg.as_string())

            print(f"Email sent successfully to {recipients}")
            return True

        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

    def send_notification(self,
                         subject: str,
                         message: str,
                         recipients: List[str],
                         priority: str = 'normal') -> bool:
        """
        Send a simple notification email.

        Args:
            subject: Email subject
            message: Plain text message
            recipients: List of recipient email addresses
            priority: Email priority ('low', 'normal', 'high')

        Returns:
            True if email was sent successfully, False otherwise
        """
        if not self.smtp_available:
            print(f"SMTP not configured. Would send notification to {recipients}: {subject}")
            return False

        try:
            # Create message
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)

            # Set priority
            if priority == 'high':
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
            elif priority == 'low':
                msg['X-Priority'] = '5'
                msg['X-MSMail-Priority'] = 'Low'

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.from_email, recipients, msg.as_string())

            print(f"Notification sent successfully to {recipients}")
            return True

        except Exception as e:
            print(f"Failed to send notification: {e}")
            return False

    def test_connection(self) -> bool:
        """
        Test SMTP connection.

        Returns:
            True if connection successful, False otherwise
        """
        if not self.smtp_available:
            print("SMTP not configured - cannot test connection")
            return False

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
            print("SMTP connection test successful")
            return True
        except Exception as e:
            print(f"SMTP connection test failed: {e}")
            return False


# Global instance for easy access
_email_service = None

def get_email_service() -> EmailService:
    """Get the global email service instance."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service