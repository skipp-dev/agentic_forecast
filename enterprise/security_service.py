"""
Enterprise Security Service

Comprehensive security features for IB Forecast system.
Provides audit logging, compliance monitoring, and security controls.
"""

import os
import sys
import json
import hashlib
import hmac
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncio
import aiofiles
from pathlib import Path
import sqlite3
import pandas as pd
from cryptography.fernet import Fernet
import redis
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class SecurityEvent:
    """Security event types."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    USER_CREATED = "user_created"
    USER_DELETED = "user_deleted"
    USER_ROLE_CHANGED = "user_role_changed"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    CONFIG_CHANGE = "config_change"
    SYSTEM_ALERT = "system_alert"

class ComplianceFramework:
    """Compliance framework enumeration."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"

class EnterpriseSecurityService:
    """
    Enterprise security service.

    Provides:
    - Comprehensive audit logging
    - Compliance monitoring
    - Security incident detection
    - Data protection and encryption
    - Access control and monitoring
    - Security alerting and reporting
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize enterprise security service.

        Args:
            config: Service configuration
        """
        self.config = config or {
            'audit_log_path': 'logs/audit.db',
            'encryption_key': os.getenv('ENCRYPTION_KEY', Fernet.generate_key()),
            'redis_url': 'redis://localhost:6379',
            'alert_email': os.getenv('ALERT_EMAIL'),
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'smtp_username': os.getenv('SMTP_USERNAME'),
            'smtp_password': os.getenv('SMTP_PASSWORD'),
            'max_failed_logins': 5,
            'alert_thresholds': {
                'failed_logins_per_hour': 10,
                'suspicious_ips_per_hour': 5,
                'unauthorized_access_per_hour': 3
            },
            'data_retention_days': 2555,  # 7 years for compliance
            'compliance_frameworks': ['gdpr', 'sox']
        }

        # Initialize components
        self.fernet = Fernet(self.config['encryption_key'])
        self.redis_client = redis.Redis.from_url(self.config['redis_url'])

        # Setup audit database
        self._setup_audit_database()

        # Security monitoring
        self.security_alerts = []
        self.active_sessions = {}

        logger.info("Enterprise Security Service initialized")

    def _setup_audit_database(self):
        """Setup audit logging database."""
        audit_dir = Path(self.config['audit_log_path']).parent
        audit_dir.mkdir(exist_ok=True)

        self.audit_conn = sqlite3.connect(self.config['audit_log_path'])

        # Create audit table
        self.audit_conn.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                username TEXT,
                ip_address TEXT,
                user_agent TEXT,
                resource TEXT,
                action TEXT,
                details TEXT,
                severity TEXT,
                compliance_flags TEXT
            )
        ''')

        # Create indexes for performance
        self.audit_conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)')
        self.audit_conn.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_log(event_type)')
        self.audit_conn.execute('CREATE INDEX IF NOT EXISTS idx_username ON audit_log(username)')

        self.audit_conn.commit()

    def log_security_event(self, event_type: str, username: str = None,
                          ip_address: str = None, user_agent: str = None,
                          resource: str = None, action: str = None,
                          details: Dict[str, Any] = None, severity: str = 'info'):
        """
        Log a security event.

        Args:
            event_type: Type of security event
            username: Username involved
            ip_address: IP address of the request
            user_agent: User agent string
            resource: Resource accessed
            action: Action performed
            details: Additional event details
            severity: Event severity (info, warning, error, critical)
        """
        timestamp = datetime.now().isoformat()

        # Determine compliance flags
        compliance_flags = self._determine_compliance_flags(event_type, details or {})

        # Prepare audit entry
        audit_entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'username': username,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'resource': resource,
            'action': action,
            'details': json.dumps(details) if details else None,
            'severity': severity,
            'compliance_flags': ','.join(compliance_flags)
        }

        # Insert into database
        self.audit_conn.execute('''
            INSERT INTO audit_log
            (timestamp, event_type, username, ip_address, user_agent, resource, action, details, severity, compliance_flags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            audit_entry['timestamp'], audit_entry['event_type'], audit_entry['username'],
            audit_entry['ip_address'], audit_entry['user_agent'], audit_entry['resource'],
            audit_entry['action'], audit_entry['details'], audit_entry['severity'],
            audit_entry['compliance_flags']
        ))

        self.audit_conn.commit()

        # Check for security alerts
        self._check_security_alerts(audit_entry)

        # Log to application logger
        log_message = f"Security Event: {event_type}"
        if username:
            log_message += f" by {username}"
        if ip_address:
            log_message += f" from {ip_address}"

        if severity == 'critical':
            logger.critical(log_message)
        elif severity == 'error':
            logger.error(log_message)
        elif severity == 'warning':
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def _determine_compliance_flags(self, event_type: str, details: Dict[str, Any]) -> List[str]:
        """Determine which compliance frameworks this event affects."""
        flags = []

        # GDPR compliance
        if event_type in ['data_access', 'user_created', 'user_deleted', 'password_change']:
            flags.append('gdpr')

        # SOX compliance
        if event_type in ['config_change', 'user_role_changed', 'unauthorized_access']:
            flags.append('sox')

        # HIPAA compliance (if handling health data)
        if 'health_data' in details:
            flags.append('hipaa')

        # PCI DSS compliance (if handling payment data)
        if 'payment_data' in details:
            flags.append('pci_dss')

        return flags

    def _check_security_alerts(self, audit_entry: Dict[str, Any]):
        """Check for security alerts based on audit entry."""
        event_type = audit_entry['event_type']
        username = audit_entry['username']
        ip_address = audit_entry['ip_address']

        # Failed login monitoring
        if event_type == 'login_failure':
            self._monitor_failed_logins(username, ip_address)

        # Unauthorized access monitoring
        if event_type == 'unauthorized_access':
            self._monitor_unauthorized_access(ip_address)

        # Suspicious activity detection
        if event_type == 'suspicious_activity':
            self._handle_suspicious_activity(audit_entry)

    def _monitor_failed_logins(self, username: str, ip_address: str):
        """Monitor failed login attempts."""
        key = f"failed_logins:{ip_address}"
        current_time = datetime.now()

        # Get recent failed logins
        failed_logins = self.redis_client.lrange(key, 0, -1)
        failed_logins = [datetime.fromisoformat(ts.decode()) for ts in failed_logins]

        # Filter to last hour
        recent_failures = [ts for ts in failed_logins if current_time - ts < timedelta(hours=1)]

        # Add current failure
        recent_failures.append(current_time)

        # Store back to Redis (keep only last hour)
        self.redis_client.delete(key)
        for ts in recent_failures[-self.config['max_failed_logins']:]:
            self.redis_client.rpush(key, ts.isoformat())
        self.redis_client.expire(key, 3600)  # Expire in 1 hour

        # Check threshold
        if len(recent_failures) >= self.config['alert_thresholds']['failed_logins_per_hour']:
            self._create_security_alert(
                'multiple_failed_logins',
                f"Multiple failed login attempts from IP {ip_address}",
                'high',
                {'ip_address': ip_address, 'attempts': len(recent_failures), 'username': username}
            )

    def _monitor_unauthorized_access(self, ip_address: str):
        """Monitor unauthorized access attempts."""
        key = f"unauthorized_access:{ip_address}"
        current_time = datetime.now()

        # Similar logic to failed logins
        unauthorized_attempts = self.redis_client.lrange(key, 0, -1)
        unauthorized_attempts = [datetime.fromisoformat(ts.decode()) for ts in unauthorized_attempts]

        recent_attempts = [ts for ts in unauthorized_attempts if current_time - ts < timedelta(hours=1)]
        recent_attempts.append(current_time)

        self.redis_client.delete(key)
        for ts in recent_attempts:
            self.redis_client.rpush(key, ts.isoformat())
        self.redis_client.expire(key, 3600)

        if len(recent_attempts) >= self.config['alert_thresholds']['unauthorized_access_per_hour']:
            self._create_security_alert(
                'multiple_unauthorized_access',
                f"Multiple unauthorized access attempts from IP {ip_address}",
                'critical',
                {'ip_address': ip_address, 'attempts': len(recent_attempts)}
            )

    def _handle_suspicious_activity(self, audit_entry: Dict[str, Any]):
        """Handle suspicious activity detection."""
        self._create_security_alert(
            'suspicious_activity_detected',
            "Suspicious activity detected",
            'high',
            audit_entry
        )

    def _create_security_alert(self, alert_type: str, message: str,
                             severity: str, details: Dict[str, Any]):
        """Create a security alert."""
        alert = {
            'id': hashlib.sha256(f"{alert_type}{datetime.now().isoformat()}".encode()).hexdigest()[:16],
            'type': alert_type,
            'message': message,
            'severity': severity,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'status': 'active'
        }

        self.security_alerts.append(alert)

        # Send email alert if configured
        if self.config.get('alert_email'):
            asyncio.create_task(self._send_alert_email(alert))

        # Log alert
        logger.warning(f"Security Alert: {alert_type} - {message}")

    async def _send_alert_email(self, alert: Dict[str, Any]):
        """Send security alert via email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['smtp_username']
            msg['To'] = self.config['alert_email']
            msg['Subject'] = f"Security Alert: {alert['type']}"

            body = f"""
            Security Alert Details:
            Type: {alert['type']}
            Severity: {alert['severity']}
            Message: {alert['message']}
            Timestamp: {alert['timestamp']}
            Details: {json.dumps(alert['details'], indent=2)}
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['smtp_username'], self.config['smtp_password'])
            server.send_message(msg)
            server.quit()

        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")

    def encrypt_sensitive_data(self, data: str) -> str:
        """
        Encrypt sensitive data.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data
        """
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data.

        Args:
            encrypted_data: Data to decrypt

        Returns:
            Decrypted data
        """
        return self.fernet.decrypt(encrypted_data.encode()).decode()

    def get_audit_log(self, filters: Dict[str, Any] = None,
                     start_date: datetime = None, end_date: datetime = None,
                     limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve audit log entries.

        Args:
            filters: Filters to apply
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of entries to return

        Returns:
            Audit log entries
        """
        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        if filters:
            if 'event_type' in filters:
                query += " AND event_type = ?"
                params.append(filters['event_type'])

            if 'username' in filters:
                query += " AND username = ?"
                params.append(filters['username'])

            if 'ip_address' in filters:
                query += " AND ip_address = ?"
                params.append(filters['ip_address'])

            if 'severity' in filters:
                query += " AND severity = ?"
                params.append(filters['severity'])

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self.audit_conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]

        results = []
        for row in cursor.fetchall():
            entry = dict(zip(columns, row))
            if entry['details']:
                entry['details'] = json.loads(entry['details'])
            if entry['compliance_flags']:
                entry['compliance_flags'] = entry['compliance_flags'].split(',')
            results.append(entry)

        return results

    def generate_compliance_report(self, framework: str,
                                 start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate compliance report for a specific framework.

        Args:
            framework: Compliance framework (gdpr, sox, etc.)
            start_date: Report start date
            end_date: Report end date

        Returns:
            Compliance report
        """
        # Get relevant audit entries
        audit_entries = self.get_audit_log(
            filters={'compliance_flags': framework},
            start_date=start_date,
            end_date=end_date
        )

        # Analyze compliance
        report = {
            'framework': framework,
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'total_events': len(audit_entries),
            'events_by_type': {},
            'compliance_status': 'compliant',
            'issues': [],
            'recommendations': []
        }

        # Count events by type
        for entry in audit_entries:
            event_type = entry['event_type']
            report['events_by_type'][event_type] = report['events_by_type'].get(event_type, 0) + 1

        # Framework-specific compliance checks
        if framework == 'gdpr':
            report.update(self._check_gdpr_compliance(audit_entries))
        elif framework == 'sox':
            report.update(self._check_sox_compliance(audit_entries))

        return report

    def _check_gdpr_compliance(self, audit_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check GDPR compliance."""
        issues = []
        recommendations = []

        # Check for data access without proper logging
        data_access_events = [e for e in audit_entries if e['event_type'] == 'data_access']
        if len(data_access_events) > 0:
            # Check if all data access events have proper details
            incomplete_access = [e for e in data_access_events if not e.get('details', {}).get('purpose')]
            if incomplete_access:
                issues.append("Some data access events missing purpose documentation")
                recommendations.append("Ensure all data access events include purpose and legal basis")

        # Check for user data changes
        user_events = [e for e in audit_entries if e['event_type'] in ['user_created', 'user_deleted']]
        if len(user_events) > 0:
            recommendations.append("Review user data processing activities for GDPR compliance")

        return {
            'issues': issues,
            'recommendations': recommendations,
            'compliance_status': 'compliant' if not issues else 'review_required'
        }

    def _check_sox_compliance(self, audit_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check SOX compliance."""
        issues = []
        recommendations = []

        # Check for configuration changes
        config_changes = [e for e in audit_entries if e['event_type'] == 'config_change']
        if config_changes:
            # Check if changes are properly authorized
            unauthorized_changes = [e for e in config_changes if e.get('details', {}).get('authorized') != True]
            if unauthorized_changes:
                issues.append("Unauthorized configuration changes detected")
                recommendations.append("Implement proper authorization controls for configuration changes")

        # Check for access to financial systems
        financial_access = [e for e in audit_entries if 'financial' in str(e.get('details', {}))]
        if financial_access:
            recommendations.append("Review access controls for financial systems")

        return {
            'issues': issues,
            'recommendations': recommendations,
            'compliance_status': 'compliant' if not issues else 'review_required'
        }

    def get_security_dashboard_data(self) -> Dict[str, Any]:
        """
        Get data for security dashboard.

        Returns:
            Security dashboard data
        """
        # Get recent alerts
        recent_alerts = self.security_alerts[-10:]  # Last 10 alerts

        # Get audit statistics for last 24 hours
        yesterday = datetime.now() - timedelta(days=1)
        recent_audit = self.get_audit_log(start_date=yesterday)

        # Count events by type
        event_counts = {}
        for entry in recent_audit:
            event_type = entry['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Count events by severity
        severity_counts = {}
        for entry in recent_audit:
            severity = entry['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'recent_alerts': recent_alerts,
            'event_counts': event_counts,
            'severity_counts': severity_counts,
            'total_events_24h': len(recent_audit),
            'active_alerts': len([a for a in self.security_alerts if a['status'] == 'active'])
        }

    def cleanup_old_data(self):
        """Clean up old audit data based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self.config['data_retention_days'])

        self.audit_conn.execute(
            "DELETE FROM audit_log WHERE timestamp < ?",
            (cutoff_date.isoformat(),)
        )

        deleted_count = self.audit_conn.execute("SELECT changes()").fetchone()[0]
        self.audit_conn.commit()

        logger.info(f"Cleaned up {deleted_count} old audit entries")

    def export_audit_log(self, file_path: str, filters: Dict[str, Any] = None,
                        start_date: datetime = None, end_date: datetime = None):
        """
        Export audit log to CSV file.

        Args:
            file_path: Path to export file
            filters: Filters to apply
            start_date: Start date filter
            end_date: End date filter
        """
        audit_data = self.get_audit_log(filters, start_date, end_date, limit=100000)

        if audit_data:
            df = pd.DataFrame(audit_data)
            df.to_csv(file_path, index=False)
            logger.info(f"Exported {len(audit_data)} audit entries to {file_path}")
        else:
            logger.warning("No audit data to export")