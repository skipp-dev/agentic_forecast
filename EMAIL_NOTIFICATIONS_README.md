# Email-Only Notifications with Guardrail Integration

## Overview

The forecasting system now includes **email-only notifications** as the primary communication channel, providing instant inbox visibility into system health and automation safety decisions.

## Key Features

### 📧 Email-Only Focus
- **Primary Channel**: Email is now the main notification method
- **Development Mode**: Console output for development/testing
- **Production Ready**: Secure SMTP configuration with environment variables

### 🛡️ Guardrail Status Integration
- **Subject Line Tags**: `[OK]`, `[WARN]`, `[FAIL]` based on guardrail decisions
- **Header Summary**: Auto-promotion/deployment permissions and severity
- **Reasons List**: Clear explanations for guardrail decisions
- **Full Report**: Complete Markdown report attached in email body

### 📬 Email Structure

#### Subject Line
```
[Forecast Daily Report][OK] run=2025-01-28T09-00-00Z
[Forecast Daily Report][WARN] run=2025-01-28T09-00-00Z
[Forecast Daily Report][FAIL] run=2025-01-28T09-00-00Z
```

#### Email Body Header
```
Daily Forecast Health Report

Run ID: 2025-01-28T09-00-00Z
Guardrail severity: LOW
Auto-promotion allowed: YES
Auto-deployment allowed: YES

Reasons:
- All guardrail conditions satisfied - safe for automation

Full report follows below:
------------------------------------------------------------
# Daily Forecast Health Report
... (complete Markdown report)
```

## Configuration

### Environment Variables
```powershell
# Set SMTP password securely
$env:FORECAST_SMTP_PASSWORD = 'your-actual-password'
```

### config.yaml
```yaml
notifications:
  enabled: true
  send_console: false  # Production: false, Development: true
  send_email: true     # Production: true, Development: false
  email:
    smtp_host: 'smtp.gmail.com'
    smtp_port: 587
    use_tls: true
    username: 'forecast-bot@yourcompany.com'
    password_env_var: 'FORECAST_SMTP_PASSWORD'
    from_addr: 'Forecast Bot <forecast-bot@yourcompany.com>'
    to_addrs:
      - 'you@yourcompany.com'
      - 'quant-team@yourcompany.com'
      - 'ops@yourcompany.com'
    subject_prefix: '[Forecast Daily Report]'
```

## Guardrail Decision Logic

### Status Tag Derivation
- **[OK]**: `allow_auto_deploy=True` AND `severity=low`
- **[WARN]**: `allow_auto_deploy=False` OR `severity=medium`
- **[FAIL]**: `allow_auto_deploy=False` AND `severity=high`

### Decision Examples

#### ✅ OK - Healthy System
```json
{
  "allow_auto_promotion": true,
  "allow_auto_deploy": true,
  "severity": "low",
  "reasons": ["All guardrail conditions satisfied - safe for automation"]
}
```

#### ⚠️ WARN - Minor Issues
```json
{
  "allow_auto_promotion": true,
  "allow_auto_deploy": false,
  "severity": "medium",
  "reasons": ["Cross-asset V2 recommends rollback - requires manual review"]
}
```

#### ❌ FAIL - Critical Issues
```json
{
  "allow_auto_promotion": false,
  "allow_auto_deploy": false,
  "severity": "high",
  "reasons": [
    "metric_sanity_status=failed, severity=high",
    "overall_score=0.45 < min_overall_score=0.60"
  ]
}
```

## Implementation Details

### Core Components

#### NotificationAgent (`agents/notification_agent.py`)
- `notify_daily_report()`: Main notification method with guardrail integration
- `_status_tag_from_guardrails()`: Derives status tags from guardrail decisions
- `_send_email()`: Enhanced email sending with header summaries

#### Graph Integration (`src/graphs/main_graph.py`)
- `notification_node()`: Passes guardrail_decision from state to NotificationAgent
- Guardrail decisions flow from GuardrailAgent → notification_node → email

### Workflow Integration
```
Daily Health Report → Guardrail Decision → Email Notification
     ↓                     ↓                     ↓
  Unified Report     Safety Assessment     Inbox Visibility
```

## Testing & Validation

### Demo Script
```bash
python demo_email_notifications.py
```
Shows guardrail scenarios, email structure, and notification simulation.

### Setup Test
```bash
python test_email_setup.py
```
Validates SMTP configuration and tests email sending.

### Full Workflow Test
```bash
python demo_complete_workflow.py
```
Runs complete forecasting pipeline with email notifications.

## Benefits

### 🎯 Instant Triage
- **2-second inbox decisions** without opening attachments
- Status tags enable immediate priority assessment
- Header summaries show automation permissions instantly

### 🔒 Security
- SMTP passwords stored in environment variables
- No hardcoded credentials in configuration
- TLS encryption for email transmission

### 📊 Comprehensive Reporting
- Full Markdown reports in email body
- No need for separate file attachments
- Rich formatting with sections and metrics

### 🤖 Automation Ready
- Guardrail decisions drive email content
- Clear indicators for auto-promotion/deployment
- Stakeholder communication integrated into workflow

## Production Deployment

1. **Configure SMTP**: Set `FORECAST_SMTP_PASSWORD` environment variable
2. **Update Recipients**: Add real email addresses in `config.yaml`
3. **Enable Email**: Set `send_email: true` in configuration
4. **Test**: Run `python test_email_setup.py` to validate
5. **Monitor**: Check inbox for daily health reports

## Troubleshooting

### Common Issues
- **No emails received**: Check spam folder, verify SMTP credentials
- **Authentication failed**: Confirm password and email provider settings
- **TLS errors**: Ensure `use_tls: true` and correct port (587 for Gmail)
- **Missing reports**: Verify daily health report generation completes

### Debug Mode
Set `send_console: true` to see email content in terminal during development.

---

**Result**: Comfortable daily operations with instant inbox visibility into system health and automation safety! 📧✅