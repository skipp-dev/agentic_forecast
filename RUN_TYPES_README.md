# Run Type Labels in Forecasting System

## Overview

The forecasting system now includes **run type labels** that clearly identify what kind of run just happened. This provides instant context in emails, logs, and reports about whether you just ran a normal daily evaluation, expensive weekend tuning, or offline backtesting.

## Run Types

### DAILY
- **Purpose**: Normal end-of-day or once-per-day run
- **What it does**: Data refresh, feature building, training/HPO if needed, evaluation, reporting, guardrails, email
- **Auto-deployment**: Allowed if guardrails pass
- **Use case**: Standard production operation

### WEEKEND_HPO
- **Purpose**: More expensive, model-focused run (e.g., Saturday/Sunday)
- **What it does**: Extra HPO trials, maybe more models/larger search spaces, same reporting/guardrails
- **Auto-deployment**: Allowed if guardrails pass (but expect more changes)
- **Use case**: Deep tuning sessions when you have more compute time

### BACKTEST
- **Purpose**: Offline historical testing run
- **What it does**: Runs pipelines on historical segments to validate strategy/model changes
- **Auto-deployment**: **NEVER ALLOWED** (by design)
- **Use case**: Safe validation of changes before production deployment

## CLI Usage

### Basic Commands

```bash
# Standard daily run (default)
python main.py --task full --run_type DAILY

# Weekend heavy HPO run
python main.py --task full --run_type WEEKEND_HPO

# Offline backtest run (safe validation)
python main.py --task full --run_type BACKTEST
```

### Default Behavior
- If `--run_type` is not specified, defaults to `DAILY`
- All run types use the same workflow but have different guardrail behaviors

## Data Flow

The run type flows through the entire system:

```
CLI Argument → GraphState → Daily Report JSON → Guardrails → Email Subject/Header
     ↓             ↓              ↓                ↓              ↓
--run_type    state.run_type  run_metadata    BACKTEST blocks   [DAILY][OK]
```

### 1. CLI → GraphState
```python
# main.py
args = parse_args()
initial_state = build_initial_state(symbols, config, args.run_type)
# GraphState.run_type = "DAILY" | "WEEKEND_HPO" | "BACKTEST"
```

### 2. GraphState → Daily Report JSON
```python
# daily_health_report.py
run_metadata = {
    "run_id": run_id,
    "run_type": state.get("run_type", "DAILY"),  # ← Here
    "evaluated_at": now_iso(),
    # ...
}
```

### 3. Daily Report → Guardrails
```python
# guardrail_agent.py
run_type = daily_report.get("run_metadata", {}).get("run_type", "DAILY")
if run_type == "BACKTEST":
    # Block auto-promotion/deployment
    allow_auto_promotion = False
    allow_auto_deploy = False
```

### 4. Daily Report → Email Notifications
```python
# notification_agent.py
run_type = run_metadata.get("run_type", "DAILY").upper()
subject = f"{subject_prefix}[{run_type}]{status_tag} run={run_id}"
# Email header includes run_type
```

## Email Integration

### Subject Lines
```
[Forecast Daily Report][DAILY][OK] run=2025-01-28T09-00-00Z
[Forecast Daily Report][WEEKEND_HPO][WARN] run=2025-01-28T09-00-00Z
[Forecast Daily Report][BACKTEST][FAIL] run=2025-01-28T09-00-00Z
```

### Email Headers

#### DAILY Run (Normal Operation)
```
Daily Forecast Health Report

Run ID: 2025-01-28T09-00-00Z
Run type: DAILY
Guardrail severity: LOW
Auto-promotion allowed: YES
Auto-deployment allowed: YES

Reasons:
- All guardrail conditions satisfied - safe for automation
```

#### WEEKEND_HPO Run (Heavy Tuning)
```
Daily Forecast Health Report

Run ID: 2025-01-28T09-00-00Z
Run type: WEEKEND_HPO
Guardrail severity: MEDIUM
Auto-promotion allowed: YES
Auto-deployment allowed: NO

Reasons:
- Cross-asset V2 recommends rollback - requires manual review
```

#### BACKTEST Run (Safe Validation)
```
Daily Forecast Health Report

Run ID: 2025-01-28T09-00-00Z
Run type: BACKTEST
Guardrail severity: MEDIUM
Auto-promotion allowed: NO
Auto-deployment allowed: NO

Reasons:
- run_type=BACKTEST: auto-promotion/deploy disabled by design.
- All guardrail conditions satisfied - safe for automation
```

## Guardrail Behavior

### BACKTEST Special Rules
BACKTEST runs have enhanced safety measures:

- **Auto-promotion**: Always `False` (prevents model changes)
- **Auto-deployment**: Always `False` (prevents production changes)
- **Severity**: At least `MEDIUM` (flags as requiring attention)
- **Reason**: Explicitly states "auto-promotion/deploy disabled by design"

This ensures backtests can never accidentally alter production state, even if all metrics look perfect.

### Implementation
```python
# In GuardrailAgent.evaluate_daily_health()
run_type = daily_report.get("run_metadata", {}).get("run_type", "DAILY")

if run_type == "BACKTEST":
    reasons.append("run_type=BACKTEST: auto-promotion/deploy disabled by design.")
    violations.append("backtest_mode")
    allow_auto_promotion = False
    allow_auto_deploy = False
    if severity == "low":
        severity = "medium"
```

## Daily Report JSON Structure

The `run_metadata` section now includes `run_type`:

```json
{
  "run_metadata": {
    "run_id": "2025-01-28T09-00-00Z",
    "run_type": "DAILY",
    "evaluated_at": "2025-01-28T09-02-15.123456Z",
    "symbols": ["AAPL", "MSFT", "NVDA", "GOOGL"],
    "horizons": ["1", "5", "10", "20"],
    "config": {
      "feature_store_v2_enabled": true,
      "models_trained": ["AutoNHITS", "AutoNBEATS", "AutoDLinear", "AutoTFT"],
      "hpo_trials_per_symbol": 30
    }
  }
}
```

## Testing

### Demo Script
```bash
python demo_run_types.py
```
Shows all run types, email subjects, and guardrail behaviors.

### Manual Testing
```bash
# Test each run type
python main.py --task full --run_type DAILY
python main.py --task full --run_type WEEKEND_HPO
python main.py --task full --run_type BACKTEST

# Check the run_type in the daily report
cat results/reports/daily_forecast_health_latest.json | jq ".run_metadata.run_type"

# Check email subjects in logs
grep -i "subject" logs/*.log
```

### Validation Points
- [ ] CLI accepts `--run_type` with proper choices
- [ ] GraphState includes `run_type` field
- [ ] Daily report JSON contains `run_metadata.run_type`
- [ ] Email subjects include `[RUN_TYPE]` tag
- [ ] Email headers show "Run type: X"
- [ ] BACKTEST runs block auto-promotion/deployment
- [ ] BACKTEST runs show appropriate reasons

## Benefits

### 📧 **Instant Email Triage**
- **What run was this?** `[DAILY]`, `[WEEKEND_HPO]`, `[BACKTEST]` in subject
- **Is it healthy?** `[OK]`, `[WARN]`, `[FAIL]` status
- **Can it act automatically?** Auto-promotion/deployment flags

### 🛡️ **Safety by Design**
- BACKTEST runs can never accidentally deploy changes
- Clear labeling prevents confusion between validation and production runs
- Guardrails respect run context

### 📊 **Better Logging/Analytics**
- All reports include run_type for filtering and analysis
- Historical tracking of what types of runs were performed
- Performance comparison across run types

### 🤖 **Operational Clarity**
- Stakeholders instantly know what kind of evaluation happened
- Different expectations for different run types
- Appropriate alerting based on run context

## Future Extensions

### Additional Run Types
```python
# Could add later:
parser.add_argument("--run_type", choices=[
    "DAILY", "WEEKEND_HPO", "BACKTEST",
    "INTRADAY", "ADHOC_DEBUG", "RESEARCH_ONLY"
])
```

### Run-Type-Specific Config
```yaml
# config.yaml
run_types:
  DAILY:
    hpo_trials_per_symbol: 20
    enable_expensive_features: false
  WEEKEND_HPO:
    hpo_trials_per_symbol: 50
    enable_expensive_features: true
  BACKTEST:
    hpo_trials_per_symbol: 10
    enable_expensive_features: false
    disable_all_auto_actions: true
```

---

**Result**: Every run now tells you WHAT it was, not just whether it passed! 🎯🏷️</content>
<parameter name="filePath">c:\Users\spreu\Documents\agentic_forecast\RUN_TYPES_README.md