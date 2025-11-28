# Agentic Forecast – Run Types & Stack Startup (API + Prometheus + Grafana)

This document explains:

- The different **run types** (`DAILY`, `WEEKEND_HPO`, `BACKTEST`)
- How to start the **full stack**:
  - Forecast engine (agentic pipeline)
  - FastAPI service (`run_api.py`)
  - Prometheus metrics collector
  - Grafana dashboards

The goal: in daily use you mostly interact via **Grafana + API**, not via raw JSON.

---

## 0. Folder & Environment Assumptions

Project root:

```text
C:\Users\spreu\Documents\agentic_forecast
```

You run commands in **PowerShell**:

```powershell
cd C:\Users\spreu\Documents\agentic_forecast
```

Python dependencies installed once:

```powershell
pip install -r requirements.txt
```

Prometheus config (example):

```text
monitoring\prometheus.yml
```

FastAPI server:

* defined in `run_api.py`
* exposes at: `http://127.0.0.1:8000`
* metrics at: `http://127.0.0.1:8000/metrics` (Prometheus format)

Grafana:

* Installed on Windows (standard installer)
* Runs at: `http://localhost:3000`
* Will read from Prometheus as data source

Email notifications are **optional** and can be enabled later.

---

## 1. Run Types

The forecasting engine is always started via:

```powershell
python main.py --task full --run_type <TYPE>
```

### 1.1 DAILY

**Purpose**

* Your normal, day-to-day run:

  * load data
  * build features (incl. cross-asset v1/v2 if enabled)
  * run models / HPO with *normal* budgets
  * evaluate metrics
  * generate reports & guardrail decisions

**Command**

```powershell
python main.py --task full --run_type DAILY
```

**What it produces**

* `results/reports/daily_forecast_health_latest.json`
* `results/reports/daily_forecast_health_latest.md`
* updated metrics for Prometheus → Grafana
* `GuardrailAgent` decision:

  * `allow_auto_promotion`
  * `allow_auto_deploy`

---

### 1.2 WEEKEND_HPO

**Purpose**

* Heavier tuning run (e.g. Saturday/Sunday):

  * more HPO trials per model
  * optional extra model families (TFT, PatchTST, iTransformer, …)
  * longer and more expensive, but still fully guarded

**Command**

```powershell
python main.py --task full --run_type WEEKEND_HPO
```

**What’s different from DAILY**

* Same pipeline, but:

  * higher HPO budgets
  * more families enabled
* Guardrails still apply:

  * bad metric sanity / low scores → no auto-promotion / deploy

---

### 1.3 BACKTEST

**Purpose**

* Offline / historical evaluation:

  * validate new features, models, or strategy logic
  * **never** touch production model state

**Command**

```powershell
python main.py --task full --run_type BACKTEST
```

**Special rules**

* Runs the same evaluation & reporting stack.
* `GuardrailAgent` is hard-coded to:

  * `allow_auto_promotion = False`
  * `allow_auto_deploy = False`

Even perfect metrics **cannot** auto-change live models from a BACKTEST run.

---

## 2. System Components

Your "stack" has four main pieces:

1. **Forecast Engine (agentic pipeline)**

   * `main.py --task full --run_type ...`
   * Does all data/features/models/HPO/reporting/guardrails
2. **FastAPI Service**

   * `run_api.py`
   * Health check (+ optionally `/metrics` for Prometheus)
3. **Prometheus**

   * scrapes `http://127.0.0.1:8000/metrics`
   * stores time-series of metrics (latency, error counts, model stats)
4. **Grafana**

   * reads from Prometheus
   * visualizes dashboards like:

     * Model Performance Overview
     * Drift & Metric Sanity
     * Cross-asset V2 impact
     * System health

---

## 3. Full Stack Startup – Daily Workflow

### 3.1 One-time installs

**Already done / assumed for you:**

* `pip install -r requirements.txt` inside `agentic_forecast`
* Grafana installed (Windows service / desktop app)
* Prometheus downloaded (e.g. `C:\prometheus\prometheus.exe`)
* `monitoring\prometheus.yml` configured with a job like:

  ```yaml
  scrape_configs:
    - job_name: 'agentic_forecast'
      scrape_interval: 15s
      static_configs:
        - targets: ['127.0.0.1:8000']  # FastAPI /metrics
  ```

### 3.2 Start Grafana (Windows)

Grafana usually runs as a service; typical options:

* Use the **"Grafana"** entry in the Start menu, or
* Ensure the **Grafana Windows service** is running

Then open:

```text
http://localhost:3000
```

Login:

* user: `admin`
* pass: `admin` (or whatever you changed it to)

You only need to do this once per PC boot.

---

### 3.3 Start Prometheus (Windows)

In a **separate PowerShell** window:

```powershell
cd C:\prometheus   # or wherever you unpacked it
.\prometheus.exe --config.file="C:\Users\spreu\Documents\agentic_forecast\monitoring\prometheus.yml"
```

* Prometheus UI: `http://localhost:9090`
* It will start scraping `http://127.0.0.1:8000/metrics` as soon as the API is up.

You can leave Prometheus running in the background.

---

### 3.4 Start FastAPI (Health + /metrics)

In another PowerShell window:

```powershell
cd C:\Users\spreu\Documents\agentic_forecast
python run_api.py
```

Assumptions:

* `run_api.py` starts a FastAPI app, e.g. on port 8000:

  * Health: `http://127.0.0.1:8000/health`
  * Metrics: `http://127.0.0.1:8000/metrics` (Prometheus)

You can check:

```powershell
python -c "import requests; print(requests.get('http://127.0.0.1:8000/health').json())"
```

---

### 3.5 Run the Forecast Engine

In a third PowerShell window:

```powershell
cd C:\Users\spreu\Documents\agentic_forecast

# Most common:
python main.py --task full --run_type DAILY
```

For heavy tuning:

```powershell
python main.py --task full --run_type WEEKEND_HPO
```

For safe offline experiments:

```powershell
python main.py --task full --run_type BACKTEST
```

What this run does:

1. Builds / updates base & cross-asset features (v1/v2)
2. Runs model training / HPO as configured
3. Evaluates performance (MAE, RMSE, MAPE, SMAPE, SWASE, DA, etc.)
4. Runs:

   * Metrics sanity checks
   * Cross-asset V2 analysis (A/B)
   * GuardrailAgent (decide auto-promotion/deploy)
5. Writes:

   * `results/reports/daily_forecast_health_latest.json`
   * `results/reports/daily_forecast_health_latest.md`
   * updated metrics files for the metrics exporter → FastAPI `/metrics`
6. If notifications are enabled later:

   * NotificationAgent sends the Daily Forecast Health Report via email

Prometheus will now see new metrics values and Grafana dashboards will update automatically.

---

## 4. Grafana as Primary Interface

Once the stack runs:

* **Open** Grafana: `http://localhost:3000`
* **Configure data source** (only once):

  * Type: Prometheus
  * URL: `http://localhost:9090`
* **Import dashboard JSON**:

  * Use the "Import dashboard" function
  * Paste/upload the dashboard JSON you generated (e.g. `grafana_dashboards/model_performance_overview.json`)
* After import, you'll see panels like:

  * Model error metrics (MAE, RMSE, MAPE, SMAPE, SWASE)
  * Directional accuracy / drift events
  * HPO performance summaries
  * Cross-asset V2 on/off comparisons
  * Guardrail overall score & allow_auto_deploy flags

For daily work, your workflow becomes:

1. Start **Grafana** (usually auto as a service)
2. Start **Prometheus**
3. Start **FastAPI** (`python run_api.py`)
4. Run the **forecast engine** (`python main.py --task full --run_type DAILY`)
5. **Watch Grafana** dashboards for:

   * today's health
   * trend over recent days
   * V2 feature impact
   * drift / metric sanity

You rarely need to open raw JSON/MD files anymore.

---

## 5. Optional: Email Notifications (Later)

When you're ready, enable email in `config.yaml`:

```yaml
notifications:
  enabled: true
  send_email: true

  email:
    smtp_host: "smtp.your-provider.com"
    smtp_port: 587
    use_tls: true
    username: "forecast-bot@yourdomain.com"
    password_env_var: "FORECAST_SMTP_PASSWORD"
    from_addr: "Forecast Bot <forecast-bot@yourdomain.com>"
    to_addrs:
      - "you@yourcompany.com"
    subject_prefix: "[Forecast]"
```

Set the password in PowerShell:

```powershell
$env:FORECAST_SMTP_PASSWORD = "your-app-password"
```

After that, each `main.py --task full --run_type ...` run will also send:

* An email with:

  * `run_id`
  * `run_type` (DAILY / WEEKEND_HPO / BACKTEST)
  * Guardrail severity (OK/WARN/FAIL)
  * Allow auto-promotion/deploy flags
  * Full Markdown report in the body

But this is **optional** – API + Prometheus + Grafana stay your primary interface.

---

## 6. Quick Reference Cheat Sheet

### Start whole stack (3 terminals)

```powershell
# Terminal 1 – Prometheus
cd C:\prometheus
.\prometheus.exe --config.file="C:\Users\spreu\Documents\agentic_forecast\monitoring\prometheus.yml"

# Terminal 2 – FastAPI
cd C:\Users\spreu\Documents\agentic_forecast
python run_api.py

# Terminal 3 – Forecast engine (daily)
cd C:\Users\spreu\Documents\agentic_forecast
python main.py --task full --run_type DAILY
```

Grafana runs as a Windows service / app and is accessible at:

```text
http://localhost:3000
```

You can bookmark your main dashboard and basically live there.
Everything else is just for debugging or power-user tweaks.