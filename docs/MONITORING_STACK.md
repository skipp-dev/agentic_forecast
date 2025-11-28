# Monitoring Stack – FastAPI + Prometheus + Grafana

This document explains how the monitoring stack for the Agentic Forecast system is set up and how the pieces work together.

The stack consists of:

- **FastAPI service** – exposes `/health` and `/metrics`
- **Prometheus** – scrapes `/metrics` and stores time-series
- **Grafana** – visualizes metrics from Prometheus
- **PowerShell scripts** – `start_stack.ps1`, `stop_stack.ps1`, `check_stack.ps1` control everything

---

## 1. FastAPI Service

Entry point: `run_api.py`

Responsibilities:

- **Health endpoint**: `GET /health`
  - Returns JSON, e.g. `{"status": "ok"}`
- **Metrics endpoint**: `GET /metrics`
  - Returns Prometheus-formatted metrics using `prometheus_client`

Example sketch:

```python
from fastapi import FastAPI, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

FastAPI runs locally on:

* `http://127.0.0.1:8000/health`
* `http://127.0.0.1:8000/metrics`

It is started by:

```powershell
python run_api.py
```

or automatically by `start_stack.ps1`.

---

## 2. Prometheus

### 2.1. Installation

* Download Prometheus for Windows
* Extract to e.g.:

```text
C:\prometheus
```

Executable path:

```text
C:\prometheus\prometheus.exe
```

### 2.2. Configuration

Project-specific config file:

```text
C:\Users\spreu\Documents\agentic_forecast\monitoring\prometheus.yml
```

Example content:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "agentic_forecast_api"
    scrape_interval: 15s
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

This tells Prometheus to scrape:

* `http://127.0.0.1:8000/metrics`

### 2.3. Running Prometheus

Manually:

```powershell
cd C:\prometheus
.\prometheus.exe --config.file="C:\Users\spreu\Documents\agentic_forecast\monitoring\prometheus.yml"
```

Or via `start_stack.ps1`, which does the same in a separate minimized window.

Prometheus UI:

* `http://localhost:9090`
* Check **Status → Targets** to see if `agentic_forecast_api` is UP.

---

## 3. Grafana

### 3.1. Installation

* Install Grafana for Windows (standard installer)
* It usually runs as a Windows service and listens on:

```text
http://localhost:3000
```

Default login:

* user: `admin`
* password: `admin` (forced reset on first login)

### 3.2. Add Prometheus as Data Source

In Grafana:

1. Go to **Connections → Data sources → Add data source**
2. Choose **Prometheus**
3. Set:

   * **URL**: `http://localhost:9090`
4. Click **Save & test**

If Prometheus is running, Grafana should say "Data source is working".

### 3.3. Import Dashboards

If the project contains dashboard JSONs (e.g. `grafana_dashboards/model_performance_overview.json`):

1. Left menu → **Dashboards → New → Import**
2. Click **Upload JSON file** and choose the JSON
3. Assign the Prometheus data source
4. Click **Import**

The dashboard will now show:

* Error metrics (MAE, RMSE, MAPE, SMAPE, SWASE)
* Directional accuracy / drift
* HPO performance summaries
* Guardrail decisions, etc.

---

## 4. PowerShell Scripts

### 4.1. `start_stack.ps1`

Starts the full stack:

* **Prometheus**
* **FastAPI**
* **Forecast engine**

Usage:

```powershell
cd C:\Users\spreu\Documents\agentic_forecast

# Default daily run
.\start_stack.ps1

# Heavy tuning weekend run
.\start_stack.ps1 -RunType WEEKEND_HPO

# Offline backtest run
.\start_stack.ps1 -RunType BACKTEST
```

Internally it:

* launches Prometheus with `monitoring\prometheus.yml`
* launches `python run_api.py`
* runs `python main.py --task full --run_type <RunType>` in the current window

### 4.2. `stop_stack.ps1`

Stops:

* Prometheus (`prometheus.exe`)
* FastAPI (`python run_api.py`)
* optionally: forecast runs (`main.py --task full ...`) if configured

Usage:

```powershell
cd C:\Users\spreu\Documents\agentic_forecast
.\stop_stack.ps1
```

### 4.3. `check_stack.ps1`

Health probe for the stack:

* FastAPI: `http://127.0.0.1:8000/health`
* Prometheus: `http://localhost:9090/-/ready`
* Grafana: `http://localhost:3000/`

Usage:

```powershell
cd C:\Users\spreu\Documents\agentic_forecast
.\check_stack.ps1
```

Exit codes:

* `0` → all components healthy
* `1` → one or more components failing

---

## 5. Typical Daily Flow

1. Start stack:

   ```powershell
   cd C:\Users\spreu\Documents\agentic_forecast
   .\start_stack.ps1           # or -RunType WEEKEND_HPO / BACKTEST
   ```

2. Open Grafana:

   ```text
   http://localhost:3000
   ```

3. Watch dashboards update as the forecast engine runs (Prometheus scrapes FastAPI).

4. Check health if needed:

   ```powershell
   .\check_stack.ps1
   ```

5. Stop infra when done:

   ```powershell
   .\stop_stack.ps1
   ```

This makes Grafana + API your primary interface, while Prometheus quietly handles metrics under the hood.