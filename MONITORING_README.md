# Agentic Forecast Monitoring Stack

Complete monitoring setup with Prometheus and Grafana for the agentic forecasting system.

## 🏗️ Architecture

```
Forecast API (localhost:8002) → Prometheus (localhost:9090) → Grafana (localhost:3000)
       ↑                           ↑                           ↑
   /metrics endpoint        scrapes every 15s          visualizes data
```

## 🚀 Quick Start

### 1. Start the Monitoring Stack

```bash
# Start Prometheus and Grafana
docker compose up -d prometheus grafana

# Or use the management script
python monitoring_manager.py start
```

### 2. Start the API

```bash
# Start the forecast API
python start_api.py
```

### 3. Access the Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **API Health**: http://localhost:8002/health
- **API Metrics**: http://localhost:8002/metrics

## 📊 Dashboard Features

### Metrics Health Overview Dashboard
The "Metrics Health Overview" dashboard includes:

### 🔴 Overall Metrics Health Status
- Shows PASSED/FAILED based on `metrics_health_status`
- Green = Healthy, Red = Issues detected

### 🟡 Severity Level
- Displays current severity levels (low/medium/high)
- Color-coded background based on severity

### 🚫 Guardrail Status
- **Can Rotate Models**: BLOCKED/ALLOWED
- **Can Update Champions**: BLOCKED/ALLOWED

### 📋 Issue Counts by Type
- Table showing specific issues found by the QualityAgent
- Examples: `mape_high`, `identical_metrics_per_horizon`
- Sorted by count (highest first)

### ⏰ Run Age – Latest Evaluation
- Shows how long ago the last quality evaluation ran
- Green < 1hr, Orange < 1 day, Red > 1 day

### 📈 Metrics Health Status Over Time
- Timeseries chart showing health status changes
- Helps track system stability over time

### Model Performance Overview Dashboard
The "Model Performance Overview" dashboard provides comprehensive model performance monitoring:

### 📊 Global Performance Statistics
- **Average MAE**: Mean Absolute Error across all symbols and horizons
- **Average MAPE**: Mean Absolute Percentage Error (as percentage)
- **Average SMAPE**: Symmetric Mean Absolute Percentage Error (as percentage)
- **Average SWASE**: Symmetric Weighted Absolute Percentage Error
- **Average Directional Accuracy**: Direction prediction accuracy (as percentage)

### 📋 Detailed Performance Table
- Per-symbol and per-horizon breakdown of all metrics
- Columns: MAE, MAPE, SMAPE, SWASE, Directional Accuracy
- Sortable by any metric value

### 📈 Horizon-based Timeseries Charts
- **MAE by Horizon**: Average MAE across symbols for each forecast horizon
- **MAPE by Horizon**: Average MAPE across symbols for each forecast horizon
- **SMAPE by Horizon**: Average SMAPE across symbols for each forecast horizon
- **SWASE by Horizon**: Average SWASE across symbols for each forecast horizon
- **Directional Accuracy by Horizon**: Average DA across symbols for each forecast horizon

### 🔍 Symbol Drilldown
- Interactive symbol selection dropdown
- Detailed timeseries for selected symbol across all horizons
- All metrics (MAE/MAPE/SMAPE/SWASE/DA) displayed together for comparison

## 🛠️ Management Commands

Use the management script for easy control:

```bash
# Check status of all services
python monitoring_manager.py status

# Restart the monitoring stack
python monitoring_manager.py restart

# Stop monitoring
python monitoring_manager.py stop

# View recent logs
python monitoring_manager.py logs
```

## 🔧 Configuration Files

### Prometheus (`monitoring/prometheus.yml`)
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "agentic_forecast"
    metrics_path: /metrics
    static_configs:
      - targets:
          - "host.docker.internal:8002"  # API endpoint
```

### Grafana Provisioning
- **Datasource**: `monitoring/grafana/provisioning/datasources/prometheus.yml`
- **Dashboard**: `monitoring/grafana/provisioning/dashboards/dashboard.yml`
- **Dashboard JSON**: `monitoring/grafana/dashboards/metrics-health.json`

## 🔧 SMAPE & SWASE Implementation

### SMAPE (Symmetric Mean Absolute Percentage Error)
**Formula**: `SMAPE = (1/N) * Σ(2 * |ŷ_t - y_t| / (|ŷ_t| + |y_t| + ε))`

- **Range**: 0 to 2 (typically reported as percentage: 0-200%)
- **Advantage**: More robust than MAPE when actual values are near zero
- **Interpretation**: Values around 0.05 = ~5% average symmetric error

### SWASE (Shock-Weighted Absolute Scaled Error)
**Formula**: `SWASE = Σ(w_t * ASE_t) / Σ(w_t)` where `ASE_t = |ŷ_t - y_t| / (|y_t| + ε)`

- **Weights**: Normal days (w=1), Shock days (w=3)
- **Shock detection**: Based on `peer_shock_flag` or `has_macro_event_today`
- **Purpose**: Penalizes forecast errors on important/high-risk days more heavily
- **Interpretation**: Higher values indicate worse performance, especially on critical days

### Implementation Files
- `analytics/evaluation_metrics.py` - Core evaluation functions
- `services/metrics_exporter.py` - Prometheus metric export
- `fix_evaluation_bug.py` - Updated to include SWASE in CSV exports

### Testing
Run `python test_smape_swase.py` to test the evaluation functions.

## 🚨 Troubleshooting

### Prometheus shows "DOWN" status
1. Check if API is running: `python monitoring_manager.py status`
2. Verify metrics endpoint: `curl http://localhost:8002/metrics`
3. Check Prometheus targets: http://localhost:9090/targets

### Grafana can't connect to Prometheus
1. Ensure Prometheus is running: `docker ps | grep prometheus`
2. Check datasource configuration in Grafana
3. Verify network connectivity between containers

### Dashboard shows "No data"
1. Confirm API is generating metrics
2. Check that evaluation has run recently
3. Verify metric names match dashboard queries

### API server crashes on requests
- Use the programmatic runner: `python start_api.py`
- Avoid running with `uvicorn` directly in PowerShell

## 🔄 Updating the Dashboard

To modify the dashboard:

1. Edit `monitoring/grafana/dashboards/metrics-health.json`
2. Restart Grafana: `docker compose restart grafana`
3. Or reload provisioning: `docker exec grafana kill -HUP 1`

## 📁 File Structure

```
monitoring/
├── prometheus.yml                    # Prometheus configuration
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/
│   │   │   └── prometheus.yml        # Grafana datasource config
│   │   └── dashboards/
│   │       └── dashboard.yml         # Dashboard provisioning
│   └── dashboards/
│       ├── metrics-health.json       # Health monitoring dashboard
│       └── model-performance.json    # Model performance dashboard
├── prometheus_data/                  # Prometheus data (created on first run)
└── grafana_data/                     # Grafana data (created on first run)
```

## 🎯 Production Deployment

For production:

1. **Secure Grafana**: Change default admin password
2. **Configure Prometheus**: Add authentication, TLS
3. **Persistent Storage**: Mount volumes for data persistence
4. **Network Security**: Use internal networks, not host.docker.internal
5. **Monitoring**: Add alerts for critical metrics

## 📞 Support

If issues persist:
1. Check service logs: `python monitoring_manager.py logs`
2. Verify all services are running: `python monitoring_manager.py status`
3. Test API endpoints manually with curl/Postman