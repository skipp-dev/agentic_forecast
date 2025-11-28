# 📋 AGENTIC_FORECAST Local Docker GPU - Quick Reference Card

## ⚠️ CRITICAL SETTING (Most Important)

**Location**: TWS/Gateway → File → Global Configuration → API → Settings

```
☑️ Enable ActiveX and Socket Clients          ← CHECK this
☐ Allow connections from localhost only       ← UNCHECK this ⚠️
```

**Why**: If unchecked, Docker can reach TWS. If checked, connection fails.

---

## 🚀 4-Step Launch

```powershell
# Step 1: Go to repo
cd C:\path\to\IB_monitoring

# Step 2: Get latest code
git pull origin main

# Step 3: Build (first time: ~5-10 min)
docker-compose build

# Step 4: Run
docker-compose up
```

---

## ✅ Success Looks Like

```
agentic-forecast-gpu  | ✅ Connection successful!
agentic-forecast-gpu  | Fetching historical data for AAPL, TSLA, NVDA...
agentic-forecast-gpu  | [*] Fetched 252 bars for AAPL
agentic-forecast-gpu  | ✅ Workflow complete!
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Container config (GPU, host network) |
| `Dockerfile` | Build recipe (Python 3.12, CUDA 12.4) |
| `AGENTIC_FORECAST/config/settings.toml` | IBKR ports & host |
| `AGENTIC_FORECAST/main.py` | Workflow entry point |

---

## 🔧 Common Issues

| Issue | Fix |
|-------|-----|
| Connection refused | Verify: `netstat -ano \| findstr :7497` |
| Still fails | UNCHECK "Allow connections from localhost only" |
| GPU not found | Run: `docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi` |
| Build fails | Run: `docker system prune && docker-compose build --no-cache` |

---

## 🏗️ Architecture

```
Windows (Your GPUs + TWS)
├─ TWS/Gateway → port 7497
└─ Docker Desktop (network_mode: host)
   └─ Container
      ├─ Python 3.12
      ├─ CUDA 12.4
      └─ ib_insync → localhost:7497 ✅
```

---

## 🎬 Stop Container

```powershell
docker-compose down
```

---

## 📚 Start Commands

```bash
# Standard mode (background/headless)
python IB_monitoring_EWMA.py --headless

# Interactive UI mode (CORRECT ✅)
python IB_monitoring_EWMA.py --curses

# Prometheus metrics export
python IB_monitoring_EWMA.py --metrics-enabled --headless

# Combined: UI + Metrics
python IB_monitoring_EWMA.py --curses --metrics-enabled

# Debug mode
python IB_monitoring_EWMA.py --headless --log-level DEBUG

# ⚠️ WRONG - These will fail silently:
# python IB_monitoring_EWMA.py --curse-mode  ❌ unrecognized argument
```

---

## Curse Mode Keyboard Shortcuts

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `?` | Help | `S` | Search |
| `F` | Filter | `Q` | Quit |
| `L` | Logs | `P` | Pause |
| `U` | Sort RSI | `T` | Sort Score |
| `A` | Sort Alerts | `R` | Refresh |
| `C` | Chart | `[/]` | Scroll |

---

## Prometheus Setup (Quick)

```bash
# 1. Start scanner with Prometheus
python IB_monitoring_EWMA.py --prometheus

# 2. Access metrics
curl http://localhost:8000/metrics

# 3. View in web UI
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

---

## Configuration

```yaml
# config.yaml
connection:
  host: "127.0.0.1"
  port: 7497

scanning:
  scan_interval: 60
  hist_concurrency: 3

gate:
  gate_threshold: 0.65

prometheus:
  enabled: true
  port: 8000
```

---

## Key Metrics (Prometheus)

```
ib_monitoring_scan_duration_seconds
ib_monitoring_alerts_generated_total
ib_monitoring_signal_score{symbol="AAPL"}
ib_monitoring_connection_status
ib_monitoring_memory_usage_bytes
ib_monitoring_buy_signals_total
ib_monitoring_sell_signals_total
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Start IB Gateway (API enabled) |
| Curse mode errors | Resize terminal to 80x24+ |
| No metrics | Check http://localhost:8000/metrics |
| Stale data | Increase hist_concurrency |
| High memory | Reduce symbol count or increase scan_interval |

---

## Ports

| Service | Port | URL |
|---------|------|-----|
| IB Gateway | 7497 | localhost:7497 |
| Prometheus Metrics | 8000 | http://localhost:8000 |
| Prometheus UI | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 |

---

## Files

| File | Purpose |
|------|---------|
| `config.yaml` | Main configuration |
| `watchlist_ibkr.csv` | Symbol list (SYMBOL,CURRENCY,EXCHANGE) |
| `email_config.ini` | Email alert settings |
| `twilio_config.ini` | Telegram/WhatsApp settings |
| `scanner.log` | Application logs |

---

## Common Commands

```bash
# View logs
tail -f scanner.log

# Check if running
ps aux | grep IB_monitoring

# Kill scanner
killall python

# View metrics
curl http://localhost:8000/metrics | head -50

# Check connections
lsof -i :7497
lsof -i :8000
```

---

## Color Guide (Curse Mode)

| Color | Meaning |
|-------|---------|
| 🟢 Green | BUY signal (RSI < 30, score > 0.65) |
| 🟡 Yellow | WATCH (interesting, not actionable) |
| 🔴 Red | EXIT signal (profit target/stop loss) |
| ⚪ White | HOLD (neutral) |

---

## Performance Tuning

```yaml
# For speed
hist_concurrency: 5
scan_interval: 30

# For reliability
hist_concurrency: 2
scan_interval: 120

# For accuracy
bar_size: "1 hour"
duration: "6 M"
```

---

**Version:** 1.0 | **Updated:** Oct 25, 2025 | **Status:** Production Ready


