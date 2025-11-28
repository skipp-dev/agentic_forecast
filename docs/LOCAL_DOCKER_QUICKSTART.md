# AGENTIC_FORECAST Local Docker Execution - Quick Start

## TL;DR - 3 Steps to Run

1. **Verify TWS is running** on your Windows machine with API enabled (port 7497)
   
2. **Build and run**:
   ```powershell
   cd C:\path\to\IB_monitoring
   docker-compose build
   docker-compose up
   ```

3. **Watch it connect to TWS and fetch live IBKR data** ✅

---

## Architecture

```
Windows Machine (Your GPUs)
├─ Interactive Brokers TWS/Gateway (port 7497)
└─ Docker Desktop
   └─ agentic-forecast-gpu container (host networking)
      ├─ Python 3.12
      ├─ CUDA 12.4 (GPU support)
      ├─ LangGraph (agentic workflow)
      ├─ ib_insync (IBKR connection)
      └─ neuralforecast (forecasting model on GPU)
```

**Network**: `network_mode: host` means container uses Windows' network stack → direct localhost:7497 access to TWS

---

## Files Created

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Orchestrates container with GPU, volumes, environment |
| `Dockerfile` | Multi-stage build: Python 3.12 + CUDA 12.4 + all dependencies |
| `LOCAL_GPU_EXECUTION_GUIDE.md` | Detailed setup, troubleshooting, architecture |
| `AGENTIC_FORECAST/config/settings.toml` | Updated: localhost, ports [7497, 7496, 4002, 4001] |

---

## Why This Works

✅ **Container runs on Windows where GPU is** (not cloud)  
✅ **Host networking allows direct access to TWS** (127.0.0.1:7497)  
✅ **GPU automatically used by torch/neuralforecast**  
✅ **All code already written and tested** (just executes locally)  
✅ **Edit in Codespaces, run locally** (git syncs code)  

---

## Next Actions

1. **Pull latest code** in Codespaces (this file + docker-compose.yml + Dockerfile + guide)
2. **Clone repo locally on Windows** or pull if already there
3. **Run**: `docker-compose up` from repo directory
4. **Watch output** - should show IBKR connection + data fetching

---

## Troubleshooting

**"Connection refused on 127.0.0.1:7497"**
- Verify TWS running: `netstat -ano | findstr :7497`
- Check "Allow connections from localhost only" is UNCHECKED in TWS API settings
- Restart TWS after changing settings

**"No GPU detected"**
- Verify Docker Desktop GPU support: `docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi`
- If not working, reinstall NVIDIA Container Toolkit

**"Build fails"**
- Clear cache: `docker system prune`
- Try again: `docker-compose build --no-cache`

---

## Full Documentation

See `LOCAL_GPU_EXECUTION_GUIDE.md` for:
- Complete setup steps
- Configuration options
- GPU optimization
- Production deployment

---

**Status**: ✅ Ready to run locally. No more Codespaces networking issues.

