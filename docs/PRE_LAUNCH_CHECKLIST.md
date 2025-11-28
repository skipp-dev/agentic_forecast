# Pre-Launch Checklist ✅

Before running `docker-compose up` on your Windows machine, verify:

## Windows Machine Setup

- [ ] **Docker Desktop installed** and running
  - Verify: `docker --version` in PowerShell
  
- [ ] **NVIDIA GPU support enabled** in Docker Desktop
  - Settings → Resources → GPU
  - Verify: `docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi`
  
- [ ] **Repository cloned locally** on Windows
  - Location: `C:\path\to\IB_monitoring`
  - Verify: `git status` shows no errors

## Interactive Brokers Setup

- [ ] **TWS or IB Gateway is running** on Windows
  - Check Windows taskbar or Applications
  - Verify: `netstat -ano | findstr :7497` shows LISTENING
  
- [ ] **API is enabled in TWS/Gateway settings**
  - Open TWS/Gateway
  - Go to: **File → Global Configuration → API → Settings**
  - ✅ Check: "Enable ActiveX and Socket Clients"
  - ✅ **UNCHECK**: "Allow connections from localhost only" ⚠️ CRITICAL
  - ✅ Verify Port: 7497 (for Paper) or 7496 (for Live)
  - Click: **Apply** and **OK**
  - Restart TWS/Gateway if it prompted for restart

## Docker Configuration

- [ ] **docker-compose.yml exists** in repo root
  - Verify: `ls docker-compose.yml`
  
- [ ] **Dockerfile exists** in repo root
  - Verify: `ls Dockerfile`
  
- [ ] **AGENTIC_FORECAST/config/settings.toml configured**
  - Verify host: `localhost`
  - Verify ports: `[7497, 7496, 4002, 4001]`

## Final Pre-Flight

- [ ] Close any other applications using GPU (optional, for performance)
- [ ] Ensure internet connection (for Docker hub pulls on first build)
- [ ] Open PowerShell in the repo directory: `cd C:\path\to\IB_monitoring`

---

## Launch Commands

Once all checks pass, in PowerShell:

```powershell
# Step 1: Build the Docker image (first time only, ~5-10 minutes)
docker-compose build

# Step 2: Run the container
docker-compose up
```

## Expected Output

First ~30 seconds should show:
```
agentic-forecast-gpu  | [*] Attempting to connect to IBKR on localhost:7497...
agentic-forecast-gpu  | ✅ Connection successful!
agentic-forecast-gpu  | Fetching historical data for AAPL, TSLA, NVDA...
agentic-forecast-gpu  | [*] Fetched 252 bars for AAPL
agentic-forecast-gpu  | [*] Fetched 252 bars for TSLA
agentic-forecast-gpu  | [*] Fetched 252 bars for NVDA
agentic-forecast-gpu  | [*] Processing features...
agentic-forecast-gpu  | [*] Generating forecasts...
agentic-forecast-gpu  | ✅ Workflow complete!
```

If you see connection errors:
1. **Verify TWS is actually running** (check taskbar)
2. **Verify "Allow connections from localhost only" is UNCHECKED** (most common issue)
3. **Check TWS port** matches docker-compose.yml (should be 7497)
4. **Restart TWS after making changes**

---

## Quick Troubleshooting

| Symptom | Fix |
|---------|-----|
| "Connection refused" | Verify TWS port 7497 is listening: `netstat -ano \| findstr :7497` |
| "Cannot connect to Docker daemon" | Start Docker Desktop |
| "GPU not found" | Verify Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi` |
| "Build fails" | Clear cache: `docker system prune` then `docker-compose build --no-cache` |

---

## What's Running After Launch

✅ **Container**: `agentic-forecast-gpu`  
✅ **Python**: 3.12 with CUDA 12.4  
✅ **Connection**: Direct to TWS on localhost:7497  
✅ **Workflow**: Fetching AAPL, TSLA, NVDA bars → generating forecasts  
✅ **GPU**: Accelerating forecast models (PyTorch/neuralforecast)  

---

## Stop the Container

When done, in PowerShell:
```powershell
docker-compose down
```

---

**Ready?** ✅ Follow checklist above → Run `docker-compose up` → See live IBKR data flowing!

