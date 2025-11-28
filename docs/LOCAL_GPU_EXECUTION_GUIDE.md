# AGENTIC_FORECAST Local GPU Execution Guide

## Overview
Run the AGENTIC_FORECAST system locally on your Windows machine with GPU support, connecting directly to Interactive Brokers TWS/Gateway.

## Prerequisites

✅ **Required:**
- Docker Desktop installed on Windows
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- Interactive Brokers TWS or Gateway running locally on port 7497
- Git (to pull latest code)
- Python 3.12 (optional, for local testing)

## Setup Steps

### 1. Verify Docker GPU Support

```powershell
# Test GPU is available to Docker
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

Expected output: Shows your GPU(s) details.

### 2. Clone/Update Repository

```powershell
# If not already cloned
git clone https://github.com/YOUR_USERNAME/IB_monitoring.git
cd IB_monitoring

# Or if already cloned, update
git pull origin main
```

### 3. Verify IB Gateway/TWS Configuration

In your TWS/Gateway application:
- **File → Global Configuration → API → Settings**
- ✅ Check "Enable ActiveX and Socket Clients"
- ✅ Uncheck "Allow connections from localhost only" (important for Docker)
- Verify Socket Port is 7497 (or update `docker-compose.yml` if different)
- Apply and restart if needed

### 4. Build Docker Image

```powershell
cd C:\path\to\IB_monitoring
docker-compose build
```

First build takes ~5-10 minutes (downloads base image, installs packages).

### 5. Run the Workflow

```powershell
# Start the AGENTIC_FORECAST container with GPU
docker-compose up

# Or run in background
docker-compose up -d

# View logs
docker-compose logs -f agentic-forecast
```

Expected output should show:
```
[*] Attempting to connect to IBKR on localhost:7497...
✅ Connection successful!
Fetching historical data for AAPL, TSLA, NVDA...
[*] Processing features...
[*] Generating forecasts...
[*] Running monitoring...
```

### 6. Interactive Shell (for debugging)

```powershell
# Open interactive bash in running container
docker-compose exec agentic-forecast bash

# Or start a new container
docker-compose run --rm agentic-forecast bash
```

### 7. Stop the Container

```powershell
docker-compose down
```

## Configuration

### Environment Variables

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - IBKR_HOST=127.0.0.1        # TWS/Gateway host
  - IBKR_PORT=7497              # TWS/Gateway port
  - PYTHONUNBUFFERED=1          # Real-time output
```

### TWS Port Reference

- **7497** = TWS Paper Trading (default in docker-compose.yml)
- **7496** = TWS Live Trading
- **4002** = IB Gateway Paper
- **4001** = IB Gateway Live

Change in `docker-compose.yml` if using different port.

## GPU Optimization

The Dockerfile uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04` which includes:
- CUDA 12.4 runtime libraries
- cuDNN support
- Optimized for inference (not training)

All Python dependencies (neuralforecast, torch, etc.) will use GPU automatically.

### Monitor GPU Usage

```powershell
# In another PowerShell window, watch GPU usage
docker exec -it agentic-forecast-gpu nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,utilization.memory --format=csv -l 1
```

## Troubleshooting

### Container can't connect to TWS

**Issue:** Connection refused to 127.0.0.1:7497

**Solutions:**
1. Verify TWS is running: `netstat -ano | findstr :7497`
2. Verify API is enabled in TWS settings (see step 3)
3. Check "Allow connections from localhost only" is UNCHECKED
4. Restart TWS after changing settings
5. Try different port (4001, 4002, 7496) if 7497 is taken

### GPU not detected in container

**Issue:** `no GPU detected` in nvidia-smi output

**Solutions:**
1. Verify Docker Desktop GPU support is enabled
2. Reinstall NVIDIA Container Toolkit
3. Restart Docker Desktop after installation
4. Test: `docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi`

### Out of memory errors

**Solutions:**
1. Reduce batch size in `AGENTIC_FORECAST/config/settings.toml`
2. Free up GPU memory: Close other GPU applications
3. In Docker Desktop settings, increase GPU memory allocation

### Build fails

**Issue:** `pip install` fails during build

**Solutions:**
1. Verify internet connection
2. Clear Docker cache: `docker system prune`
3. Try build again: `docker-compose build --no-cache`

## Next Steps

Once running successfully:

1. **Add Streamlit UI** - Visualize forecasts
2. **Add Prometheus/Grafana** - Monitor metrics
3. **Add LangSmith** - Debug agentic workflows
4. **Deploy to Kubernetes** - Scale horizontally

## Production Deployment

For production on a cloud VM with GPU:

1. Install NVIDIA Container Toolkit on your server
2. Push Docker image to registry (Docker Hub, ECR, etc.)
3. Deploy via docker-compose or Kubernetes
4. Configure SSH tunneling if TWS is on different machine
5. Set up reverse proxy (nginx) for API/Streamlit access

## Architecture Diagram

```
┌─ Your Windows Machine ────────────────────┐
│                                            │
│  ┌─ Interactive Brokers ───────────────┐  │
│  │ TWS or Gateway                       │  │
│  │ Port: 7497 (default)                │  │
│  │ API: Enabled                        │  │
│  └─────────────────────────────────────┘  │
│           ↑                                │
│    127.0.0.1:7497                        │
│           ↑                                │
│  ┌─ Docker Desktop ────────────────────┐  │
│  │ ┌─ AGENTIC_FORECAST Container ────────┐ │  │
│  │ │ Python 3.12                    │ │  │
│  │ │ CUDA 12.4 + cuDNN              │ │  │
│  │ │ LangGraph + ib_insync          │ │  │
│  │ │ neuralforecast + torch (GPU)   │ │  │
│  │ │                                │ │  │
│  │ │ AGENTIC_FORECAST/main.py            │ │  │
│  │ │ ├─ fetch_data                  │ │  │
│  │ │ ├─ feature_engineer            │ │  │
│  │ │ ├─ generate_forecasts (GPU)    │ │  │
│  │ │ └─ run_monitoring              │ │  │
│  │ └────────────────────────────────┘ │  │
│  │ Network: host (--net=host)         │  │
│  │ GPU: NVIDIA GPU with CUDA          │  │
│  └────────────────────────────────────┘  │
│                                            │
└────────────────────────────────────────────┘
```

## References

- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Interactive Brokers TWS API](https://ibkr.info/article/2482)
- [ib_insync Documentation](https://ib-insync.readthedocs.io/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

