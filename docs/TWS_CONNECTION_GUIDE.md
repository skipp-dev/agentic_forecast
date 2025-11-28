# TWS/Gateway Connection Troubleshooting Guide

## Overview

AGENTIC_FORECAST connects to Interactive Brokers TWS (Trader Workstation) or Gateway to retrieve market data. This guide covers setup, configuration, and troubleshooting for different environments.

## Connection Architecture

```
┌─────────────────────┐
│   Your Computer     │
│  ┌───────────────┐  │
│  │  TWS/Gateway  │  │
│  │  (Port 7497)  │  │
│  └───────────────┘  │
│          ▲          │
│          │          │
└──────────┼──────────┘
           │
    ┌──────┴────────┐
    │               │
    ▼               ▼
┌─────────┐  ┌──────────────────┐
│ Bare    │  │ Docker / Dev      │
│ Metal   │  │ Container        │
│ Python  │  │ Python           │
└─────────┘  └──────────────────┘
```

## Installation & Prerequisites

### Prerequisites
- Interactive Brokers account (paper or live)
- TWS or IB Gateway installed and running on your local machine
- API access enabled in TWS/Gateway settings

### Setup Steps

1. **Install TWS/Gateway**
   - Download from [Interactive Brokers website](https://www.interactivebrokers.com/en/trading/platforms/tws.php)
   - Install on your local machine (not in container)

2. **Enable API Access in TWS/Gateway**
   - Open TWS or IB Gateway
   - Go to **Edit → Settings (or File → Global Configuration)**
   - Navigate to **API → Settings**
   - ✅ Check "Enable ActiveX and Socket Clients"
   - Note the port number (default: 7497 for TWS paper trading)
   - Click **OK**

3. **Note Your Port Number**
   - **Paper Trading (default)**: 7497
   - **Live Trading**: 7496
   - **Gateway Paper**: 4002
   - **Gateway Live**: 4001

## Environment-Specific Configuration

### Option 1: Local Development (Bare Metal Python)

**Configuration:**
```toml
# config/settings.toml
[ibkr]
host = "localhost"
ports = [7497, 7496, 4002, 4001]
client_id = 1
```

**Testing Connection:**
```bash
python diagnose_tws.py
```

**Expected Output:**
```
✅ Port 7497: OPEN
✅ Successfully connected to TWS at localhost:7497
```

---

### Option 2: Dev Container (VS Code Remote Containers)

#### Issue: Dev Container in Bridge Mode
When using VS Code dev containers on Windows/Mac, the container runs in **bridge mode** by default, not host mode. This requires using `host.docker.internal` instead of `localhost`.

**Solution:**
The connection logic now automatically tries both:
1. `localhost` (for native Linux host mode)
2. `host.docker.internal` (for Docker Desktop on Windows/Mac)

**No configuration needed** - it works automatically!

**Verification:**
```bash
python diagnose_tws.py
```

Should show:
```
✅ Network IPs available: [172.17.0.X]
Attempting connection to IBKR at localhost:7497...
  ❌ Failed
Attempting connection to IBKR at host.docker.internal:7497...
  ✅ Successfully connected!
```

---

### Option 3: Docker Container (Full Containerization)

#### On Windows/Mac with Docker Desktop:
```bash
docker run --gpus all -v $(pwd):/app agentic-forecast
```

**Why**: Docker Desktop on Windows/Mac handles `host.docker.internal` resolution automatically.

#### On Linux with Docker:
```bash
docker run --gpus all --network host -v $(pwd):/app agentic-forecast
```

**Why**: Linux Docker can use `--network host` for direct host access.

---

## Troubleshooting

### Problem: "Connection failed: [Errno 111] Connect call failed"

**Cause:** Port is not accessible or TWS is not listening

**Solutions:**
1. **Verify TWS is running**
   - Check taskbar/system tray
   - Confirm it's the right TWS instance (Paper vs Live)

2. **Verify API is enabled**
   - Open TWS/Gateway
   - Go to Settings → API
   - ✅ "Enable ActiveX and Socket Clients" must be checked
   - Note the port number

3. **Check Windows Firewall** (Windows only)
   - Open Windows Defender Firewall
   - Click "Allow an app through firewall"
   - TWS should be listed and allowed
   - Or temporarily disable firewall to test

4. **Check macOS Firewall** (Mac only)
   - System Preferences → Security & Privacy → Firewall
   - Click "Firewall Options"
   - Add TWS to allowed apps

5. **Test with different port**
   ```bash
   # Edit config/settings.toml
   [ibkr]
   ports = [7497]  # Try just paper trading first
   ```

---

### Problem: "No GPU devices found" but ports are open

This is normal! The system falls back to CPU. To verify TWS connection is working:
```bash
python diagnose_tws.py
```

Should show:
```
✅ Port 7497: OPEN
✅ Successfully connected to TWS at localhost:7497
```

---

### Problem: Dev Container can't reach host TWS

**Symptoms:**
- `diagnose_tws.py` shows network available (172.17.x.x)
- All ports show CLOSED
- `host.docker.internal` not working

**Solutions:**

1. **Verify Docker Desktop setting (Windows/Mac)**
   - Docker Desktop → Settings → Resources
   - ✅ "Use Docker Compose V2" enabled (if using compose)

2. **Test host.docker.internal directly**
   ```bash
   ping host.docker.internal
   ```
   Should resolve to host IP (e.g., 192.168.x.x or similar)

3. **Check Windows Firewall blocking host.docker.internal**
   - Create firewall rule for port 7497
   - Or temporarily disable firewall

4. **Fallback: Use IP address instead of hostname**
   ```bash
   # Find your host machine IP
   ipconfig getifaddr en0  # macOS
   hostname -I            # Linux
   ipconfig               # Windows
   
   # Then in config/settings.toml
   [ibkr]
   host = "192.168.x.x"  # Your actual host IP
   ```

---

### Problem: Connection works locally but fails in Docker

**Docker Compose** may override network settings.

**Solution:**
Edit `docker-compose.yml`:
```yaml
services:
  agentic-forecast:
    # ... other settings ...
    
    # For Windows/Mac (Docker Desktop)
    extra_hosts:
      - "host.docker.internal:host-gateway"
    
    # For Linux (uncomment if needed)
    # network_mode: "host"
```

---

## Network Modes Explained

| Mode | Platform | Behavior | Access to Host |
|------|----------|----------|-----------------|
| **Bridge** | Default | Container on separate network | Via `host.docker.internal` |
| **Host** | Linux only | Container shares host network | Direct via `localhost` |
| **None** | All | No network access | ❌ |

---

## Diagnostic Commands

### Quick Network Test
```bash
python diagnose_tws.py
```

### Manual Port Test (Linux/Mac)
```bash
# Test if port is listening
lsof -i :7497
netstat -tuln | grep 7497

# Test connection
telnet localhost 7497
nc -zv localhost 7497
```

### Manual Port Test (Windows)
```bash
# Test if port is listening
netstat -ano | findstr :7497

# Test connection
Test-NetConnection -ComputerName localhost -Port 7497
```

---

## Default Configuration

The system comes with reasonable defaults that auto-detect the connection:

```toml
# config/settings.toml
[ibkr]
# Automatically tries both localhost and host.docker.internal
host = "localhost"

# All standard ports in order
ports = [7497, 7496, 4002, 4001]

# Rotating client IDs to avoid conflicts
client_id = 1
```

---

## Connection Flow Diagram

```
START
  │
  ├─→ Try localhost:7497
  │   ├─→ Success? ✅ DONE
  │   └─→ Fail? Continue
  │
  ├─→ Try localhost:7496
  │   ├─→ Success? ✅ DONE
  │   └─→ Fail? Continue
  │
  ├─→ Try host.docker.internal:7497 (Docker only)
  │   ├─→ Success? ✅ DONE
  │   └─→ Fail? Continue
  │
  └─→ All ports failed?
      └─→ Use mock data for testing
          ✅ System continues with test data
```

---

## Best Practices

1. **Use Paper Trading First**
   - Lower risk for testing
   - Default port 7497
   - Perfect for development

2. **Keep One Connection Open**
   - Close other TWS instances
   - Avoid port conflicts
   - Reduces connection attempts

3. **Monitor Connection Status**
   ```bash
   # Watch the logs
   tail -f logs/AGENTIC_FORECAST.log
   ```

4. **Use Different Client IDs**
   - Each connection needs unique client_id
   - Default increments automatically
   - AGENTIC_FORECAST handles this

5. **Test Before Production**
   ```bash
   python diagnose_tws.py
   ```

---

## Still Having Issues?

1. **Run diagnostic**
   ```bash
   python diagnose_tws.py
   ```

2. **Check TWS settings**
   - API enabled? ✅
   - Port number matches? ✅
   - No firewall blocking? ✅

3. **Review logs**
   ```bash
   # Check detailed connection logs
   tail -f logs/AGENTIC_FORECAST.log | grep -i "connection\|tws\|ibkr"
   ```

4. **Try manual connection**
   ```python
   from ib_insync import IB
   ib = IB()
   ib.connect('localhost', 7497, clientId=1)
   print(ib.isConnected())
   ```

5. **Check ib_insync version**
   ```python
   import ib_insync
   print(ib_insync.__version__)
   ```

---

## Reference

- [Interactive Brokers TWS](https://www.interactivebrokers.com/en/trading/platforms/tws.php)
- [ib_insync Documentation](https://github.com/erdewit/ib_insync)
- [Docker host.docker.internal](https://docs.docker.com/desktop/networking/#host-networking)

