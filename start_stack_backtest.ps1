<#
  start_stack_backtest.ps1

  Starts the complete forecasting stack for BACKTEST run:

  - Prometheus (separate PowerShell window)
  - FastAPI /metrics server (separate PowerShell window)
  - Forecast engine backtest run (current window)

  Adjust the paths in the CONFIG section once, then you can just:
    - Right-click → "Run with PowerShell"
    - or run:  .\start_stack_backtest.ps1
#>

############################################################
# CONFIG – ADJUST THESE PATHS ONCE
############################################################

# Path to your project root
$ProjectRoot = "C:\Users\spreu\Documents\agentic_forecast"

# Path to Prometheus folder and config
$PrometheusExe  = "C:\prometheus\prometheus.exe"
$PrometheusConf = "C:\Users\spreu\Documents\agentic_forecast\monitoring\prometheus.yml"

# Python executable (usually just "python" if on PATH)
$Python = "python"

# Run type for the forecast engine: DAILY | WEEKEND_HPO | BACKTEST
$RunType = "BACKTEST"

############################################################
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
############################################################

Write-Host "=== Starting Agentic Forecast Stack (BACKTEST) ===" -ForegroundColor Cyan
Write-Host "Project root : $ProjectRoot"
Write-Host "Run type     : $RunType"
Write-Host ""

# 1) Start Prometheus in a new PowerShell window
if (Test-Path $PrometheusExe) {
    Write-Host "[1/3] Starting Prometheus..." -ForegroundColor Yellow
    $promCmd = "`"$PrometheusExe`" --config.file=`"$PrometheusConf`""
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $promCmd `
        -WindowStyle Minimized
} else {
    Write-Host "[1/3] Skipping Prometheus – executable not found at $PrometheusExe" -ForegroundColor DarkYellow
}

# 2) Start FastAPI server in a new PowerShell window
Write-Host "[2/3] Starting FastAPI server (run_api.py)..." -ForegroundColor Yellow
$apiCmd = "cd `"$ProjectRoot`"; $Python run_api.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiCmd `
    -WindowStyle Minimized

# Short pause so API can start before Prometheus scrapes
Start-Sleep -Seconds 3

# 3) Run the forecasting engine in this window
Write-Host "[3/3] Running forecast engine: main.py --task full --run_type $RunType" -ForegroundColor Yellow
Set-Location $ProjectRoot

& $Python main.py --task full --run_type $RunType

$ExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "=== Forecast engine finished with exit code $ExitCode ===" -ForegroundColor Cyan

if ($ExitCode -eq 0) {
    Write-Host "You can now inspect Grafana at http://localhost:3000" -ForegroundColor Green
    Write-Host "Latest report: results\reports\daily_forecast_health_latest.md" -ForegroundColor Green
} else {
    Write-Host "Something went wrong – check logs in results\ and console output." -ForegroundColor Red
}