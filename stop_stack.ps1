<#
  stop_stack.ps1

  Stops the components started by start_stack.ps1:

  - Prometheus (prometheus.exe)
  - FastAPI server (python run_api.py)
  - (optional) long-running forecast runs (python main.py --task full ...)

  Run from the project root:

      cd C:\Users\spreu\Documents\agentic_forecast
      .\stop_stack.ps1
#>

############################################################
# CONFIG – you can tweak this if needed
############################################################

# Process names / markers
$PrometheusProcessName = "prometheus"
$FastApiScriptMarker   = "run_api.py"
$MainScriptMarker      = "main.py --task full"

# If you also want to kill ongoing forecast runs, set this to $true
$StopForecastRuns = $false

############################################################
# Helper: stop a list of processes safely
############################################################

function Stop-ProcessesSafely {
    param(
        [Parameter(Mandatory = $true)]
        [System.Collections.IEnumerable] $Processes,
        [string] $Label = "process"
    )

    if (-not $Processes) {
        Write-Host "No $Label processes found." -ForegroundColor DarkYellow
        return
    }

    foreach ($p in $Processes) {
        try {
            Write-Host "Stopping $Label PID=$($p.ProcessId) Name=$($p.Name)" -ForegroundColor Yellow
            Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
        }
        catch {
            Write-Host "Failed to stop $Label PID=$($p.ProcessId): $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

############################################################
# 1) Stop Prometheus
############################################################

Write-Host "=== Stopping Agentic Forecast Stack ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Checking for Prometheus processes..." -ForegroundColor Cyan

try {
    $promProcs = Get-Process -Name $PrometheusProcessName -ErrorAction SilentlyContinue
} catch {
    $promProcs = @()
}

if ($promProcs -and $promProcs.Count -gt 0) {
    # Wrap into objects with ProcessId/Name to reuse helper
    $promWrapped = @($promProcs | ForEach-Object {
        [PSCustomObject]@{
            ProcessId = $_.Id
            Name      = $_.ProcessName
        }
    })
    Stop-ProcessesSafely -Processes $promWrapped -Label "Prometheus"
} else {
    Write-Host "No Prometheus (prometheus.exe) process found." -ForegroundColor DarkYellow
}

############################################################
# 2) Stop FastAPI (python run_api.py)
############################################################

Write-Host ""
Write-Host "[2/3] Checking for FastAPI (run_api.py) processes..." -ForegroundColor Cyan

# Use Win32_Process to inspect command lines
$powershellApiProcs = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like "*$FastApiScriptMarker*" }

if ($powershellApiProcs -and $powershellApiProcs.Count -gt 0) {
    Stop-ProcessesSafely -Processes $powershellApiProcs -Label "FastAPI (run_api.py)"
} else {
    Write-Host "No powershell run_api.py processes found." -ForegroundColor DarkYellow
}

############################################################
# 3) (Optional) Stop long-running forecast engine runs
############################################################

Write-Host ""
Write-Host "[3/3] Checking for forecast engine (main.py --task full) processes..." -ForegroundColor Cyan

if ($StopForecastRuns) {
    $pythonMainProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*$MainScriptMarker*" }

    if ($pythonMainProcs -and $pythonMainProcs.Count -gt 0) {
        Stop-ProcessesSafely -Processes $pythonMainProcs -Label "Forecast engine (main.py --task full)"
    } else {
        Write-Host "No main.py --task full processes found." -ForegroundColor DarkYellow
    }
} else {
    Write-Host "Skipping forecast engine termination (StopForecastRuns = \$false)." -ForegroundColor DarkYellow
}

############################################################
# Done
############################################################

Write-Host ""
Write-Host "=== stop_stack.ps1 completed ===" -ForegroundColor Cyan