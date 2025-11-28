<#
  check_stack.ps1

  Lightweight health check for your Agentic Forecast stack.

  It checks:
    - FastAPI:     http://127.0.0.1:8000/health
    - Prometheus:  http://localhost:9090/-/ready
    - Grafana:     http://localhost:3000/

  Usage:
    cd C:\Users\spreu\Documents\agentic_forecast
    .\check_stack.ps1

  Exit codes:
    0 = all components healthy
    1 = one or more components unhealthy / unreachable
#>

############################################################
# CONFIG – tweak if you change ports/routes
############################################################

$FastApiHealthUrl   = "http://127.0.0.1:8000/health"
$PrometheusReadyUrl = "http://localhost:9090/-/ready"
$GrafanaUrl         = "http://localhost:3000/"

$TimeoutSeconds = 3

############################################################
# Helper: check a single HTTP endpoint
############################################################

function Test-HttpEndpoint {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name,

        [Parameter(Mandatory = $true)]
        [string] $Url,

        [int] $Timeout = 3,

        # Optional scriptblock to validate response body/json
        [ScriptBlock] $Validate = $null
    )

    Write-Host "Checking $Name at $Url ..." -ForegroundColor Cyan

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $Timeout
        $statusCode = [int]$response.StatusCode

        if ($statusCode -lt 200 -or $statusCode -ge 300) {
            Write-Host "  ❌ $Name returned HTTP $statusCode" -ForegroundColor Red
            return @{
                Name    = $Name
                Url     = $Url
                Healthy = $false
                Status  = "HTTP $statusCode"
            }
        }

        # Optional custom validation on content
        if ($Validate -ne $null) {
            $ok = & $Validate $response
            if (-not $ok) {
                Write-Host "  ❌ $Name responded but validation failed" -ForegroundColor Red
                return @{
                    Name    = $Name
                    Url     = $Url
                    Healthy = $false
                    Status  = "ValidationFailed"
                }
            }
        }

        Write-Host "  ✅ $Name is healthy (HTTP $statusCode)" -ForegroundColor Green
        return @{
            Name    = $Name
            Url     = $Url
            Healthy = $true
            Status  = "OK"
        }
    }
    catch {
        Write-Host "  ❌ $Name unreachable: $($_.Exception.Message)" -ForegroundColor Red
        return @{
            Name    = $Name
            Url     = $Url
            Healthy = $false
            Status  = "Exception: $($_.Exception.Message)"
        }
    }
}

############################################################
# 1) FastAPI health
############################################################

Write-Host "=== Agentic Forecast Stack Health Check ===" -ForegroundColor Yellow
Write-Host ""

# For FastAPI we expect JSON with e.g. {"status": "ok"} but we're lenient
$fastApiResult = Test-HttpEndpoint -Name "FastAPI" -Url $FastApiHealthUrl -Timeout $TimeoutSeconds -Validate {
    param($resp)
    try {
        $json = $resp.Content | ConvertFrom-Json
        return $json.status -eq "ok" -or $json.Status -eq "ok"
    }
    catch {
        # If it's not JSON but 200 OK, we still accept it
        return $true
    }
}

############################################################
# 2) Prometheus ready
############################################################

$prometheusResult = Test-HttpEndpoint -Name "Prometheus" -Url $PrometheusReadyUrl -Timeout $TimeoutSeconds

############################################################
# 3) Grafana UI
############################################################

$grafanaResult = Test-HttpEndpoint -Name "Grafana" -Url $GrafanaUrl -Timeout $TimeoutSeconds

############################################################
# Summary
############################################################

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Yellow

$results = @($fastApiResult, $prometheusResult, $grafanaResult)

foreach ($r in $results) {
    $icon = if ($r.Healthy) { "✅" } else { "❌" }
    Write-Host ("{0} {1} - {2}" -f $icon, $r.Name, $r.Status)
}

$allHealthy = $results | Where-Object { -not $_.Healthy } | Measure-Object | Select-Object -ExpandProperty Count
if ($allHealthy -eq 0) {
    Write-Host ""
    Write-Host "Overall status: ✅ ALL HEALTHY" -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "Overall status: ❌ ONE OR MORE COMPONENTS UNHEALTHY" -ForegroundColor Red
    exit 1
}