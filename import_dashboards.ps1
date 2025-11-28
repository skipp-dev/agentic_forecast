<#
  import_dashboards.ps1

  Imports Grafana dashboard JSON files from the repo into Grafana.

  Assumes:
  - Grafana is running on http://localhost:3000
  - Default admin/admin credentials (change if you set a password)
  - Dashboards are in monitoring/grafana/dashboards/

  Usage:
    cd C:\Users\spreu\Documents\agentic_forecast
    .\import_dashboards.ps1
#>

############################################################
# CONFIG
############################################################

$GrafanaUrl = "http://localhost:3000"
$Username = "admin"
$Password = "admin"  # Change if you updated the password

$DashboardDir = "C:\Users\spreu\Documents\agentic_forecast\monitoring\grafana\dashboards"

############################################################
# Helper: Import a single dashboard
############################################################

function Import-Dashboard {
    param(
        [Parameter(Mandatory = $true)]
        [string] $JsonFilePath
    )

    if (-not (Test-Path $JsonFilePath)) {
        Write-Host "Dashboard file not found: $JsonFilePath" -ForegroundColor Red
        return
    }

    try {
        $dashboardJson = Get-Content $JsonFilePath -Raw | ConvertFrom-Json

        # Prepare the payload
        $payload = @{
            dashboard = $dashboardJson
            overwrite = $true
        } | ConvertTo-Json -Depth 10

        # Create basic auth header
        $base64AuthInfo = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes(("$Username`:$Password")))
        $headers = @{
            "Authorization" = "Basic $base64AuthInfo"
            "Content-Type" = "application/json"
        }

        $url = "$GrafanaUrl/api/dashboards/db"

        Write-Host "Importing dashboard: $($dashboardJson.title) from $JsonFilePath" -ForegroundColor Cyan

        $response = Invoke-WebRequest -Uri $url -Method Post -Headers $headers -Body $payload -UseBasicParsing

        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Successfully imported: $($dashboardJson.title)" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to import: $($dashboardJson.title) - Status: $($response.StatusCode)" -ForegroundColor Red
        }

    } catch {
        Write-Host "❌ Error importing $JsonFilePath : $($_.Exception.Message)" -ForegroundColor Red
    }
}

############################################################
# Main
############################################################

Write-Host "=== Importing Grafana Dashboards ===" -ForegroundColor Yellow
Write-Host "Grafana URL: $GrafanaUrl"
Write-Host "Dashboard dir: $DashboardDir"
Write-Host ""

# Get all JSON files in the dashboard directory
$jsonFiles = Get-ChildItem -Path $DashboardDir -Filter "*.json"

if ($jsonFiles.Count -eq 0) {
    Write-Host "No JSON files found in $DashboardDir" -ForegroundColor Red
    exit 1
}

foreach ($file in $jsonFiles) {
    Import-Dashboard -JsonFilePath $file.FullName
}

Write-Host ""
Write-Host "=== Import Complete ===" -ForegroundColor Yellow
Write-Host "Visit http://localhost:3000 to view your dashboards!" -ForegroundColor Green