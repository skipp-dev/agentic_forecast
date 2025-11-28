# Automated Daily Pipeline Scheduler for Windows Task Scheduler
# This script runs the complete agentic forecasting pipeline daily

param(
    [string]$PythonPath = "python",
    [string]$ProjectPath = $PSScriptRoot,
    [string]$LogPath = "$PSScriptRoot\logs",
    [switch]$SkipPhase2 = $false
)

# Ensure log directory exists
if (!(Test-Path $LogPath)) {
    New-Item -ItemType Directory -Path $LogPath | Out-Null
}

# Set execution policy for this session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

# Get current timestamp
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logFile = "$LogPath\daily_pipeline_$timestamp.log"

# Function to log messages
function Write-Log {
    param([string]$Message)
    $logMessage = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): $Message"
    Write-Host $logMessage
    Add-Content -Path $logFile -Value $logMessage
}

Write-Log "Starting automated daily pipeline execution"
Write-Log "Project Path: $ProjectPath"
Write-Log "Python Path: $PythonPath"

# Change to project directory
Set-Location $ProjectPath

# Check if Python is available
try {
    $pythonVersion = & $PythonPath --version 2>&1
    Write-Log "Python version: $pythonVersion"
} catch {
    Write-Log "ERROR: Python not found at $PythonPath"
    exit 1
}

# Check if virtual environment exists and activate it
$venvPath = "$ProjectPath\venv"
if (Test-Path "$venvPath\Scripts\activate.ps1") {
    Write-Log "Activating virtual environment"
    & "$venvPath\Scripts\activate.ps1"
    $PythonPath = "$venvPath\Scripts\python.exe"
} elseif (Test-Path "$venvPath\bin\activate") {
    Write-Log "Activating Linux-style virtual environment"
    & "$venvPath\bin\activate"
    $PythonPath = "$venvPath\bin\python"
} else {
    Write-Log "No virtual environment found, using system Python"
}

# Step 1: Data Ingestion
Write-Log "Step 1: Running data ingestion"
try {
    $dataOutput = "$ProjectPath\data\raw_data_$timestamp.pkl"
    & $PythonPath "scripts\run_data_ingestion.py" --output $dataOutput --log-level INFO
    if ($LASTEXITCODE -eq 0) {
        Write-Log "Data ingestion completed successfully"
    } else {
        Write-Log "ERROR: Data ingestion failed with exit code $LASTEXITCODE"
        exit 1
    }
} catch {
    Write-Log "ERROR: Data ingestion failed: $_"
    exit 1
}

# Step 2: Feature Engineering
Write-Log "Step 2: Running feature engineering"
try {
    $featureOutput = "$ProjectPath\data\features_$timestamp.pkl"
    & $PythonPath "scripts\run_feature_engineering.py" --input $dataOutput --output $featureOutput --log-level INFO
    if ($LASTEXITCODE -eq 0) {
        Write-Log "Feature engineering completed successfully"
    } else {
        Write-Log "ERROR: Feature engineering failed with exit code $LASTEXITCODE"
        exit 1
    }
} catch {
    Write-Log "ERROR: Feature engineering failed: $_"
    exit 1
}

# Step 3: Model Training and Forecasting
Write-Log "Step 3: Running model training and forecasting"
try {
    $forecastOutput = "$ProjectPath\models\forecasts_$timestamp.pkl"
    & $PythonPath "scripts\run_model_training.py" --input $featureOutput --output $forecastOutput --log-level INFO
    if ($LASTEXITCODE -eq 0) {
        Write-Log "Model training and forecasting completed successfully"
    } else {
        Write-Log "ERROR: Model training failed with exit code $LASTEXITCODE"
        exit 1
    }
} catch {
    Write-Log "ERROR: Model training failed: $_"
    exit 1
}

# Step 4: Monitoring
Write-Log "Step 4: Running monitoring"
try {
    $monitoringOutput = "$ProjectPath\data\monitoring_$timestamp.pkl"
    & $PythonPath "scripts\run_monitoring.py" --input $dataOutput --output $monitoringOutput --log-level INFO
    if ($LASTEXITCODE -eq 0) {
        Write-Log "Monitoring completed successfully"
    } else {
        Write-Log "ERROR: Monitoring failed with exit code $LASTEXITCODE"
        # Don't exit here as monitoring failure shouldn't stop the pipeline
        Write-Log "Continuing despite monitoring failure..."
    }
} catch {
    Write-Log "ERROR: Monitoring failed: $_"
    Write-Log "Continuing despite monitoring failure..."
}

# Step 5: Phase 2 Pipeline (optional)
if (!$SkipPhase2) {
    Write-Log "Step 5: Running Phase 2 pipeline (macro data, regimes, strategies)"
    try {
        $phase2Output = "$ProjectPath\data\phase2_results_$timestamp.pkl"
        & $PythonPath "scripts\run_phase2_pipeline.py" --output $phase2Output --log-level INFO
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Phase 2 pipeline completed successfully"
        } else {
            Write-Log "ERROR: Phase 2 pipeline failed with exit code $LASTEXITCODE"
            # Don't exit here as Phase 2 is additional functionality
            Write-Log "Continuing despite Phase 2 failure..."
        }
    } catch {
        Write-Log "ERROR: Phase 2 pipeline failed: $_"
        Write-Log "Continuing despite Phase 2 failure..."
    }
} else {
    Write-Log "Skipping Phase 2 pipeline as requested"
}

# Step 6: Generate Reports
Write-Log "Step 6: Generating reports"
try {
    # Run the main graph to generate analytics and reports
    & $PythonPath "main.py" 2>&1 | Out-File -FilePath "$LogPath\main_execution_$timestamp.log" -Append
    if ($LASTEXITCODE -eq 0) {
        Write-Log "Report generation completed successfully"
    } else {
        Write-Log "ERROR: Report generation failed with exit code $LASTEXITCODE"
        exit 1
    }
} catch {
    Write-Log "ERROR: Report generation failed: $_"
    exit 1
}

# Cleanup old files (keep last 7 days)
Write-Log "Step 7: Cleaning up old files"
try {
    $cutoffDate = (Get-Date).AddDays(-7)

    # Clean up old data files
    Get-ChildItem "$ProjectPath\data\*.pkl" | Where-Object { $_.LastWriteTime -lt $cutoffDate } | Remove-Item -Force
    Get-ChildItem "$ProjectPath\models\*.pkl" | Where-Object { $_.LastWriteTime -lt $cutoffDate } | Remove-Item -Force

    # Clean up old logs (keep last 30 days)
    $logCutoffDate = (Get-Date).AddDays(-30)
    Get-ChildItem "$LogPath\*.log" | Where-Object { $_.LastWriteTime -lt $logCutoffDate } | Remove-Item -Force

    Write-Log "Cleanup completed"
} catch {
    Write-Log "Warning: Cleanup failed: $_"
}

Write-Log "Daily pipeline execution completed successfully"

# Send notification (optional - requires mail setup)
# You can uncomment and configure the following for email notifications:
#
# $smtpServer = "your-smtp-server.com"
# $smtpPort = 587
# $username = "your-email@domain.com"
# $password = "your-password"
#
# Send-MailMessage -From "pipeline@agenticforecast.com" `
#                  -To "admin@yourcompany.com" `
#                  -Subject "Daily Pipeline Completed Successfully" `
#                  -Body "The automated daily pipeline has completed successfully at $timestamp" `
#                  -SmtpServer $smtpServer `
#                  -Port $smtpPort `
#                  -Credential (New-Object System.Management.Automation.PSCredential ($username, (ConvertTo-SecureString $password -AsPlainText -Force))) `
#                  -UseSsl

exit 0</content>
<parameter name="filePath">c:\Users\spreu\Documents\agentic_forecast\scripts\daily_pipeline.ps1