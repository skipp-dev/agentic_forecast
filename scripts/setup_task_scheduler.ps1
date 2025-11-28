# Windows Task Scheduler Setup Script
# This script creates a scheduled task to run the daily pipeline

param(
    [string]$TaskName = "AgenticForecast_DailyPipeline",
    [string]$TaskTime = "06:00",  # 6:00 AM daily
    [string]$PythonPath = "python",
    [switch]$Uninstall = $false
)

Write-Host "Agentic Forecast - Windows Task Scheduler Setup"
Write-Host "================================================"

if ($Uninstall) {
    Write-Host "Uninstalling scheduled task: $TaskName"

    # Check if task exists
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

    if ($existingTask) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Successfully uninstalled task: $TaskName"
    } else {
        Write-Host "Task $TaskName not found - nothing to uninstall"
    }

    exit 0
}

# Get current script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir

Write-Host "Project directory: $projectDir"
Write-Host "Task name: $TaskName"
Write-Host "Scheduled time: $TaskTime"

# Check if PowerShell script exists
$psScriptPath = Join-Path $scriptDir "daily_pipeline.ps1"
if (!(Test-Path $psScriptPath)) {
    Write-Host "ERROR: PowerShell script not found at $psScriptPath"
    exit 1
}

# Check if Python is available
try {
    $pythonVersion = & $PythonPath --version 2>&1
    Write-Host "Python found: $pythonVersion"
} catch {
    Write-Host "ERROR: Python not found at $PythonPath"
    Write-Host "Please ensure Python is installed and accessible"
    exit 1
}

# Remove existing task if it exists
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Removing existing task: $TaskName"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create new scheduled task
Write-Host "Creating scheduled task..."

# Build the PowerShell command
$psCommand = "powershell.exe -ExecutionPolicy Bypass -File `"$psScriptPath`" -PythonPath `"$PythonPath`""

# Create the task action
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$psScriptPath`" -PythonPath `"$PythonPath`""

# Create daily trigger
$trigger = New-ScheduledTaskTrigger -Daily -At $TaskTime

# Create task settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Create task principal (run with highest privileges)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken -RunLevel Highest

# Register the task
try {
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Automated daily pipeline for Agentic Forecast system"
    Write-Host "Successfully created scheduled task: $TaskName"
    Write-Host ""
    Write-Host "Task Details:"
    Write-Host "  - Runs daily at: $TaskTime"
    Write-Host "  - Task Name: $TaskName"
    Write-Host "  - Command: $psCommand"
    Write-Host ""
    Write-Host "To view/edit the task, open Task Scheduler and look for '$TaskName'"
    Write-Host "To test the task manually, you can run: $psCommand"

} catch {
    Write-Host "ERROR: Failed to create scheduled task: $_"
    Write-Host ""
    Write-Host "Troubleshooting:"
    Write-Host "1. Make sure you have administrator privileges"
    Write-Host "2. Try running this script as Administrator"
    Write-Host "3. Check that Task Scheduler service is running"
    exit 1
}

Write-Host ""
Write-Host "Setup completed successfully!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. The task will run automatically at $TaskTime daily"
Write-Host "2. Check the logs directory for execution logs"
Write-Host "3. Monitor the first few runs to ensure everything works correctly"
Write-Host "4. Consider setting up email notifications in the PowerShell script"

# Optional: Run a test execution
Write-Host ""
$runTest = Read-Host "Would you like to run a test execution now? (y/n)"
if ($runTest -eq 'y' -or $runTest -eq 'Y') {
    Write-Host "Running test execution..."
    try {
        & powershell.exe -ExecutionPolicy Bypass -File $psScriptPath -PythonPath $PythonPath
        Write-Host "Test execution completed. Check the logs for results."
    } catch {
        Write-Host "Test execution failed: $_"
    }
}</content>
<parameter name="filePath">c:\Users\spreu\Documents\agentic_forecast\scripts\setup_task_scheduler.ps1