@echo off
REM Daily Pipeline Automation Setup
REM This batch file sets up automated execution of the agentic forecasting pipeline

echo Agentic Forecast - Daily Pipeline Setup
echo ======================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python and add to PATH.
    pause
    exit /b 1
)

echo Python found. Setting up daily automation...

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Setup is complete - user needs to manually configure Windows Task Scheduler
echo.
echo SETUP COMPLETE!
echo.
echo To complete automation setup:
echo 1. Open Windows Task Scheduler
echo 2. Create a new task with these settings:
echo    - Name: AgenticForecast_DailyPipeline
echo    - Trigger: Daily at 6:00 AM
echo    - Action: Start a program
echo    - Program: cmd.exe
echo    - Arguments: /c "cd /d C:\Users\%USERNAME%\Documents\agentic_forecast && python scripts\daily_pipeline.ps1"
echo    - Start in: C:\Users\%USERNAME%\Documents\agentic_forecast
echo.
echo Or run the PowerShell script manually:
echo powershell scripts\setup_task_scheduler.ps1
echo.
pause