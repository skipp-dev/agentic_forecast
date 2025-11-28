@echo off
REM Full Forecast Run Batch Script
REM Startet einen kompletten Forecast-Run für alle Symbole

echo ================================================================================
echo AGENTIC FORECAST - FULL RUN FOR ALL SYMBOLS
echo ================================================================================
echo Start Time: %date% %time%
echo ================================================================================
echo.

REM Wechsle ins Projektverzeichnis
cd /d "%~dp0"

REM Setze Python-Pfad
set PYTHON="C:\Program Files\Python312\python.exe"

REM Überprüfe ob Python verfügbar ist
%PYTHON% --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found at %PYTHON%
    pause
    exit /b 1
)

echo.
echo Starting full pipeline execution...
echo.
echo ================================================================================
echo PIPELINE OUTPUT:
echo ================================================================================
echo.

REM Führe main.py aus
%PYTHON% main.py

echo.
echo ================================================================================
if %errorlevel% equ 0 (
    echo PIPELINE COMPLETED SUCCESSFULLY
) else (
    echo PIPELINE FAILED WITH EXIT CODE: %errorlevel%
)
echo End Time: %date% %time%
echo ================================================================================
echo.

pause
