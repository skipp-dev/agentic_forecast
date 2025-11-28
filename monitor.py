#!/usr/bin/env python3
"""
Real-time System Monitor

Continuous monitoring of the agentic forecasting system
with live updates and progress tracking.
"""

import sys
import os
import time
import psutil
from datetime import datetime
import pandas as pd

# Add project paths
sys.path.append(os.path.dirname(__file__))

from data.metrics_database import MetricsDatabase

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_system_status():
    """Get comprehensive system status."""
    status = {
        'main_running': False,
        'llm_agents_active': False,
        'gpu_available': False,
        'data_files': 0,
        'processed_files': 0,
        'model_files': 0,
        'forecast_files': 0,
        'log_files': 0,
        'anomalies': 0,
        'drift_events': 0,
        'current_phase': 'Unknown'
    }

    # Check running processes
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and len(cmdline) > 1:
                    cmd_str = ' '.join(cmdline)
                    if 'main.py' in cmd_str and '--task full' in cmd_str:
                        status['main_running'] = True
                        status['current_phase'] = 'Active Forecasting Pipeline'
                    if 'python' in cmd_str and any(word in cmd_str.lower() for word in ['llm', 'agent', 'forecast']):
                        status['llm_agents_active'] = True
            except:
                continue
    except:
        pass

    # Check GPU
    try:
        import torch
        status['gpu_available'] = torch.cuda.is_available()
    except:
        pass

    # Count files
    try:
        # Raw data
        if os.path.exists('data/raw'):
            status['data_files'] = len([f for root, dirs, files in os.walk('data/raw')
                                      for f in files if f.endswith('.csv')])

        # Processed data
        if os.path.exists('data/processed'):
            status['processed_files'] = len([f for root, dirs, files in os.walk('data/processed')
                                           for f in files if f.endswith('.parquet')])

        # Models
        if os.path.exists('models'):
            status['model_files'] = len([f for root, dirs, files in os.walk('models')
                                       for f in files if f.endswith(('.pkl', '.joblib', '.h5', '.pt', '.pth'))])

        # Forecasts/Results
        if os.path.exists('results'):
            status['forecast_files'] = len([f for root, dirs, files in os.walk('results')
                                          for f in files if f.endswith(('.csv', '.json', '.pkl'))])

        # Logs
        if os.path.exists('logs'):
            status['log_files'] = len([f for f in os.listdir('logs') if f.endswith('.log')])

        # Metrics
        if os.path.exists('data/metrics/anomaly_summary_latest.csv'):
            try:
                df = pd.read_csv('data/metrics/anomaly_summary_latest.csv')
                status['anomalies'] = int(df['total_anomalies'].sum())
            except:
                pass

        if os.path.exists('data/metrics/drift_metrics_latest.csv'):
            try:
                df = pd.read_csv('data/metrics/drift_metrics_latest.csv')
                status['drift_events'] = int(df['drift_detected'].sum())
            except:
                pass

    except Exception as e:
        print(f"Error counting files: {e}")

    return status

def get_recent_logs():
    """Get recent log activity."""
    logs = []
    try:
        if os.path.exists('logs'):
            log_files = [f for f in os.listdir('logs') if f.endswith('.log')]
            for log_file in sorted(log_files, key=lambda x: os.path.getmtime(os.path.join('logs', x)), reverse=True)[:3]:
                mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join('logs', log_file)))
                logs.append(f"{log_file}: {mtime.strftime('%H:%M:%S')}")
    except:
        pass
    return logs

def print_monitor(status, logs):
    """Print the monitoring display."""
    clear_screen()

    print("🔴 LIVE SYSTEM MONITOR - Agentic Forecast")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # System Status
    print("🔄 SYSTEM STATUS:")
    print("-" * 20)
    if status['main_running']:
        print("🟢 Main Pipeline: RUNNING (Active)")
        print(f"   📊 Phase: {status['current_phase']}")
    else:
        print("🔴 Main Pipeline: IDLE")
        print("   💡 Run: python main.py --task full")

    gpu_status = "🟢 AVAILABLE" if status['gpu_available'] else "🔴 CPU ONLY"
    print(f"🎮 GPU Status: {gpu_status}")

    llm_status = "🟢 ACTIVE" if status['llm_agents_active'] else "🟡 IDLE"
    print(f"🤖 LLM Agents: {llm_status}")
    print()

    # Data Pipeline
    print("📊 DATA PIPELINE:")
    print("-" * 15)
    print(f"📁 Raw Data Files: {status['data_files']}")
    print(f"🔧 Processed Files: {status['processed_files']}")
    print(f"🤖 Model Files: {status['model_files']}")
    print(f"📈 Forecast Results: {status['forecast_files']}")
    print(f"📋 Log Files: {status['log_files']}")
    print()

    # Risk Monitoring
    print("🛡️ RISK MONITORING:")
    print("-" * 17)
    print(f"🚨 Anomalies Detected: {status['anomalies']}")
    print(f"📊 Drift Events: {status['drift_events']}")
    print()

    # Recent Activity
    print("📋 RECENT ACTIVITY:")
    print("-" * 18)
    if logs:
        for log in logs:
            print(f"📄 {log}")
    else:
        print("📄 No recent activity")
    print()

    # Progress Indicators
    print("📊 PROGRESS INDICATORS:")
    print("-" * 22)
    total_symbols = 576  # Based on watchlist
    processed_ratio = status['processed_files'] / max(total_symbols, 1)
    print(f"🔄 Data Processing: {status['processed_files']}/{total_symbols} symbols ({processed_ratio:.1%})")

    if status['forecast_files'] > 0:
        print(f"🎯 Forecasts Generated: {status['forecast_files']} result files")
    else:
        print("⏳ Forecasts: Pending (waiting for pipeline completion)")
    print()

    # Commands
    print("🚀 QUICK COMMANDS:")
    print("-" * 17)
    print("• Start Full Pipeline: python main.py --task full")
    print("• View Dashboard: python show_dashboard.py")
    print("• Check Logs: Get-ChildItem logs\\*.log -Last 5")
    print("• Exit Monitor: Ctrl+C")
    print()
    print("🔄 Auto-refreshing every 5 seconds... (Ctrl+C to exit)")

def main():
    """Main monitoring loop."""
    print("Starting real-time system monitor...")
    print("Press Ctrl+C to exit")
    time.sleep(2)

    try:
        while True:
            status = get_system_status()
            logs = get_recent_logs()
            print_monitor(status, logs)
            time.sleep(5)  # Update every 5 seconds

    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped by user")
        print("System continues running in background")

if __name__ == "__main__":
    main()