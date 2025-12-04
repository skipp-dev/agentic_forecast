#!/usr/bin/env python3

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(__file__))

from data.metrics_database import MetricsDatabase, MetricQuery

def show_dashboard():
    """Display the analytics dashboard with current system status."""

    print("🚀 Agentic Forecast Analytics Dashboard")
    print("=" * 50)

    try:
        # Initialize components
        metrics_db = MetricsDatabase()

        # Get system status
        print("\n📊 System Status:")
        print("-" * 20)

        # Check for recent metrics using available methods
        try:
            # Try to get some basic metrics
            system_metrics = metrics_db.get_metric_stats('system.cpu_usage', hours=24)
            if system_metrics.get('count', 0) > 0:
                print(f"✅ System metrics available: {system_metrics['count']} data points")
                print(f"📊 Average CPU usage: {system_metrics.get('avg', 0):.2f}%")
            else:
                print("⚠️  No recent system metrics found")
        except Exception as e:
            print(f"⚠️  Metrics database query failed: {e}")

        # Agent Architecture Overview
        print("\n🤖 Agent Architecture (20 Specialized Agents):")
        print("-" * 45)

        agents_info = {
            "Core Orchestration": [
                "orchestrator_agent.py - Main coordinator with GPU orchestration",
                "supervisor_agent.py - Base workflow supervisor",
                "guardrail_agent.py - Risk management and safety checks"
            ],
            "Data & Features": [
                "feature_agent.py - Feature generation from raw data",
                "feature_engineer_agent.py - Advanced feature engineering",
                "alpha_vantage_data_agent.py - Stock market data",
                "commodity_data_agent.py - Commodity price data",
                "macro_data_agent.py - Economic indicators"
            ],
            "Forecasting & Models": [
                "forecast_agent.py - Time series forecasting with GPU accel",
                "global_model_agent.py - Cross-asset model coordination",
                "hyperparameter_search_agent.py - HPO optimization"
            ],
            "LLM-Powered Agents": [
                "llm_analytics_agent.py - Performance analysis & insights",
                "llm_hpo_planner_agent.py - Hyperparameter planning",
                "llm_news_agent.py - News sentiment & impact analysis",
                "openai_research_agent.py - Autonomous external news & market intelligence (no manual prompts needed)"
            ],
            "Monitoring & Reporting": [
                "drift_monitor_agent.py - Model drift detection",
                "monitoring_agent.py - System health monitoring",
                "reporting_agent.py - Report generation"
            ]
        }

        total_agents = sum(len(agents) for agents in agents_info.values())
        print(f"📊 Total Agents: {total_agents} specialized agents")

        for category, agents in agents_info.items():
            print(f"\n🔹 {category} ({len(agents)} agents):")
            for agent in agents:
                print(f"   • {agent}")

        print("\n🔄 Agent Workflow Integration:")
        print("   Graph-based execution with LangGraph framework")
        print("   Conditional routing based on data quality, drift, and performance")
        print("   GPU acceleration for compute-intensive forecasting tasks")

        # Sample User Prompts
        print("\n💬 Sample User Prompts:")
        print("-" * 23)

        sample_prompts = [
            "Research latest news and sentiment for TSLA",  # Manual override available
            "Gather external market intelligence for portfolio",  # Manual override available
            "Analyze breaking news impact on VIX futures",  # Manual override available
            "Monitor social sentiment for crypto markets",  # Manual override available
            "[AUTONOMOUS] Agent automatically researches all market segments",  # Autonomous mode
            "[AUTONOMOUS] Covers: stocks, indices, commodities, crypto, economic data",  # Autonomous mode
            "[AUTONOMOUS] Includes: Fed policy, labor market, geopolitical events",  # Autonomous mode
            "[AUTONOMOUS] Generates trading signals across all assets",  # Autonomous mode
            "[AUTONOMOUS] No manual prompts needed - fully autonomous operation",  # Autonomous mode
            "Override autonomous mode for specific research focus"  # Manual override option
        ]

        for i, prompt in enumerate(sample_prompts, 1):
            print(f"   {i:2d}. {prompt}")

        print("\n💡 How to Use Prompts:")
        print("   • Interactive mode: python interactive.py")
        print("   • Direct execution: python main.py --task full")
        print("   • Custom prompts modify agent behavior and focus")

        # Check for test data files (from our successful tests)
        print("\n🧪 Test Data Status:")
        print("-" * 18)

        test_files = [
            'data/raw/AAPL_test.csv',
            'data/raw/MSFT_test.csv',
            'data/raw/GOOGL_test.csv',
            'data/raw/AAPL_seq.csv',
            'data/raw/MSFT_seq.csv'
        ]

        test_data_found = 0
        for file_path in test_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"✅ {os.path.basename(file_path)}: {size} bytes")
                test_data_found += 1
            else:
                print(f"❌ {os.path.basename(file_path)}: Not found")

        if test_data_found > 0:
            print(f"\n📊 Data ingestion tests successful: {test_data_found}/{len(test_files)} files created")
        else:
            print("\n⚠️  No test data files found")

        # Check for guardrail logs
        print("\n🛡️ Guardrail Status:")
        print("-" * 18)

        guardrail_files = [
            'guardrail_log_latest.txt',
            'guardrail_risk_snapshot_latest.csv'
        ]

        for filename in guardrail_files:
            if os.path.exists(filename):
                stat = os.stat(filename)
                mtime = datetime.fromtimestamp(stat.st_mtime)
                print(f"✅ {filename}: {stat.st_size} bytes (modified: {mtime.strftime('%Y-%m-%d %H:%M')})")
            else:
                print(f"❌ {filename}: Not found")

        # Check for model files
        print("\n🤖 Model Status:")
        print("-" * 15)

        models_dir = 'models'
        if os.path.exists(models_dir):
            model_files = []
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    if file.endswith(('.pkl', '.joblib', '.h5', '.pt', '.pth')):
                        model_files.append(os.path.join(root, file))

            if model_files:
                print(f"✅ Models found: {len(model_files)} files")
                # Show most recent
                model_files.sort(key=os.path.getmtime, reverse=True)
                recent = model_files[0]
                mtime = datetime.fromtimestamp(os.path.getmtime(recent))
                print(f"📅 Most recent: {os.path.basename(recent)} ({mtime.strftime('%Y-%m-%d %H:%M')})")
            else:
                print("⚠️  No trained models found")
        else:
            print("❌ Models directory not found")

        # Check for logs
        print("\n📋 Recent Activity:")
        print("-" * 18)

        logs_dir = 'logs'
        if os.path.exists(logs_dir):
            log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
            if log_files:
                print(f"✅ Log files: {len(log_files)}")
                # Check most recent log modification
                log_paths = [os.path.join(logs_dir, f) for f in log_files]
                most_recent = max(log_paths, key=os.path.getmtime)
                mtime = datetime.fromtimestamp(os.path.getmtime(most_recent))
                print(f"📅 Last activity: {os.path.basename(most_recent)} ({mtime.strftime('%Y-%m-%d %H:%M')})")
            else:
                print("⚠️  No log files found")
        else:
            print("❌ Logs directory not found")

        # Show recommendations
        print("\n💡 System Health Check:")
        print("-" * 23)

        issues = []
        recommendations = []

        # Check if models exist
        if not os.path.exists('models') or not any(f.endswith(('.pkl', '.joblib', '.h5', '.pt', '.pth'))
                                                  for root, dirs, files in os.walk('models')
                                                  for f in files):
            issues.append("No trained models found")
            recommendations.append("Run full forecasting cycle to train models")

        # Check if guardrail logs exist
        if not os.path.exists('guardrail_log_latest.txt'):
            issues.append("No guardrail activity detected")
            recommendations.append("Execute forecasting to trigger guardrail monitoring")

        # Check if recent logs exist
        if not os.path.exists('logs') or not any(f.endswith('.log') for f in os.listdir('logs')):
            issues.append("No recent system activity")
            recommendations.append("Run system to generate activity logs")

        if issues:
            print("⚠️  Issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ All systems operational")

        print("\n🎯 Recommendations:")
        print("-" * 18)
        if recommendations:
            for rec in recommendations:
                print(f"• {rec}")
        else:
            print("• System is ready for production use")
            print("• Monitor guardrail logs for risk management")
            print("• Review model performance metrics regularly")

        print("\n🚀 Quick Actions:")
        print("-" * 15)
        print("• Run full cycle: python main.py --task full")
        print("• Test connection: python test_connection.py")
        print("• View this dashboard: python show_dashboard.py")

    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_dashboard()