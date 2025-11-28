#!/usr/bin/env python3
"""
Demo: Unified Daily Forecast Health Report

This script demonstrates the complete pipeline that generates a unified
Daily Forecast Health Report combining:
1. Metric sanity checks
2. Model performance metrics
3. Cross-asset V2 feature analysis

The report provides automated decision-making for model promotion and
stakeholder communication.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from graphs.state import GraphState
from daily_health_report import daily_health_report_node


def demo_unified_health_report():
    """Demonstrate the unified daily health report generation."""

    print("🚀 Demo: Unified Daily Forecast Health Report")
    print("=" * 60)
    print()

    # Create demo state with realistic configuration
    state = GraphState(
        run_id="demo-run-2025-01-28",
        symbols=["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"],
        horizons=["1", "5", "10"],
        config={
            "feature_store_v2_enabled": True,
            "models_trained": ["AutoNHITS", "AutoDLinear", "AutoTiDE"],
            "hpo_trials_per_symbol": 50,
            "evaluation_horizons": ["1d", "5d", "10d"],
            "cross_asset_features": True
        }
    )

    print("📋 Pipeline Configuration:")
    print(f"  Run ID: {state.run_id}")
    print(f"  Symbols: {', '.join(state.symbols)}")
    print(f"  Horizons: {', '.join(state.horizons)}")
    print(f"  Models: {', '.join(state.config['models_trained'])}")
    print(f"  V2 Features: {'✅ Enabled' if state.config['feature_store_v2_enabled'] else '❌ Disabled'}")
    print()

    # Simulate pipeline steps (in real pipeline these would run actual analysis)
    print("🔄 Simulating Pipeline Steps:")
    print("  1. ✅ Data loading and preprocessing")
    print("  2. ✅ Feature engineering (including V2 cross-asset features)")
    print("  3. ✅ Model training and hyperparameter optimization")
    print("  4. ✅ Model evaluation and performance metrics")
    print("  5. ✅ Metric sanity checks and validation")
    print("  6. ✅ Cross-asset V2 A/B analysis")
    print("  7. 🚀 Generating unified health report...")
    print()

    # Run the daily health report node
    try:
        result_state = daily_health_report_node(state)

        print("✅ Unified health report generated successfully!")
        print()

        # Display results
        json_path = result_state.daily_report_path_json
        md_path = result_state.daily_report_path_md
        can_promote = result_state.can_auto_promote_models

        print("📄 Generated Reports:")
        print(f"  📊 JSON Report: {json_path}")
        print(f"  📝 Markdown Report: {md_path}")
        print()

        print("🤖 Automated Decisions:")
        print(f"  Auto-promotion enabled: {'✅ YES' if can_promote else '❌ NO'}")
        print()

        # Show key insights from the report
        if Path(json_path).exists():
            import json
            with open(json_path, 'r') as f:
                report = json.load(f)

            sanity = report["metric_sanity"]
            perf = report["model_performance"]
            v2 = report["cross_asset_v2"]
            alerts = report["alerts_and_decisions"]

            print("📊 Key Health Metrics:")
            print(f"  Metric Sanity: {'✅ PASSED' if sanity['status'] == 'passed' else '❌ FAILED'} ({sanity['severity']} severity)")
            print(".3f")
            print(f"  V2 Features: {'✅ ENABLED' if v2['status'] == 'enabled' else '❌ DISABLED'} ({v2['decision'].replace('_', ' ')})")
            print()

            print("💡 Automated Actions:")
            for i, action in enumerate(alerts["recommended_actions"][:3], 1):
                print(f"  {i}. {action}")
            print()

        print("🎯 Business Impact:")
        print("  • Single source of truth for system health")
        print("  • Automated guardrail decisions prevent bad deployments")
        print("  • Stakeholder-friendly reports for communication")
        print("  • One-command pipeline experience")
        print()

        print("📋 Files for Review:")
        print(f"  • View JSON report: {json_path}")
        print(f"  • View Markdown report: {json_path.replace('.json', '.md')}")
        print()

        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_usage_examples():
    """Show how to use the unified health report in different scenarios."""

    print("\n📚 Usage Examples:")
    print("-" * 30)

    examples = [
        ("CI/CD Pipeline", "python -m graphs.main_graph --config production.yaml"),
        ("Manual Review", "python demo_unified_health_report.py"),
        ("API Integration", "curl -X GET http://localhost:8000/health/daily"),
        ("Stakeholder Email", "Send results/reports/daily_forecast_health_latest.md"),
        ("Automated Alerts", "Check can_auto_promote_models flag in JSON report")
    ]

    for scenario, command in examples:
        print(f"  {scenario}: {command}")

    print()


if __name__ == "__main__":
    success = demo_unified_health_report()
    show_usage_examples()

    if success:
        print("🎉 Demo completed successfully!")
        print("The unified Daily Forecast Health Report is ready for production use.")
    else:
        print("❌ Demo failed. Check the error messages above.")

    sys.exit(0 if success else 1)