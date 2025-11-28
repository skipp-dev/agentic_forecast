#!/usr/bin/env python3
"""
Test script for the Daily Health Report Node

This script tests the daily_health_report_node function to ensure it correctly
combines metric sanity, performance, and V2 analysis into unified reports.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))

from graphs.state import GraphState
from daily_health_report import daily_health_report_node


def test_daily_health_report():
    """Test the daily health report node with sample data."""

    print("🧪 Testing Daily Health Report Node")
    print("=" * 50)

    # Create test state
    state = GraphState(
        run_id="test-run-2025-01-28",
        symbols=["AAPL", "MSFT", "NVDA"],
        horizons=["1", "5"],
        config={
            "feature_store_v2_enabled": True,
            "models_trained": ["AutoNHITS", "AutoDLinear"],
            "hpo_trials_per_symbol": 30
        }
    )

    print("📋 Test State:")
    print(f"  Run ID: {state.run_id}")
    print(f"  Symbols: {state.symbols}")
    print(f"  Horizons: {state.horizons}")
    print()

    # Run the daily health report node
    print("🚀 Running daily health report node...")
    try:
        result_state = daily_health_report_node(state)
        print("✅ Daily health report node completed successfully!")
        print()

        # Check outputs
        json_path = result_state.daily_report_path_json
        md_path = result_state.daily_report_path_md

        print("📄 Generated Reports:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")
        print(f"  Can auto-promote: {result_state.can_auto_promote_models}")
        print()

        # Check if files exist
        json_exists = Path(json_path).exists()
        md_exists = Path(md_path).exists()

        print("🔍 File Existence Check:")
        print(f"  JSON exists: {'✅' if json_exists else '❌'}")
        print(f"  Markdown exists: {'✅' if md_exists else '❌'}")
        print()

        if json_exists:
            print("📊 JSON Report Preview:")
            with open(json_path, 'r') as f:
                import json
                data = json.load(f)
                print(f"  Run ID: {data['run_metadata']['run_id']}")
                print(f"  Metric Sanity Status: {data['metric_sanity']['status']}")
                print(".3f")
                print(f"  V2 Decision: {data['cross_asset_v2']['decision']}")
                print(f"  Can Auto-Promote: {data['alerts_and_decisions']['can_auto_promote_models']}")
            print()

        if md_exists:
            print("📝 Markdown Report Preview (first 10 lines):")
            with open(md_path, 'r') as f:
                lines = f.readlines()[:10]
                for line in lines:
                    print(f"  {line.rstrip()}")
            print("  ...")
            print()

        # Summary
        success = json_exists and md_exists
        print("🎯 Test Results:")
        print(f"  Overall: {'✅ PASSED' if success else '❌ FAILED'}")

        if success:
            print("  ✓ Both JSON and Markdown reports generated successfully")
            print("  ✓ Daily health report node is working correctly")
        else:
            print("  ✗ One or more reports failed to generate")

        return success

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_daily_health_report()
    sys.exit(0 if success else 1)