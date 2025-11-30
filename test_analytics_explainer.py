#!/usr/bin/env python3
"""
Test script for the new LLMAnalyticsExplainerAgent with SmithLLMClient

This script tests the agent with the example metrics payload to ensure
it works correctly with the new prompt structure and JSON output schema.
"""

import os
import sys
import json
from pathlib import Path

# Add src and agents to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "agents"))

from llm_analytics_agent import LLMAnalyticsExplainerAgent


def test_analytics_explainer():
    """Test the LLMAnalyticsExplainerAgent with example metrics."""

    print("=== Testing LLMAnalyticsExplainerAgent ===")
    print()

    # Example metrics payload from the user's specification
    metrics_payload = {
        "run_metadata": {
            "run_type": "DAILY",
            "date": "2025-11-29",
            "universe_size": 42,
            "cross_asset_v2_enabled": True
        },
        "metrics_global": {
            "mae": {"mean": 0.023, "trend": "decreasing"},
            "rmse": {"mean": 0.031, "trend": "decreasing"},
            "mape": {"mean": 0.048, "trend": "stable"},
            "smape": {"mean": 0.052, "trend": "increasing"},
            "swase": {"mean": 0.61, "trend": "decreasing"},
            "directional_accuracy": {"mean": 0.62, "trend": "increasing"}
        },
        "per_symbol_metrics": [
            {
                "symbol": "NVDA",
                "target_horizon": 1,
                "mae": 0.013,
                "rmse": 0.019,
                "mape": 0.032,
                "smape": 0.038,
                "swase": 0.57,
                "directional_accuracy": 0.66,
                "peer_shock_flag_rate": 0.22
            },
            {
                "symbol": "AAPL",
                "target_horizon": 5,
                "mae": 0.021,
                "rmse": 0.028,
                "mape": 0.049,
                "smape": 0.055,
                "swase": 0.65,
                "directional_accuracy": 0.59
            }
        ],
        "regime_metrics": {
            "peer_shock_flag": {
                "0": {
                    "mae": 0.020,
                    "mape": 0.045,
                    "directional_accuracy": 0.64
                },
                "1": {
                    "mae": 0.028,
                    "mape": 0.061,
                    "directional_accuracy": 0.52
                }
            }
        },
        "feature_importance": {
            "overall": [
                ["peer_mean_ret_1d", 0.18],
                ["peer_shock_flag", 0.15],
                ["sector_index_ret_1d", 0.09],
                ["asset_vs_sector_ret_1d", 0.07]
            ],
            "shock_regime": [
                ["peer_shock_flag", 0.27],
                ["beta_vs_sector_60d", 0.21],
                ["sector_drawdown_60d", 0.16]
            ]
        },
        "guardrail_summary": {
            "status": "warning",
            "issues": [
                "MAPE above 0.08 for 3 symbols in shock regime",
                "Directional accuracy dropped >5% vs last week on 10d horizon"
            ]
        }
    }

    # Initialize the agent
    try:
        agent = LLMAnalyticsExplainerAgent()
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ ERROR initializing agent: {e}")
        return False

    # Test the new explain_metrics method
    try:
        print("Calling explain_metrics()...")
        result = agent.explain_metrics(metrics_payload)
        print("✅ explain_metrics() completed")

        # Validate the result structure
        required_keys = ["global_summary", "metric_explanations", "regime_insights",
                        "symbol_outliers", "feature_insights", "recommendations"]

        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            print(f"⚠️  Missing keys in result: {missing_keys}")
        else:
            print("✅ Result has all required keys")

        # Print a summary of the result
        print("\n--- Result Summary ---")
        print(f"Global Summary: {result.get('global_summary', 'N/A')[:100]}...")
        print(f"Regime Insights: {len(result.get('regime_insights', []))} items")
        print(f"Symbol Outliers: {len(result.get('symbol_outliers', []))} items")
        print(f"Recommendations: {len(result.get('recommendations', []))} items")

        # Pretty print the full result for inspection
        print("\n--- Full Result ---")
        print(json.dumps(result, indent=2)[:1000] + "..." if len(json.dumps(result)) > 1000 else json.dumps(result, indent=2))

    except Exception as e:
        print(f"❌ ERROR in explain_metrics(): {e}")
        return False

    print("\n🎉 LLMAnalyticsExplainerAgent test completed successfully!")
    print("\nNext steps:")
    print("1. Check LangSmith UI for traces of this test run")
    print("2. Integrate into main.py workflow")
    print("3. Test with real metrics from your forecasting pipeline")

    return True


if __name__ == "__main__":
    success = test_analytics_explainer()
    sys.exit(0 if success else 1)