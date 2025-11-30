#!/usr/bin/env python3
"""
Test script for OpenAI LLM with LangSmith tracing integration.
Tests the LLMAnalyticsExplainerAgent with the new OpenAILLMClient and tracing.
"""
import os
import sys
import json
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars

# Add src and agents to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # For agents directory

def test_openai_llm_factory():
    """Test that the LLM factory creates OpenAI client correctly."""
    print("=== Testing LLM Factory ===")

    try:
        from src.llm.llm_factory import create_llm_for_role
        llm = create_llm_for_role("analytics_explainer")
        print(f"✅ Created LLM client: {type(llm).__name__}")
        print(f"   Model: {llm.model}")
        return True
    except Exception as e:
        print(f"❌ Failed to create LLM client: {e}")
        return False

def test_openai_llm_call():
    """Test a simple LLM call to verify OpenAI integration."""
    print("\n=== Testing OpenAI LLM Call ===")

    try:
        from src.llm.llm_factory import create_llm_for_role
        llm = create_llm_for_role("analytics_explainer")

        # Simple test prompt
        response = llm.complete(
            prompt="Explain what MAPE means in forecasting. Keep it brief.",
            system="You are a helpful AI assistant.",
            temperature=0.1,
            max_tokens=100
        )

        print("✅ OpenAI LLM call successful")
        print(f"   Response length: {len(response)} characters")
        print(f"   Response preview: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ OpenAI LLM call failed: {e}")
        return False

def test_analytics_explainer_agent():
    """Test the LLMAnalyticsExplainerAgent with tracing."""
    print("\n=== Testing LLMAnalyticsExplainerAgent ===")

    try:
        from agents.llm_analytics_agent import LLMAnalyticsExplainerAgent

        # Create agent
        agent = LLMAnalyticsExplainerAgent()
        print(f"✅ Created agent with LLM: {type(agent.llm).__name__}")

        # Create minimal test payload
        test_payload = {
            "run_metadata": {
                "run_type": "TEST",
                "date": "2024-01-01",
                "universe_size": 2,
                "cross_asset_v2_enabled": False
            },
            "metrics_global": {
                "mape": {"mean": 0.15, "trend": "improving"},
                "mae": {"mean": 0.12, "trend": "stable"},
                "directional_accuracy": {"mean": 0.65, "trend": "improving"}
            },
            "per_symbol_metrics": [
                {"symbol": "AAPL", "target_horizon": 1, "mae": 0.10, "mape": 0.12, "directional_accuracy": 0.70},
                {"symbol": "MSFT", "target_horizon": 1, "mae": 0.14, "mape": 0.18, "directional_accuracy": 0.60}
            ],
            "regime_metrics": {"peer_shock_flag": {}},
            "feature_importance": {"overall": [], "shock_regime": []},
            "guardrail_summary": {"status": "healthy", "issues": []}
        }

        # Call explain_metrics (this should be traced)
        result = agent.explain_metrics(test_payload)

        print("✅ Agent explain_metrics call successful")
        print(f"   Result keys: {list(result.keys())}")
        print(f"   Global summary: {result.get('global_summary', 'N/A')[:100]}...")

        return True
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing OpenAI LLM + LangSmith Tracing Integration")
    print("=" * 60)

    # Check environment variables
    required_env_vars = ["OPENAI_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_PROJECT"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"⚠️  Missing environment variables: {missing_vars}")
        print("   Please set them before running tests.")
        return False

    # Run tests
    tests = [
        test_openai_llm_factory,
        test_openai_llm_call,
        test_analytics_explainer_agent
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! OpenAI + LangSmith integration is working.")
        print("   Check LangSmith dashboard for traces from the analytics explainer.")
    else:
        print("❌ Some tests failed. Check the output above for details.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)