#!/usr/bin/env python3
"""
Test script for LangSmith LLM backend integration.

This script tests the new LLM factory and Smith client integration.
"""

import os
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_llm_backends():
    """Test LLM backend configuration and factory."""
    print("🧪 Testing LLM Backends Configuration")
    print("=" * 50)

    try:
        from src.llm.llm_factory import (
            load_config, get_llm_backend, create_llm_for_role,
            test_llm_backends
        )

        # Run the built-in test
        test_llm_backends()

    except Exception as e:
        print(f"❌ LLM factory test failed: {e}")
        return False

    return True

def test_smith_client():
    """Test Smith LLM client initialization."""
    print("\n🔧 Testing Smith LLM Client")
    print("-" * 30)

    try:
        from src.llm.smith_client import SmithLLMClient

        # Test with a dummy model name (won't actually call API)
        client = SmithLLMClient(model="test-model")
        print("✅ SmithLLMClient initialized successfully")

        # Test health check (will fail without real API key, but should not crash)
        try:
            health = client.health_check()
            print(f"✅ Health check completed: {health['status']}")
        except Exception as e:
            print(f"⚠️  Health check failed (expected without API key): {e}")

        return True

    except Exception as e:
        print(f"❌ Smith client test failed: {e}")
        return False

def test_local_client():
    """Test local LM Studio client."""
    print("\n🏠 Testing Local LM Studio Client")
    print("-" * 35)

    try:
        from src.llm.local_client import LocalLlamaClient

        # Test initialization (won't actually connect)
        client = LocalLlamaClient(
            base_url="http://127.0.0.1:1234/v1",
            model="test-model"
        )
        print("✅ LocalLlamaClient initialized successfully")

        # Test health check (will fail without running server, but should not crash)
        try:
            health = client.health_check()
            print(f"✅ Health check completed: {health['status']}")
        except Exception as e:
            print(f"⚠️  Health check failed (expected without server): {e}")

        return True

    except Exception as e:
        print(f"❌ Local client test failed: {e}")
        return False

def test_openai_client():
    """Test OpenAI client compatibility."""
    print("\n🔑 Testing OpenAI Client")
    print("-" * 25)

    try:
        from src.llm.client import LLMClient

        # Test initialization (won't actually call API without key)
        client = LLMClient(model="gpt-4o")
        print("✅ LLMClient initialized successfully")

        # Test that it has the complete method
        assert hasattr(client, 'complete'), "LLMClient missing complete method"
        print("✅ LLMClient has complete method for protocol compliance")

        return True

    except Exception as e:
        print(f"❌ OpenAI client test failed: {e}")
        return False

def test_role_creation():
    """Test creating LLMs for different roles."""
    print("\n🎭 Testing Role-Based LLM Creation")
    print("-" * 35)

    try:
        from src.llm.llm_factory import create_analytics_explainer_llm

        # Test creating an analytics explainer LLM
        llm = create_analytics_explainer_llm()
        print(f"✅ Created analytics explainer LLM: {type(llm).__name__}")

        # Test that it implements the protocol
        assert hasattr(llm, 'complete'), "LLM missing complete method"
        assert hasattr(llm, 'generate'), "LLM missing generate method"
        print("✅ LLM implements required protocol methods")

        return True

    except Exception as e:
        print(f"❌ Role creation test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\n⚙️  Testing Configuration Loading")
    print("-" * 32)

    try:
        from src.llm.llm_factory import load_config

        config = load_config()

        # Check for required sections
        assert 'llm_backends' in config, "Missing llm_backends in config"
        assert 'default_llm_roles' in config, "Missing default_llm_roles in config"
        print("✅ Configuration loaded with required sections")

        # Check that Smith backends are configured
        backends = config['llm_backends']
        smith_backends = [k for k, v in backends.items() if v.get('provider') == 'smith']
        assert len(smith_backends) > 0, "No Smith backends configured"
        print(f"✅ Found {len(smith_backends)} Smith backends: {smith_backends}")

        return True

    except Exception as e:
        print(f"❌ Config loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 LangSmith LLM Backend Integration Test Suite")
    print("=" * 55)

    tests = [
        ("Configuration Loading", test_config_loading),
        ("LLM Backends", test_llm_backends),
        ("Smith Client", test_smith_client),
        ("Local Client", test_local_client),
        ("OpenAI Client", test_openai_client),
        ("Role Creation", test_role_creation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print("\n" + "=" * 55)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! LangSmith LLM backend integration is ready.")
        print("\nNext steps:")
        print("1. Configure your LangSmith API key: LANGSMITH_API_KEY")
        print("2. Set up model aliases in LangSmith Models/Usage page")
        print("3. Test with a real BACKTEST run")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)