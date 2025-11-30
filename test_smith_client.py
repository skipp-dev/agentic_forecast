#!/usr/bin/env python3
"""
Test script for SmithLLMClient

This script tests the SmithLLMClient with the EU endpoint configuration.
Run this to verify that the client can connect and make API calls.

Usage:
    python test_smith_client.py
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.smith_client import SmithLLMClient


def test_smith_client():
    """Test the SmithLLMClient with a simple completion."""

    print("=== Testing SmithLLMClient ===")
    print()

    # Check environment variables
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")

    print(f"LANGSMITH_ENDPOINT: {endpoint}")
    print(f"LANGSMITH_API_KEY: {'***' + api_key[-4:] if api_key else 'NOT SET'}")
    print()

    if not endpoint:
        print("❌ ERROR: LANGSMITH_ENDPOINT not set. Set it to https://eu.smith.langchain.com")
        return False

    if not api_key:
        print("❌ ERROR: LANGSMITH_API_KEY not set. Get your API key from LangSmith UI.")
        return False

    # Test client initialization
    try:
        client = SmithLLMClient(model="o4-mini")  # Use model from config
        print("✅ Client initialized successfully")
    except Exception as e:
        print(f"❌ ERROR initializing client: {e}")
        return False

    # Test health check
    try:
        health = client.health_check()
        print(f"Health check: {health['status']}")
        if health['status'] != 'healthy':
            print(f"❌ Health check failed: {health.get('error', 'Unknown error')}")
            return False
        print("✅ Health check passed")
    except Exception as e:
        print(f"❌ ERROR in health check: {e}")
        return False

    # Test simple completion
    try:
        prompt = "Say 'Hello from SmithLLMClient test!' and nothing else."
        response = client.complete(prompt, max_tokens=20)
        print(f"Test completion response: {response}")
        if "Hello from SmithLLMClient test" in response:
            print("✅ Completion test passed")
        else:
            print("⚠️  Completion test: unexpected response format")
    except Exception as e:
        print(f"❌ ERROR in completion test: {e}")
        return False

    print()
    print("🎉 All SmithLLMClient tests passed!")
    return True


if __name__ == "__main__":
    success = test_smith_client()
    sys.exit(0 if success else 1)