#!/usr/bin/env python3
"""
Test script for LangSmith tracing with SmithLLMClient

This script tests that LangSmith tracing is working correctly with the
SmithLLMClient. It makes a few API calls and checks that traces appear
in the LangSmith UI.

Usage:
    python test_langsmith_trace.py
"""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.smith_client import SmithLLMClient


def test_langsmith_tracing():
    """Test LangSmith tracing with SmithLLMClient."""

    print("=== Testing LangSmith Tracing ===")
    print()

    # Check tracing environment variables
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "default")

    print(f"LANGCHAIN_TRACING_V2: {tracing_enabled}")
    print(f"LANGSMITH_ENDPOINT: {endpoint}")
    print(f"LANGSMITH_PROJECT: {project}")
    print(f"LANGSMITH_API_KEY: {'***' + api_key[-4:] if api_key else 'NOT SET'}")
    print()

    if not tracing_enabled:
        print("⚠️  WARNING: Tracing is not enabled. Set LANGCHAIN_TRACING_V2=true")
        print("    Tracing tests will still run but won't be visible in LangSmith UI.")
        print()

    if not endpoint:
        print("❌ ERROR: LANGSMITH_ENDPOINT not set. Set it to https://eu.smith.langchain.com")
        return False

    if not api_key:
        print("❌ ERROR: LANGSMITH_API_KEY not set. Get your API key from LangSmith UI.")
        return False

    # Initialize client
    try:
        client = SmithLLMClient(model="o4-mini")
        print("✅ Client initialized successfully")
    except Exception as e:
        print(f"❌ ERROR initializing client: {e}")
        return False

    # Make several test calls to generate traces
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
    ]

    print("Making test API calls to generate traces...")
    print()

    for i, prompt in enumerate(test_prompts, 1):
        try:
            print(f"Test call {i}: {prompt[:50]}...")
            start_time = time.time()
            response = client.complete(prompt, max_tokens=100)
            duration = time.time() - start_time
            print(f"Duration: {duration:.1f}s")
            print(f"Response: {response[:100]}{'...' if len(response) > 100 else ''}")
            print()
        except Exception as e:
            print(f"❌ ERROR in test call {i}: {e}")
            return False

        # Small delay between calls
        time.sleep(1)

    print("🎉 All tracing test calls completed!")
    print()
    print("Next steps:")
    print("1. Go to your LangSmith UI: https://eu.smith.langchain.com")
    print("2. Check the 'agentic_forecast' project (or your configured project)")
    print("3. Look for recent traces from these test calls")
    print("4. Verify that the model calls are being traced correctly")
    print()
    print("If you don't see traces:")
    print("- Make sure LANGCHAIN_TRACING_V2=true")
    print("- Check that your API key has tracing permissions")
    print("- Verify the project name matches your LangSmith setup")

    return True


if __name__ == "__main__":
    success = test_langsmith_tracing()
    sys.exit(0 if success else 1)