#!/usr/bin/env python3
"""
Simple test for SmithLLMClient tracing
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.smith_client import SmithLLMClient

def test_tracing():
    """Test if tracing is working."""

    print("=== Testing SmithLLMClient Tracing ===")

    # Check if langsmith is available
    try:
        import langsmith
        print("✅ LangSmith library available")
    except ImportError:
        print("❌ LangSmith library not installed")
        return

    # Check environment
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")

    print(f"Tracing enabled: {tracing_enabled}")
    print(f"Endpoint: {endpoint}")
    print(f"API key set: {'Yes' if api_key else 'No'}")

    if not api_key:
        print("❌ No API key - skipping actual test")
        return

    try:
        client = SmithLLMClient(model="o4-mini")
        print("✅ Client created")

        # Try a simple call
        result = client.complete("Say 'tracing test' and nothing else.", max_tokens=10)
        print(f"✅ API call successful: {result}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_tracing()