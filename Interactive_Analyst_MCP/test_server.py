#!/usr/bin/env python3
"""
Enhanced Test script for the Interactive Analyst MCP Server

This script demonstrates the advanced natural language processing capabilities
of the enhanced MCP server including compound queries, context awareness, and suggestions.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

def test_mcp_tool(tool_name: str, arguments: dict, query_id: int = 1) -> dict:
    """Test any MCP tool with given arguments."""
    # Create JSON-RPC request
    request = {
        "jsonrpc": "2.0",
        "id": query_id,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    # Convert to JSON string
    request_json = json.dumps(request)

    # Run the MCP server
    server_path = Path(__file__).parent / "server.py"
    try:
        result = subprocess.run(
            [sys.executable, str(server_path)],
            input=request_json,
            text=True,
            capture_output=True,
            timeout=60
        )

        if result.returncode == 0:
            response = json.loads(result.stdout.strip())
            return response
        else:
            return {"error": result.stderr.strip()}

    except subprocess.TimeoutExpired:
        return {"error": "Server timeout"}
    except Exception as e:
        return {"error": str(e)}

def test_analyze_performance(query: str, session_id: str = "test_session", query_id: int = 1) -> None:
    """Test the analyze_forecast_performance tool."""
    print(f"\n=== Test {query_id}: Analyze Performance ===")
    print(f"Query: {query}")
    print(f"Session: {session_id}")
    print("-" * 50)

    response = test_mcp_tool("analyze_forecast_performance", {
        "query": query,
        "session_id": session_id
    }, query_id)

    if "error" in response:
        print(f"❌ Error: {response['error']}")
    elif "result" in response and "content" in response["result"]:
        content = response["result"]["content"][0]["text"]
        # Print first 800 characters to avoid overwhelming output
        print(content[:800] + ("..." if len(content) > 800 else ""))
    else:
        print("❌ Unexpected response format")

def test_query_suggestions(partial_query: str, query_id: int = 1) -> None:
    """Test the get_query_suggestions tool."""
    print(f"\n=== Test {query_id}: Query Suggestions ===")
    print(f"Partial Query: '{partial_query}'")
    print("-" * 30)

    response = test_mcp_tool("get_query_suggestions", {
        "partial_query": partial_query
    }, query_id)

    if "error" in response:
        print(f"❌ Error: {response['error']}")
    elif "result" in response and "content" in response["result"]:
        content = response["result"]["content"][0]["text"]
        print(content)
    else:
        print("❌ Unexpected response format")

def test_conversation_context(session_id: str = "test_session", query_id: int = 1) -> None:
    """Test the get_conversation_context tool."""
    print(f"\n=== Test {query_id}: Conversation Context ===")
    print(f"Session: {session_id}")
    print("-" * 30)

    response = test_mcp_tool("get_conversation_context", {
        "session_id": session_id
    }, query_id)

    if "error" in response:
        print(f"❌ Error: {response['error']}")
    elif "result" in response and "content" in response["result"]:
        content = response["result"]["content"][0]["text"]
        print(content)
    else:
        print("❌ Unexpected response format")

def main():
    """Run comprehensive tests of the enhanced MCP server."""
    print("🚀 Enhanced Interactive Analyst MCP Server - Comprehensive Test Suite")
    print("=" * 80)

    # Test 1: Basic performance analysis
    test_analyze_performance("Show me the weakest performing buckets", "session1", 1)

    # Test 2: Compound query
    test_analyze_performance("Show me the weakest buckets and give me a summary", "session1", 2)

    # Test 3: Bucket-specific analysis
    test_analyze_performance("Analyze ai_basket performance over 30 days", "session1", 3)

    # Test 4: Complex compound query
    test_analyze_performance("What are the top HPO candidates and check for guardrail violations", "session1", 4)

    # Test 5: Context-aware query (should remember ai_basket preference)
    test_analyze_performance("Show me the performance summary", "session1", 5)

    # Test 6: New session (no context)
    test_analyze_performance("Plot the residuals for crypto_exposed", "session2", 6)

    # Test 7: Query suggestions
    test_query_suggestions("weak", 7)
    test_query_suggestions("plot", 8)
    test_query_suggestions("summary", 9)

    # Test 8: Conversation context
    test_conversation_context("session1", 10)
    test_conversation_context("session2", 11)

    # Test 9: Advanced queries
    test_analyze_performance("Compare ai_basket and defensive bucket performance", "session3", 12)
    test_analyze_performance("Show me trends over the last week", "session3", 13)

    # Test 10: Export functionality
    test_analyze_performance("Export the analysis to weekly_report.md", "session3", 14)

    print("\n" + "=" * 80)
    print("✅ Enhanced MCP Server Test Suite Complete!")
    print("🎯 Features Tested:")
    print("  • Natural Language Processing with confidence scores")
    print("  • Compound query detection and processing")
    print("  • Context awareness and conversation memory")
    print("  • Intelligent entity extraction")
    print("  • Query suggestions and auto-completion")
    print("  • Caching and performance optimization")
    print("  • Enhanced error handling and validation")
    print("  • Multiple analysis types (summary, weakest, bucket, guardrails, HPO, plots, export)")

if __name__ == "__main__":
    main()