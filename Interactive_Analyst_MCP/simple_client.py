#!/usr/bin/env python3
"""
Simple MCP Client Example for Interactive Analyst

This script demonstrates how to interact with the Interactive Analyst MCP Server
using a simple Python client. It shows how to make tool calls and handle responses.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

class SimpleMCPClient:
    """Simple MCP client for testing the Interactive Analyst server."""

    def __init__(self, server_path: Path):
        self.server_path = server_path
        self.next_id = 1

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool and return the response."""
        request = {
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        self.next_id += 1
        request_json = json.dumps(request)

        try:
            result = subprocess.run(
                [sys.executable, str(self.server_path)],
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
            return {"error": "Request timeout"}
        except Exception as e:
            return {"error": str(e)}

    def list_tools(self) -> Dict[str, Any]:
        """List available MCP tools."""
        request = {
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": "tools/list"
        }

        self.next_id += 1
        request_json = json.dumps(request)

        try:
            result = subprocess.run(
                [sys.executable, str(self.server_path)],
                input=request_json,
                text=True,
                capture_output=True,
                timeout=30
            )

            if result.returncode == 0:
                response = json.loads(result.stdout.strip())
                return response
            else:
                return {"error": result.stderr.strip()}

        except Exception as e:
            return {"error": str(e)}

def demo_interactive_session():
    """Demonstrate an interactive session with the MCP server."""
    server_path = Path(__file__).parent / "server.py"
    client = SimpleMCPClient(server_path)

    print("🤖 Interactive Analyst MCP Client Demo")
    print("=" * 50)

    # 1. List available tools
    print("\n1. Available Tools:")
    tools_response = client.list_tools()
    if "result" in tools_response and "tools" in tools_response["result"]:
        for tool in tools_response["result"]["tools"]:
            print(f"   • {tool['name']}: {tool['description'][:80]}...")
    else:
        print("   ❌ Failed to list tools")

    # 2. Test query suggestions
    print("\n2. Query Suggestions for 'weak':")
    suggestions = client.call_tool("get_query_suggestions", {"partial_query": "weak"})
    if "result" in suggestions and "content" in suggestions["result"]:
        print(suggestions["result"]["content"][0]["text"])
    else:
        print("   ❌ Failed to get suggestions")

    # 3. Analyze performance with natural language
    print("\n3. Natural Language Analysis:")
    analysis = client.call_tool("analyze_forecast_performance", {
        "query": "Show me the weakest performing buckets",
        "session_id": "demo_session"
    })
    if "result" in analysis and "content" in analysis["result"]:
        content = analysis["result"]["content"][0]["text"]
        print(content[:500] + ("..." if len(content) > 500 else ""))
    else:
        print("   ❌ Failed to analyze performance")

    # 4. Test compound query
    print("\n4. Compound Query Analysis:")
    compound = client.call_tool("analyze_forecast_performance", {
        "query": "Show me the weakest buckets and give me a summary",
        "session_id": "demo_session"
    })
    if "result" in compound and "content" in compound["result"]:
        content = compound["result"]["content"][0]["text"]
        print(content[:500] + ("..." if len(content) > 500 else ""))
    else:
        print("   ❌ Failed compound query")

    # 5. Check conversation context
    print("\n5. Conversation Context:")
    context = client.call_tool("get_conversation_context", {"session_id": "demo_session"})
    if "result" in context and "content" in context["result"]:
        print(context["result"]["content"][0]["text"])
    else:
        print("   ❌ Failed to get context")

    print("\n" + "=" * 50)
    print("✅ MCP Client Demo Complete!")
    print("\n💡 Key Features Demonstrated:")
    print("   • Tool discovery and listing")
    print("   • Query suggestions and auto-completion")
    print("   • Natural language performance analysis")
    print("   • Compound query processing")
    print("   • Context awareness and conversation memory")

if __name__ == "__main__":
    demo_interactive_session()