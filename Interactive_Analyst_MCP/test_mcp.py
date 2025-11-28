#!/usr/bin/env python3
"""Test script for MCP server initialization"""

import subprocess
import json
import sys
import time

def test_mcp_server():
    # Start the server process
    server_process = subprocess.Popen(
        [sys.executable, 'server.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=r'C:\Users\spreu\Documents\agentic_forecast\Interactive_Analyst_MCP'
    )

    try:
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0"
                }
            }
        }

        print("Sending initialize request...")
        server_process.stdin.write(json.dumps(init_request) + '\n')
        server_process.stdin.flush()

        # Read response
        response_line = server_process.stdout.readline().strip()
        print(f"Server response: {response_line}")

        if response_line:
            try:
                response = json.loads(response_line)
                if 'result' in response and 'serverInfo' in response['result']:
                    print("✅ MCP initialization successful!")
                    print(f"Server: {response['result']['serverInfo']}")
                    return True
                else:
                    print("❌ Invalid MCP response format")
                    return False
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON response: {e}")
                return False
        else:
            print("❌ No response from server")
            return False

    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False
    finally:
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    success = test_mcp_server()
    sys.exit(0 if success else 1)