# MCP Client Configuration for Interactive Analyst

This directory contains configuration and examples for integrating
the Interactive Analyst MCP Server with various MCP clients.

## Supported MCP Clients

### 1. Claude Desktop (Anthropic)
### 2. VS Code Extension
### 3. Custom MCP Client

## Configuration Files

### Claude Desktop Configuration

Add this to your Claude Desktop configuration file:

```json
{
  "mcpServers": {
    "interactive-analyst": {
      "command": "python",
      "args": ["/path/to/Interactive_Analyst_MCP/server.py"],
      "cwd": "/path/to/Interactive_Analyst_MCP",
      "env": {
        "PYTHONPATH": "/path/to/Interactive_Analyst_MCP"
      }
    }
  }
}
```

### VS Code MCP Extension Configuration

For VS Code MCP extensions, use:

```json
{
  "mcp.server.interactive-analyst": {
    "command": "python",
    "args": ["server.py"],
    "cwd": "${workspaceFolder}/Interactive_Analyst_MCP"
  }
}
```

## Usage Examples

### Basic Query
```
"Show me the weakest performing buckets"
```

### Compound Query
```
"Show me the weakest buckets and plot the residuals"
```

### Context-Aware Query
```
"Analyze that bucket over 30 days" (remembers previous bucket)
```

### Advanced Analysis
```
"What are the top HPO candidates and check for guardrail violations?"
```

## Features Demonstrated

- ✅ Natural Language Processing
- ✅ Intent Classification with Confidence Scores
- ✅ Entity Extraction (buckets, horizons, plot types, etc.)
- ✅ Compound Query Detection
- ✅ Context Awareness and Memory
- ✅ Query Suggestions
- ✅ Conversation History
- ✅ Caching for Performance
- ✅ Enhanced Error Handling

## Testing

Run the comprehensive test suite:

```bash
cd Interactive_Analyst_MCP
python test_server.py
```

## Integration Benefits

1. **User-Friendly**: No need to learn command syntax
2. **Conversational**: Supports natural follow-up queries
3. **Contextual**: Remembers preferences and previous queries
4. **Comprehensive**: Handles complex multi-part requests
5. **Intelligent**: Provides suggestions and auto-completion
6. **Performant**: Caches results and optimizes execution