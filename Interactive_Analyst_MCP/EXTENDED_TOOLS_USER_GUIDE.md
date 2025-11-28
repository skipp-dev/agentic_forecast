# Interactive Analyst MCP Server - Extended Tools Documentation

## Overview

The Interactive Analyst MCP Server has been extended with 8 powerful new tools that provide advanced analytics capabilities, automation features, and enhanced user experience. These tools enable comprehensive report generation, automated monitoring, batch processing, and personalized analytics.

## New Tools Summary

| Tool | Purpose | Key Features |
|------|---------|--------------|
| `export_analysis_report` | Generate comprehensive reports | Multi-format support (Markdown, JSON, HTML), chart embedding |
| `schedule_recurring_analysis` | Automate periodic analysis | Cron-like scheduling, email notifications |
| `create_performance_alert` | Monitor performance metrics | Configurable thresholds, multiple comparison operators |
| `process_batch_queries` | Execute multiple queries concurrently | Error handling, progress tracking, concurrency control |
| `set_user_preference` | Personalize user experience | Customizable settings, per-user preferences |
| `get_query_history` | Access historical queries | Pattern analysis, query trends, database persistence |
| `get_dashboard_data` | Provide structured dashboard data | Multiple dashboard types, time-range filtering |
| `compare_models` | Side-by-side model comparison | Statistical rankings, performance metrics analysis |

## Tool Details

### 1. export_analysis_report

**Purpose**: Generate comprehensive analysis reports in multiple formats with optional chart embedding.

**What it provides**:
- Multi-format report generation (Markdown, JSON, HTML)
- Chart embedding capabilities for visual reports
- Automatic file saving with timestamped filenames
- Database persistence of report metadata

**Parameters**:
- `report_type` (string): Type of report - "summary", "detailed", "performance", "comparison"
- `format` (string): Output format - "markdown", "json", "html" (default: "markdown")
- `include_charts` (boolean): Whether to include charts (default: false)

**Example 1**: Generate a summary report in Markdown format
```json
{
  "report_type": "summary",
  "format": "markdown",
  "include_charts": true
}
```
**Output**: Creates `report_20241201_143022.md` with comprehensive analysis summary including embedded charts.

**Example 2**: Generate a performance comparison report in HTML
```json
{
  "report_type": "comparison",
  "format": "html",
  "include_charts": false
}
```
**Output**: Creates `report_20241201_143023.html` with detailed model comparison data.

**Example 3**: Generate a detailed analysis report in JSON
```json
{
  "report_type": "detailed",
  "format": "json",
  "include_charts": true
}
```
**Output**: Creates `report_20241201_143024.json` with structured analysis data and chart specifications.

### 2. schedule_recurring_analysis

**Purpose**: Schedule recurring analysis queries with customizable intervals and optional email notifications.

**What it provides**:
- Automated periodic execution of analysis queries
- Flexible scheduling (hourly, daily, weekly, monthly)
- Email notification support for results
- Persistent scheduling with run tracking

**Parameters**:
- `query` (string): Natural language query to schedule
- `schedule` (string): Schedule type - "hourly", "daily", "weekly", "monthly"
- `email` (string, optional): Email address for notifications
- `user_id` (string): User identifier (default: "default")

**Example 1**: Schedule daily performance monitoring
```json
{
  "query": "Check MSE performance for all buckets",
  "schedule": "daily",
  "email": "analyst@company.com",
  "user_id": "analyst_team"
}
```
**Output**: Schedules daily execution with email notifications to analyst@company.com.

**Example 2**: Schedule weekly model comparison analysis
```json
{
  "query": "Compare all active models on MAPE metric",
  "schedule": "weekly",
  "user_id": "model_team"
}
```
**Output**: Schedules weekly model comparison without email notifications.

**Example 3**: Schedule hourly alert monitoring
```json
{
  "query": "Check for any triggered performance alerts",
  "schedule": "hourly",
  "email": "alerts@company.com",
  "user_id": "monitoring"
}
```
**Output**: Schedules hourly alert checks with notifications.

### 3. create_performance_alert

**Purpose**: Create performance alerts that trigger when metrics exceed configurable thresholds.

**What it provides**:
- Real-time performance monitoring
- Flexible threshold configuration
- Multiple comparison operators (greater, less, equal)
- Alert history and trigger counting

**Parameters**:
- `alert_name` (string): Descriptive name for the alert
- `metric` (string): Metric to monitor (e.g., "mse", "mape", "accuracy")
- `threshold` (number): Threshold value for triggering
- `comparison` (string): Comparison operator - "greater", "less", "equal" (default: "greater")
- `user_id` (string): User identifier (default: "default")

**Example 1**: Alert when MSE exceeds 0.05
```json
{
  "alert_name": "High MSE Alert",
  "metric": "mse",
  "threshold": 0.05,
  "comparison": "greater",
  "user_id": "production_monitoring"
}
```
**Output**: Creates alert that triggers when MSE > 0.05.

**Example 2**: Alert when accuracy drops below 0.90
```json
{
  "alert_name": "Low Accuracy Warning",
  "metric": "accuracy",
  "threshold": 0.90,
  "comparison": "less",
  "user_id": "quality_assurance"
}
```
**Output**: Creates alert that triggers when accuracy < 0.90.

**Example 3**: Alert when MAPE equals exactly 0.10
```json
{
  "alert_name": "MAPE Threshold Reached",
  "metric": "mape",
  "threshold": 0.10,
  "comparison": "equal",
  "user_id": "analytics_team"
}
```
**Output**: Creates alert that triggers when MAPE = 0.10.

### 4. process_batch_queries

**Purpose**: Execute multiple analysis queries concurrently with error handling and progress tracking.

**What it provides**:
- Concurrent query execution for improved performance
- Comprehensive error handling for individual queries
- Progress tracking and result aggregation
- Configurable concurrency limits

**Parameters**:
- `queries` (array): List of natural language queries to execute
- `max_concurrent` (integer): Maximum concurrent executions (1-10, default: 3)
- `user_id` (string): User identifier (default: "default")

**Example 1**: Batch process multiple bucket analyses
```json
{
  "queries": [
    "Analyze performance of ai_basket bucket",
    "Show residuals for tech_stocks bucket",
    "Compare mse for all buckets"
  ],
  "max_concurrent": 3,
  "user_id": "batch_analyst"
}
```
**Output**: Executes 3 queries concurrently, returns results with success/failure status for each.

**Example 2**: Large batch with error handling
```json
{
  "queries": [
    "Get summary statistics",
    "Plot performance trends",
    "Invalid query that will fail",
    "Show top performing models",
    "Generate forecast accuracy report"
  ],
  "max_concurrent": 2,
  "user_id": "comprehensive_analysis"
}
```
**Output**: Processes 5 queries with 2 concurrent executions, handles errors gracefully.

**Example 3**: Single query batch for consistency
```json
{
  "queries": ["Analyze all buckets performance over 30 days"],
  "max_concurrent": 1,
  "user_id": "single_analysis"
}
```
**Output**: Executes single query with batch processing framework.

### 5. set_user_preference

**Purpose**: Set user preferences for personalized analysis experience and default behaviors.

**What it provides**:
- Personalized user settings storage
- Customizable default behaviors
- Per-user preference isolation
- Timestamp tracking for preference changes

**Parameters**:
- `preference_key` (string): Preference identifier (e.g., "default_horizon", "favorite_buckets")
- `preference_value` (any): Preference value to store
- `user_id` (string): User identifier (default: "default")

**Example 1**: Set default forecast horizon
```json
{
  "preference_key": "default_horizon",
  "preference_value": "30d",
  "user_id": "analyst_john"
}
```
**Output**: Sets John's default forecast horizon to 30 days.

**Example 2**: Set favorite buckets preference
```json
{
  "preference_key": "favorite_buckets",
  "preference_value": ["ai_basket", "tech_stocks", "energy"],
  "user_id": "portfolio_manager"
}
```
**Output**: Sets portfolio manager's favorite buckets for quick access.

**Example 3**: Set notification preferences
```json
{
  "preference_key": "email_notifications",
  "preference_value": true,
  "user_id": "alert_recipient"
}
```
**Output**: Enables email notifications for alert recipient.

### 6. get_query_history

**Purpose**: Retrieve historical query data for analysis, pattern recognition, and audit trails.

**What it provides**:
- Historical query retrieval with metadata
- Pattern analysis capabilities
- Database-backed persistence
- Configurable result limits

**Parameters**:
- `limit` (integer): Maximum queries to return (1-100, default: 10)
- `user_id` (string): User identifier (default: "default")

**Example 1**: Get recent query history
```json
{
  "limit": 5,
  "user_id": "analyst_sarah"
}
```
**Output**: Returns Sarah's last 5 queries with timestamps and results.

**Example 2**: Get extensive history for pattern analysis
```json
{
  "limit": 50,
  "user_id": "research_team"
}
```
**Output**: Returns research team's last 50 queries for pattern analysis.

**Example 3**: Get default user history
```json
{
  "limit": 10,
  "user_id": "default"
}
```
**Output**: Returns default user's query history.

### 7. get_dashboard_data

**Purpose**: Provide structured data for web dashboards and visualization tools.

**What it provides**:
- Multiple dashboard types (overview, performance, alerts)
- Time-range filtering capabilities
- Structured JSON data for frontend consumption
- Real-time metrics aggregation

**Parameters**:
- `dashboard_type` (string): Dashboard type - "overview", "performance", "alerts" (default: "overview")
- `time_range` (string): Time range - "1d", "7d", "30d", "90d" (default: "7d")
- `user_id` (string): User identifier (default: "default")

**Example 1**: Get overview dashboard data
```json
{
  "dashboard_type": "overview",
  "time_range": "7d",
  "user_id": "dashboard_user"
}
```
**Output**: Returns overview data including active alerts, scheduled queries, and key metrics.

**Example 2**: Get performance dashboard for 30 days
```json
{
  "dashboard_type": "performance",
  "time_range": "30d",
  "user_id": "performance_monitor"
}
```
**Output**: Returns performance metrics, trends, and top/bottom performers.

**Example 3**: Get alerts dashboard for recent activity
```json
{
  "dashboard_type": "alerts",
  "time_range": "1d",
  "user_id": "alert_manager"
}
```
**Output**: Returns active alerts, recent triggers, and alert history.

### 8. compare_models

**Purpose**: Compare multiple models side by side with statistical rankings and performance analysis.

**What it provides**:
- Side-by-side model performance comparison
- Statistical rankings across multiple metrics
- Overall score calculations
- Best model identification

**Parameters**:
- `models` (array): List of model names to compare
- `metrics` (array): Metrics to compare (default: ["mse", "mape", "accuracy"])
- `time_range` (string): Time range for comparison (default: "30d")
- `user_id` (string): User identifier (default: "default")

**Example 1**: Compare top 3 models on default metrics
```json
{
  "models": ["xgboost_v2", "lstm_v1", "transformer_v3"],
  "user_id": "model_comparison_team"
}
```
**Output**: Compares 3 models on MSE, MAPE, and accuracy with rankings.

**Example 2**: Compare models on specific metrics
```json
{
  "models": ["model_a", "model_b", "model_c", "model_d"],
  "metrics": ["mse", "mape", "r2_score"],
  "time_range": "90d",
  "user_id": "comprehensive_evaluation"
}
```
**Output**: Compares 4 models on 3 specific metrics over 90 days.

**Example 3**: Single model comparison (for consistency)
```json
{
  "models": ["champion_model"],
  "metrics": ["accuracy", "precision", "recall"],
  "user_id": "validation_team"
}
```
**Output**: Provides baseline comparison data for single model evaluation.

## Integration Examples

### Claude Desktop Integration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "interactive-analyst": {
      "command": "python",
      "args": ["/path/to/Interactive_Analyst_MCP/server.py"],
      "env": {
        "PYTHONPATH": "/path/to/your/project"
      }
    }
  }
}
```

### Programmatic Usage

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_extended_tools():
    server_params = StdioServerParameters(
        command="python",
        args=["/path/to/server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Export a comprehensive report
            result = await session.call_tool(
                "export_analysis_report",
                {"report_type": "performance", "format": "html", "include_charts": True}
            )

            # Schedule recurring analysis
            result = await session.call_tool(
                "schedule_recurring_analysis",
                {
                    "query": "Monitor all bucket performance",
                    "schedule": "daily",
                    "email": "team@company.com"
                }
            )

            # Create performance alert
            result = await session.call_tool(
                "create_performance_alert",
                {
                    "alert_name": "Critical MSE Alert",
                    "metric": "mse",
                    "threshold": 0.08,
                    "comparison": "greater"
                }
            )

asyncio.run(use_extended_tools())
```

## Best Practices

### Report Generation
- Use Markdown format for human-readable reports
- Use JSON format for programmatic processing
- Use HTML format for web-based sharing
- Enable charts for visual reports

### Scheduling
- Use appropriate intervals based on analysis needs
- Set up email notifications for critical analyses
- Monitor scheduled query performance regularly

### Alert Management
- Set reasonable thresholds based on historical data
- Use descriptive alert names for easy identification
- Regularly review and adjust alert conditions

### Batch Processing
- Limit concurrent queries to prevent resource exhaustion
- Handle partial failures gracefully
- Use batch processing for bulk analysis tasks

### User Preferences
- Set preferences for frequently used parameters
- Use descriptive preference keys
- Regularly update preferences based on usage patterns

### Dashboard Integration
- Choose appropriate time ranges for data freshness
- Cache dashboard data for performance
- Implement real-time updates for critical metrics

### Model Comparison
- Compare similar models for meaningful results
- Include relevant metrics for your use case
- Use longer time ranges for stable comparisons

## Troubleshooting

### Common Issues

**Report Generation Fails**
- Check file system permissions for report directory
- Verify database connectivity for metadata storage
- Ensure sufficient disk space for large reports

**Scheduled Queries Not Running**
- Check server uptime and restart if necessary
- Verify scheduling configuration
- Review error logs for execution failures

**Alerts Not Triggering**
- Confirm metric names match available data
- Check threshold values against actual ranges
- Verify alert conditions are logically correct

**Batch Processing Errors**
- Reduce concurrent queries if resource constrained
- Check individual query validity before batching
- Monitor system resources during execution

**Preference Not Applied**
- Verify user_id matches when setting/retrieving
- Check preference key spelling and format
- Ensure preferences are set before use

### Performance Optimization

- Use appropriate concurrency limits for your system
- Cache frequently accessed data
- Schedule heavy operations during off-peak hours
- Monitor resource usage and adjust limits accordingly

### Security Considerations

- Validate all user inputs
- Implement rate limiting per user
- Log all tool usage for audit trails
- Use secure storage for sensitive preferences

## Support and Updates

For additional support or feature requests, please refer to the main project documentation or create an issue in the project repository.