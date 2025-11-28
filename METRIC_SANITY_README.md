# Metric Sanity Reporting System

This system provides comprehensive metric sanity checking and reporting for the agentic forecasting platform. It closes the loop between raw metrics, automated sanity checks, and LLM-powered explanations.

## Overview

The system consists of three main components:

1. **QualityAgent** - Generates detailed sanity reports (JSON + Markdown)
2. **LangGraph Node** - Calls ReportingLLM for intelligent explanations
3. **MCP Tools** - Provides interactive access for analysis

## Generated Reports

### JSON Report (`results/quality/metric_sanity_latest.json`)

Machine-readable report with:
- Run metadata (timestamps, source files, coverage)
- Overall status (passed/failed, severity, issue count)
- Per-metric summaries (stats + issues)
- Horizon-level issues
- Symbol examples
- Sanity flags

### Markdown Report (`results/quality/metric_sanity_latest.md`)

Human-readable report with:
- Executive summary
- Detailed metric breakdowns
- Issue highlights
- Recommended follow-up actions

### LLM Summary (`results/quality/metric_sanity_summary.json`)

AI-generated analysis with:
- Status summary
- Key findings
- Recommended actions
- Risk assessment

## Usage

### Command Line

Generate sanity reports:
```bash
python -m agents.quality_agent --sanity-report
```

### Python API

```python
from agents.quality_agent import QualityAssuranceAgent

agent = QualityAssuranceAgent()
result = agent.run_metric_sanity_report()  # Returns JSON structure
```

### LangGraph Integration

```python
from metric_sanity_explainer import metric_sanity_explainer_node, GraphState

state = GraphState(
    metric_sanity_report_path="results/quality/metric_sanity_latest.json",
    metric_sanity_summary_path="results/quality/metric_sanity_summary.json"
)

result_state = metric_sanity_explainer_node(state)
```

### MCP Tool (for Claude/Interactive Analyst)

```python
from mcp_tools import handle_metric_sanity_explainer

# Structured JSON response
result = handle_metric_sanity_explainer({"mode": "structured"})

# Executive markdown response
result = handle_metric_sanity_explainer({"mode": "executive"})
```

## LLM Prompt Templates

The system includes several prompt templates in `llm_prompts.py`:

- `METRIC_SANITY_EXEC_SUMMARY_PROMPT` - Executive summary
- `METRIC_SANITY_EXEC_STRUCTURED_PROMPT` - Structured JSON
- `METRIC_SANITY_CIO_PROMPT` - CIO-friendly summary
- `METRIC_SANITY_DEV_DEBUG_PROMPT` - Developer debugging

## Integration Points

### Grafana Dashboard

The JSON report feeds into Prometheus metrics:
- `metric_sanity_ok` - Overall status
- `metric_sanity_issue_count` - Issues per metric
- `metric_sanity_severity_level` - Severity indicators

### Automated Workflows

After evaluation runs:
1. QualityAgent generates sanity reports
2. LangGraph node calls LLM for analysis
3. Results feed into dashboards and alerts
4. Stakeholders get automated explanations

## File Structure

```
results/quality/
├── metric_sanity_latest.json      # Main sanity report
├── metric_sanity_latest.md        # Human-readable version
└── metric_sanity_summary.json     # LLM analysis summary

agents/
└── quality_agent.py               # Extended with sanity reporting

llm_prompts.py                     # LLM prompt templates
metric_sanity_explainer.py         # LangGraph node
mcp_tools.py                       # MCP tool handlers
```

## Example Output

### JSON Report Structure
```json
{
  "run_metadata": {
    "run_id": "2025-11-28T21-01-55Z",
    "evaluated_at": "2025-11-28T21:01:55.980612Z",
    "source_file": "data/metrics/evaluation_results_baseline_latest.csv",
    "n_rows": 1972,
    "symbols_covered": 986,
    "horizons_covered": ["1", "5"]
  },
  "overall_status": {
    "status": "failed",
    "severity": "low",
    "issue_count": 1,
    "summary": "Found 1 quality issues"
  },
  "metric_summaries": {
    "smape": {
      "mean": 0.31,
      "std": 0.12,
      "min": 0.01,
      "max": 2.2,
      "unique_values": 8,
      "issues": ["Only 8 unique SMAPE values across all symbols/horizons."]
    }
  }
}
```

### LLM Analysis
```json
{
  "status_summary": "Metric sanity check shows low-severity issues with SMAPE variability.",
  "key_findings": [
    "SMAPE has very few unique values, suggesting potential calculation issues.",
    "Other metrics appear healthy with good variability."
  ],
  "recommended_actions": [
    "Review SMAPE implementation for per-symbol calculation consistency.",
    "Check if shock weighting is being applied correctly in SWASE."
  ],
  "risk_assessment": "Metrics are likely trustworthy but SMAPE issues should be investigated before relying on automated model selection."
}
```

## Development

Run the demo:
```bash
python demo_metric_sanity_pipeline.py
```

Test individual components:
```bash
# Quality agent
python -m agents.quality_agent --sanity-report

# LangGraph node
python metric_sanity_explainer.py

# MCP tools
python mcp_tools.py
```