"""
LLM Prompt Library for Agentic Forecasting.
Contains templates for various agent tasks.
"""

PROMPTS = {
    "analytics_summary": """
You are a quantitative analyst.

Here is a JSON table "performance_summary" with one row per model family, symbol, and horizon.
Each row has: symbol, horizon, model_family, mape, mae, directional_accuracy, sample_size.

performance_summary:
```json
{performance_summary_json}
```

Tasks:
1. Summarize in plain language which model families are performing best and worst.
2. Highlight any symbols or horizons where performance is clearly unacceptable.
3. Keep the summary under 300 words.

Output:
- A short markdown summary.
""",

    "drift_analysis": """
You are a drift analysis expert.

You receive a JSON array "drift_events", each row has:
symbol, horizon, model_family, baseline_mape, recent_mape, mape_change, drift_score, first_detected_at, last_confirmed_at.

drift_events:
```json
{drift_events_json}
```

Tasks:
1. Identify the top 10 most critical drift cases.
2. For each, explain in 2-3 sentences what changed and why this is concerning.
3. Suggest a recommended action tag for each: "retrain", "hpo", "switch_model_family", or "investigate_data".

Output:
- A JSON array with fields: symbol, horizon, model_family, severity, action, explanation.
""",

    "hpo_search_space": """
You are a senior ML engineer.

Here are past HPO runs for several model families. Each row has:
model_family, params, val_mape, directional_accuracy, status.

past_hpo_runs:
```json
{past_hpo_runs_json}
```

Task:
Based on the successful trials (low val_mape, high directional_accuracy), propose a narrowed hyperparameter search space for the next HPO round for each model_family.

For each model_family, output JSON like:
{{
  "model_family": "NHITS",
  "search_space": {{
    "hidden_size": {{"type": "int", "low": 64, "high": 256}},
    "learning_rate": {{"type": "logfloat", "low": 1e-4, "high": 3e-3}},
    "num_layers": {{"type": "int", "low": 2, "high": 4}}
  }},
  "notes": "short comment"
}}

Output:
A JSON array with one object per model_family.
""",

    "hpo_budget_planning": """
You are planning the HPO budget for the next cycle.

Constraints:
- total_trials_budget = {total_trials}
- min_trials_per_family = {min_trials}
- max_trials_per_family = {max_trials}

Here is the latest performance summary per model family:
```json
{family_performance_json}
```

Task:
1. Decide which model families should get HPO budget and how many trials.
2. For each chosen family, assign a priority: "high", "medium", or "low".

Output:
JSON:
{{
  "jobs": [
    {{"model_family": "...", "n_trials": 20, "priority": "high", "reason": "..."}},
    ...
  ],
  "global_notes": "..."
}}
""",

    "news_enrichment": """
You are a financial news analyst. Convert the following news item into structured features:

News:
symbol: {symbol}
timestamp: {timestamp}
headline: {headline}
body: {body}

Task:
1. Assign one or more categories from:
   ["earnings", "guidance", "mergers_acquisitions", "regulatory", "geopolitics",
    "product_launch", "management_change", "macro", "legal", "other"]
2. Decide directional_impact: "bullish", "bearish", "neutral", or "mixed".
3. Decide impact_horizon: "intraday", "short_term", "medium_term", "long_term".
4. Decide volatility_impact: "low", "medium", "high".
5. Provide a confidence between 0 and 1 and a short note.

Output JSON:
{{
  "categories": [...],
  "directional_impact": "...",
  "impact_horizon": "...",
  "volatility_impact": "...",
  "confidence": 0.0,
  "notes": "..."
}}
"""
}

def get_prompt(task_name: str, **kwargs) -> str:
    """Get a formatted prompt for a specific task."""
    if task_name not in PROMPTS:
        raise ValueError(f"Unknown task: {task_name}")
    
    return PROMPTS[task_name].format(**kwargs)
