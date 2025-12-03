from typing import Any

def get_prompt(prompt_type: str, **kwargs: Any) -> str:
    """
    Get a prompt template by type and fill it with arguments.
    """
    if prompt_type == "news_enrichment":
        return _get_news_enrichment_prompt(**kwargs)
    elif prompt_type == "analytics_explanation":
        return _get_analytics_prompt(**kwargs)
    elif prompt_type == "hpo_budget_planning":
        return _get_hpo_planning_prompt(**kwargs)
    else:
        return f"Prompt type {prompt_type} not found."

def _get_news_enrichment_prompt(symbol: str, headline: str, body: str = "", timestamp: str = "") -> str:
    return f"""
You are a financial news analyst. Analyze the following news item for {symbol}.

Headline: {headline}
Date: {timestamp}
Body Snippet: {body}

Return a JSON object with the following fields:
- "categories": List of strings (e.g., "earnings", "merger", "macro", "product_launch", "legal", "analyst_rating").
- "directional_impact": "positive", "negative", or "neutral".
- "impact_horizon": "intraday", "short_term" (1-3 days), "medium_term" (1-4 weeks), or "long_term".
- "volatility_impact": "low", "medium", or "high".
- "confidence": Float between 0.0 and 1.0.
- "sentiment_score": Float between -1.0 (very negative) and 1.0 (very positive).
- "notes": Brief explanation (max 1 sentence).

JSON Output:
"""

def _get_analytics_prompt(**kwargs) -> str:
    return "Analyze the provided financial metrics and provide insights."

def _get_hpo_planning_prompt(total_trials: int, min_trials: int, max_trials: int, family_performance_json: str) -> str:
    return f"""
You are an expert Machine Learning Engineer specializing in Time Series Forecasting.
Your task is to allocate a hyperparameter optimization (HPO) budget across different model families based on their past performance.

Constraints:
- Total Budget: {total_trials} trials
- Min Trials per Family: {min_trials}
- Max Trials per Family: {max_trials}

Past Performance (JSON):
{family_performance_json}

Task:
1. Analyze the past performance.
2. Allocate trials to model families. Give more trials to promising families (low MAPE, high accuracy) or unexplored ones.
3. Return a JSON object with the plan.

JSON Format:
{{
    "jobs": [
        {{
            "model_family": "string",
            "priority": "high|medium|low",
            "n_trials": int,
            "search_space": {{ "param": "range" }},
            "notes": "reasoning"
        }}
    ],
    "global_notes": "overall strategy"
}}
"""
