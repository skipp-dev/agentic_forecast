from typing import Any

def get_prompt(prompt_type: str, **kwargs: Any) -> str:
    """
    Get a prompt template by type and fill it with arguments.
    """
    if prompt_type == "news_enrichment":
        return _get_news_enrichment_prompt(**kwargs)
    elif prompt_type == "analytics_explanation":
        return _get_analytics_prompt(**kwargs)
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
