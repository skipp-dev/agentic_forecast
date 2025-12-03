from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta

def detect_news_shock(
    symbol: str,
    daily_aggregates: List[Dict[str, Any]],
    config: Dict[str, Any] = None
) -> bool:
    """
    Detect if a symbol is in a news shock state based on recent news aggregates.
    
    Args:
        symbol: The stock symbol.
        daily_aggregates: List of daily news aggregate dictionaries.
                          Expected keys: 'date', 'news_count', 'max_abs_sentiment', 
                          'news_impact_score', 'news_volatility_score'.
        config: Configuration dictionary containing thresholds.
        
    Returns:
        True if a shock is detected, False otherwise.
    """
    if not daily_aggregates:
        return False
        
    # Default config
    cfg = config or {}
    lookback_days = cfg.get("lookback_days", 3)
    sentiment_threshold = cfg.get("sentiment_max_abs", 0.8)
    count_threshold = cfg.get("article_count_spike", 10)
    impact_threshold = cfg.get("impact_score_threshold", 5.0)
    
    # Filter for recent days
    # Assuming daily_aggregates are sorted or we just check the last few
    # We'll parse dates to be safe
    
    # Sort by date descending
    try:
        sorted_aggs = sorted(daily_aggregates, key=lambda x: x.get('date', ''), reverse=True)
    except Exception:
        sorted_aggs = daily_aggregates # Fallback
        
    recent_aggs = sorted_aggs[:lookback_days]
    
    for agg in recent_aggs:
        # Check 1: Extreme sentiment
        if agg.get("max_abs_sentiment", 0) >= sentiment_threshold:
            return True
            
        # Check 2: Volume spike
        if agg.get("news_count", 0) >= count_threshold:
            return True
            
        # Check 3: High Impact Score
        if agg.get("news_impact_score", 0) >= impact_threshold:
            return True

        # Check 4: High Volatility Score (if normalized 0-1)
        if agg.get("news_volatility_score", 0) >= 0.9:
            return True
            
    return False
