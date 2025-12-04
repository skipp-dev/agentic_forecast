"""
Seasonality Agent

Handles detection of seasonal patterns and calendar events.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class SeasonalityAgent:
    """
    Agent for detecting seasonal patterns and calendar events.
    """
    
    def __init__(self):
        pass
        
    def get_seasonality_features(self, date: str) -> Dict[str, Any]:
        """
        Get seasonality features for a specific date.
        
        Args:
            date: Date string in YYYY-MM-DD format
            
        Returns:
            Dictionary of seasonality features
        """
        dt = pd.to_datetime(date)
        
        features = {
            'month': dt.month,
            'day_of_week': dt.dayofweek,
            'quarter': dt.quarter,
            'is_month_end': self._is_month_end(dt),
            'is_quarter_end': dt.is_quarter_end,
            'is_year_end': dt.is_year_end,
            'is_earnings_season': self._is_earnings_season(dt),
            'season': self._get_season(dt)
        }
        
        return features
        
    def _is_month_end(self, dt: pd.Timestamp) -> bool:
        """Check if date is near month end (last 3 days)."""
        next_month = dt.replace(day=28) + timedelta(days=4)
        last_day = next_month - timedelta(days=next_month.day)
        return (last_day - dt).days <= 2
        
    def _is_earnings_season(self, dt: pd.Timestamp) -> bool:
        """
        Heuristic for earnings season.
        Usually starts 2 weeks after quarter end and lasts ~6 weeks.
        """
        # Quarter ends: Mar 31, Jun 30, Sep 30, Dec 31
        # Earnings: Apr 15-May 31, Jul 15-Aug 31, Oct 15-Nov 30, Jan 15-Feb 28
        
        month = dt.month
        day = dt.day
        
        if month in [1, 4, 7, 10]:
            return day >= 15
        elif month in [2, 5, 8, 11]:
            return True
        return False
        
    def _get_season(self, dt: pd.Timestamp) -> str:
        """Get meteorological season (Northern Hemisphere)."""
        month = dt.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
