import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class LiquidityAgent:
    """
    Agent responsible for analyzing liquidity, spread, and tradeability of symbols.
    Implements checks for:
    - Spread analysis (Bid-Ask or High-Low proxy)
    - Volume & Dollar Volume
    - Liquidity Stress / Regime Detection
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.min_dollar_volume = self.config.get('min_dollar_volume', 1_000_000)
        self.max_spread_bps = self.config.get('max_spread_bps', 50) # 0.5%
        self.lookback_window = self.config.get('lookback_window', 20)

    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate liquidity and spread metrics for a single symbol's DataFrame.
        Expects columns: 'open', 'high', 'low', 'close', 'volume'.
        Optional columns: 'bid', 'ask'.
        """
        df = df.copy()
        
        # 1. Spread Metrics
        if 'bid' in df.columns and 'ask' in df.columns:
            df['mid_price'] = 0.5 * (df['bid'] + df['ask'])
            df['spread_abs'] = df['ask'] - df['bid']
            df['spread_rel'] = df['spread_abs'] / df['mid_price']
            df['spread_bps'] = df['spread_rel'] * 10000
        else:
            # Proxy: High-Low Range as a volatility/liquidity proxy
            # This is NOT a true spread, but a "trading range" metric
            df['mid_price'] = df['close'] # Fallback
            df['spread_abs'] = df['high'] - df['low'] # Proxy
            df['spread_rel'] = df['spread_abs'] / df['close']
            df['spread_bps'] = df['spread_rel'] * 10000
            # Flag that this is a proxy
            df['spread_is_proxy'] = True

        # 2. Volume Metrics
        df['dollar_volume'] = df['close'] * df['volume']
        df['avg_dollar_vol_20d'] = df['dollar_volume'].rolling(self.lookback_window).mean()
        df['avg_spread_bps_20d'] = df['spread_bps'].rolling(self.lookback_window).mean()
        
        # 3. Gap Risk
        df['gap_return'] = df['open'] / df['close'].shift(1) - 1
        df['gap_abs'] = df['gap_return'].abs()
        
        # 4. Z-Scores (Stress Detection)
        spread_mean = df['spread_bps'].rolling(self.lookback_window).mean()
        spread_std = df['spread_bps'].rolling(self.lookback_window).std()
        df['spread_zscore'] = (df['spread_bps'] - spread_mean) / spread_std.replace(0, 1) # Avoid div/0

        return df

    def check_tradeability(self, symbol: str, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check if a symbol is tradeable based on the latest metrics.
        """
        if metrics_df.empty:
            return {'tradeable': False, 'reason': 'No data'}

        latest = metrics_df.iloc[-1]
        
        reasons = []
        is_tradeable = True

        # Check Dollar Volume
        if latest.get('avg_dollar_vol_20d', 0) < self.min_dollar_volume:
            is_tradeable = False
            reasons.append(f"Low Dollar Volume: ${latest.get('avg_dollar_vol_20d', 0):,.0f} < ${self.min_dollar_volume:,.0f}")

        # Check Spread (if not proxy, or use proxy with higher threshold)
        # If using proxy (High-Low), "spread" is just volatility, so we might not want to filter strictly on it unless it's extreme.
        threshold = self.max_spread_bps
        if latest.get('spread_is_proxy', False):
            threshold *= 4 # Relax threshold for High-Low proxy (e.g. 2%)

        if latest.get('avg_spread_bps_20d', 0) > threshold:
            # Don't hard fail, just flag, unless it's extreme
            reasons.append(f"Wide Spread: {latest.get('avg_spread_bps_20d', 0):.1f} bps > {threshold} bps")
            # is_tradeable = False # Optional: strict filtering

        # Check Liquidity Stress (Z-Score)
        if latest.get('spread_zscore', 0) > 3.0:
            reasons.append(f"Liquidity Stress: Spread Z-Score {latest.get('spread_zscore', 0):.2f} > 3.0")

        # Check Gap Risk
        if latest.get('gap_abs', 0) > 0.10: # 10% gap
            reasons.append(f"Extreme Gap: {latest.get('gap_abs', 0)*100:.1f}%")

        return {
            'symbol': symbol,
            'tradeable': is_tradeable,
            'reasons': reasons,
            'metrics': {
                'dollar_volume': latest.get('avg_dollar_vol_20d', 0),
                'spread_bps': latest.get('avg_spread_bps_20d', 0),
                'spread_zscore': latest.get('spread_zscore', 0)
            }
        }

    def get_slippage_estimate(self, symbol: str, metrics_df: pd.DataFrame) -> float:
        """
        Estimate slippage in basis points based on spread/volatility.
        """
        if metrics_df.empty:
            return 10.0 # Default 10 bps

        latest = metrics_df.iloc[-1]
        
        # Conservative: Slippage is half the spread
        spread_bps = latest.get('spread_bps', 20.0)
        
        # If using proxy (High-Low), this overestimates spread significantly.
        # High-Low is often 1-2%, spread is 0.05%.
        # So if proxy, we need to scale it down or use a fixed model.
        if latest.get('spread_is_proxy', False):
            # Use volatility-based slippage model: k * volatility
            # Simple heuristic: 1% of the daily range
            return spread_bps * 0.01 
        
        return spread_bps * 0.5
