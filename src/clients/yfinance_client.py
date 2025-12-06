"""
YFinance Client

Fallback data provider using Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import logging
from typing import Optional
from src.interfaces.data_provider import DataProvider

logger = logging.getLogger(__name__)

class YFinanceClient(DataProvider):
    """
    Data provider implementation using yfinance.
    """
    
    def get_name(self) -> str:
        return "YFinance"

    def fetch_stock_data(self, symbol: str, period: str = '2y', interval: str = 'daily') -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        """
        logger.info(f"Fetching data for {symbol} from YFinance (period={period}, interval={interval})")
        
        # Map interval to yfinance format
        # yfinance supports: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        interval_map = {
            'daily': '1d',
            'weekly': '1wk',
            'monthly': '1mo',
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '30min': '30m',
            '60min': '60m'
        }
        
        yf_interval = interval_map.get(interval, '1d')
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=yf_interval)
            
            if df.empty:
                logger.warning(f"YFinance returned empty data for {symbol}")
                return pd.DataFrame()
                
            # Standardize columns
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            
            # Remove timezone info if present to match other providers
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            # Select only required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[required_cols]
            
            return df
            
        except Exception as e:
            logger.error(f"YFinance fetch failed for {symbol}: {e}")
            return pd.DataFrame()
