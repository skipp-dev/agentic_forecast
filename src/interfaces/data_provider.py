"""
Data Provider Interface

Defines the contract for all data providers (Alpha Vantage, YFinance, etc.).
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List

class DataProvider(ABC):
    """
    Abstract base class for data providers.
    """
    
    @abstractmethod
    def fetch_stock_data(self, symbol: str, period: str = '2y', interval: str = 'daily') -> pd.DataFrame:
        """
        Fetch historical stock data.
        
        Returns:
            DataFrame with columns: [open, high, low, close, volume] and DatetimeIndex.
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the provider.
        """
        pass
