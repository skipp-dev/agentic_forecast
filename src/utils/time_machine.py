import pandas as pd
from typing import Union, Optional, List
import logging

logger = logging.getLogger(__name__)

class TimeMachine:
    """
    A wrapper around pandas DataFrames to enforce strict time-windowing.
    Prevents look-ahead bias by filtering data based on a cutoff date.
    """
    def __init__(self, data: pd.DataFrame, date_col: str = 'date'):
        self.data = data.copy()
        self.date_col = date_col
        
        # Ensure date column is datetime
        if self.date_col in self.data.columns:
            self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])
        elif isinstance(self.data.index, pd.DatetimeIndex):
            # If index is datetime, we can use it
            pass
        else:
            logger.warning(f"Date column {date_col} not found and index is not DatetimeIndex.")

    def get_data_as_of(self, cutoff_date: Union[str, pd.Timestamp], lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        Get data available as of the cutoff date.
        
        Args:
            cutoff_date: The maximum allowed date (inclusive).
            lookback_days: Optional number of days to look back from cutoff.
            
        Returns:
            Filtered DataFrame.
        """
        cutoff = pd.to_datetime(cutoff_date)
        
        if self.date_col in self.data.columns:
            mask = self.data[self.date_col] <= cutoff
            if lookback_days:
                start_date = cutoff - pd.Timedelta(days=lookback_days)
                mask = mask & (self.data[self.date_col] >= start_date)
            return self.data[mask].copy()
        elif isinstance(self.data.index, pd.DatetimeIndex):
            mask = self.data.index <= cutoff
            if lookback_days:
                start_date = cutoff - pd.Timedelta(days=lookback_days)
                mask = mask & (self.data.index >= start_date)
            return self.data[mask].copy()
        else:
            # Fallback: return all data but warn
            logger.warning("TimeMachine cannot filter data without valid date column/index.")
            return self.data.copy()

    def get_latest_date(self) -> pd.Timestamp:
        """Get the latest date in the dataset."""
        if self.date_col in self.data.columns:
            return self.data[self.date_col].max()
        elif isinstance(self.data.index, pd.DatetimeIndex):
            return self.data.index.max()
        return pd.Timestamp.now()
