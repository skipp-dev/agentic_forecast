import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from datetime import datetime, time, date
import pytz

class MarketCalendar:
    """
    A helper class to handle market calendar logic (NYSE/NASDAQ).
    Uses pandas USFederalHolidayCalendar as a proxy for market holidays.
    """
    def __init__(self):
        self.cal = USFederalHolidayCalendar()
        self.holidays = self.cal.holidays(start='2020-01-01', end='2030-12-31')
        self.bday_us = CustomBusinessDay(calendar=self.cal)
        self.timezone = pytz.timezone('US/Eastern')
        
        # Standard market hours (simplified)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)

    def is_trading_day(self, check_date: date) -> bool:
        """Check if a given date is a trading day."""
        # Check if weekend
        if check_date.weekday() >= 5:  # 5=Sat, 6=Sun
            return False
        
        # Check if holiday
        if pd.Timestamp(check_date) in self.holidays:
            return False
            
        return True

    def is_market_open(self, check_datetime: datetime) -> bool:
        """Check if the market is currently open."""
        # Convert to Eastern time if needed
        if check_datetime.tzinfo is None:
            # Assume UTC if naive, or handle as error? 
            # For safety, let's assume input is UTC and convert to Eastern
            check_datetime = pytz.utc.localize(check_datetime).astimezone(self.timezone)
        else:
            check_datetime = check_datetime.astimezone(self.timezone)
            
        check_date = check_datetime.date()
        
        if not self.is_trading_day(check_date):
            return False
            
        current_time = check_datetime.time()
        return self.market_open <= current_time <= self.market_close

    def get_last_trading_day(self, check_date: date) -> date:
        """Get the last valid trading day before or on the given date."""
        ts = pd.Timestamp(check_date)
        if self.is_trading_day(check_date):
            return check_date
        
        # Go back until we find a trading day
        # Using pandas offset logic
        last_trading = ts - self.bday_us
        # The offset might jump multiple days (e.g. over weekend)
        # But CustomBusinessDay logic with holidays should handle it.
        # However, CustomBusinessDay arithmetic is sometimes tricky with "on" the day.
        
        # Simpler approach: iterate back (max 10 days)
        for i in range(1, 10):
            prev = ts - pd.Timedelta(days=i)
            if self.is_trading_day(prev.date()):
                return prev.date()
        return check_date # Fallback

    def get_market_status(self, check_datetime: datetime) -> dict:
        """
        Return a status dict for the agent.
        """
        # Ensure timezone awareness
        if check_datetime.tzinfo is None:
             check_datetime = pytz.utc.localize(check_datetime).astimezone(self.timezone)
        else:
             check_datetime = check_datetime.astimezone(self.timezone)

        d = check_datetime.date()
        is_trading = self.is_trading_day(d)
        is_open = self.is_market_open(check_datetime)
        
        status = "closed"
        reason = "market_hours"
        
        if not is_trading:
            if d.weekday() >= 5:
                reason = "weekend"
            else:
                reason = "holiday"
        elif is_open:
            status = "open"
            reason = "market_open"
        else:
            # Trading day but outside hours
            reason = "after_hours" if check_datetime.time() > self.market_close else "pre_market"

        return {
            "status": status,
            "reason": reason,
            "is_trading_day": is_trading,
            "date": str(d),
            "timestamp": str(check_datetime)
        }
