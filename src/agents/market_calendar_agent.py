from typing import Dict, Any
from datetime import datetime
from ..utils.market_calendar import MarketCalendar

class MarketCalendarAgent:
    """
    Agent responsible for checking market status and determining if the pipeline should run.
    """
    def __init__(self):
        self.calendar = MarketCalendar()

    def check_market_status(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks the current market status and updates the state.
        """
        print("📅 MarketCalendarAgent: Checking market status...")
        
        # Use current time or a time from config if provided (for backtesting/simulation)
        current_time = datetime.now()
        if 'config' in state and 'simulation_date' in state['config']:
             sim_date_str = state['config']['simulation_date']
             # Assuming format YYYY-MM-DD or ISO
             try:
                 current_time = datetime.fromisoformat(sim_date_str)
                 print(f"📅 MarketCalendarAgent: Using simulation date: {current_time}")
             except ValueError:
                 print(f"⚠️ MarketCalendarAgent: Invalid simulation_date format {sim_date_str}, using current time.")

        status = self.calendar.get_market_status(current_time)
        
        print(f"📅 MarketCalendarAgent: Status is {status['status']} ({status['reason']})")
        
        return {
            "market_status": status
        }
