import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """
    Risk management system to halt trading when safety limits are breached.
    Acts as a "Kill Switch" for the trading engine.
    """
    def __init__(self, max_drawdown_pct: float = 0.05, max_daily_loss_pct: float = 0.03, max_leverage: float = 1.0):
        """
        Args:
            max_drawdown_pct: Maximum allowed drawdown from peak equity (e.g., 0.05 for 5%).
            max_daily_loss_pct: Maximum allowed loss in a single day (e.g., 0.03 for 3%).
            max_leverage: Maximum allowed leverage (Gross Exposure / Equity).
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_leverage = max_leverage
        
        self.peak_equity = 0.0
        self.start_of_day_equity = 0.0
        self.is_tripped = False
        self.trip_reason = ""

    def update_equity(self, current_equity: float, is_new_day: bool = False):
        """
        Update equity tracking. Call this periodically or before orders.
        """
        if self.peak_equity == 0.0:
            self.peak_equity = current_equity
            self.start_of_day_equity = current_equity
            
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        if is_new_day:
            self.start_of_day_equity = current_equity

    def check_risk(self, current_equity: float, current_exposure: float) -> bool:
        """
        Check if any risk limits are breached.
        Returns True if safe, False if tripped.
        """
        if self.is_tripped:
            logger.error(f"Circuit Breaker already TRIPPED: {self.trip_reason}")
            return False
            
        # 1. Max Drawdown Check
        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        if drawdown > self.max_drawdown_pct:
            self._trip(f"Max Drawdown breached: {drawdown:.2%} > {self.max_drawdown_pct:.2%}")
            return False
            
        # 2. Daily Loss Check
        daily_loss = (self.start_of_day_equity - current_equity) / self.start_of_day_equity if self.start_of_day_equity > 0 else 0
        if daily_loss > self.max_daily_loss_pct:
            self._trip(f"Max Daily Loss breached: {daily_loss:.2%} > {self.max_daily_loss_pct:.2%}")
            return False
            
        # 3. Leverage Check
        leverage = current_exposure / current_equity if current_equity > 0 else float('inf')
        if leverage > self.max_leverage:
            # Leverage breach usually blocks new orders but doesn't necessarily kill the system, 
            # but for safety we can treat it as a trip or just a block.
            # Here we treat it as a block (return False) but maybe not a permanent trip?
            # Let's trip for now to be safe.
            self._trip(f"Max Leverage breached: {leverage:.2f} > {self.max_leverage:.2f}")
            return False
            
        return True

    def _trip(self, reason: str):
        self.is_tripped = True
        self.trip_reason = reason
        logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")
