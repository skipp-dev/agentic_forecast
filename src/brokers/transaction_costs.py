from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TransactionCostModel(ABC):
    """
    Abstract base class for transaction cost models (Slippage & Fees).
    """
    @abstractmethod
    def get_slippage_cost(self, order_value: float, quantity: float, price: float, **kwargs) -> float:
        """Calculate slippage cost in dollars."""
        pass

    @abstractmethod
    def get_commission_cost(self, order_value: float, quantity: float, price: float, **kwargs) -> float:
        """Calculate commission cost in dollars."""
        pass

class ZeroCostModel(TransactionCostModel):
    """Ideal execution with no costs."""
    def get_slippage_cost(self, order_value: float, quantity: float, price: float, **kwargs) -> float:
        return 0.0
    
    def get_commission_cost(self, order_value: float, quantity: float, price: float, **kwargs) -> float:
        return 0.0

class LinearSlippageModel(TransactionCostModel):
    """
    Slippage is a fixed percentage of the trade value (bps).
    Commission is a fixed rate per share or per dollar.
    """
    def __init__(self, slippage_bps: float = 5.0, commission_per_share: float = 0.005, min_commission: float = 1.0):
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

    def get_slippage_cost(self, order_value: float, quantity: float, price: float, **kwargs) -> float:
        # Slippage = Value * (bps / 10000)
        return abs(order_value) * (self.slippage_bps / 10000.0)

    def get_commission_cost(self, order_value: float, quantity: float, price: float, **kwargs) -> float:
        # IBKR Pro style: $0.005 per share, min $1.00
        comm = abs(quantity) * self.commission_per_share
        return max(comm, self.min_commission)

class SpreadBasedSlippageModel(TransactionCostModel):
    """
    Slippage is based on the Bid-Ask spread (or a proxy).
    """
    def __init__(self, commission_model: Optional[TransactionCostModel] = None):
        self.commission_model = commission_model or LinearSlippageModel(slippage_bps=0)

    def get_slippage_cost(self, order_value: float, quantity: float, price: float, **kwargs) -> float:
        # Expect 'spread_bps' in kwargs
        spread_bps = kwargs.get('spread_bps', 5.0) # Default fallback
        
        # Cost is half the spread (crossing the spread)
        # If spread is 10bps, you pay 5bps to enter.
        return abs(order_value) * (spread_bps / 2.0 / 10000.0)

    def get_commission_cost(self, order_value: float, quantity: float, price: float, **kwargs) -> float:
        return self.commission_model.get_commission_cost(order_value, quantity, price, **kwargs)
