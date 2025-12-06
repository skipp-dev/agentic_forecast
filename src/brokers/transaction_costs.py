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

class VolatilityAdjustedSlippageModel(TransactionCostModel):
    """
    Slippage model that adjusts costs based on market volatility and trade size relative to volume.
    
    Model:
    Slippage (bps) = Base Spread + (Vol Factor * Daily Volatility) + (Impact Factor * (Order Size / Avg Volume)^0.5)
    """
    def __init__(self, base_spread_bps: float = 2.0, vol_factor: float = 0.05, impact_factor: float = 0.01, commission_per_share: float = 0.005):
        self.base_spread_bps = base_spread_bps
        self.vol_factor = vol_factor
        self.impact_factor = impact_factor
        self.commission_per_share = commission_per_share
        self.min_commission = 1.0

    def get_slippage_cost(self, order_value: float, quantity: float, price: float, **kwargs) -> float:
        """
        Requires 'volatility' (decimal, e.g. 0.015 for 1.5%) and 'avg_volume' (shares) in kwargs.
        """
        volatility = kwargs.get('volatility', 0.01) # Default 1% daily vol
        avg_volume = kwargs.get('avg_volume', 1000000.0) # Default 1M shares
        
        # 1. Volatility Component
        # If vol is high, spreads widen.
        # Volatility (decimal) * 10000 -> Vol in Bps
        # e.g. 0.01 * 10000 = 100 bps. 100 * 0.05 = 5 bps added.
        vol_component_bps = (volatility * 10000.0) * self.vol_factor
        
        # 2. Market Impact Component (Square Root Law)
        # Impact ~ sigma * sqrt(Q / V)
        # We approximate impact as proportional to sqrt of participation rate.
        participation_rate = abs(quantity) / avg_volume if avg_volume > 0 else 0.01
        # Cap participation rate for calculation to avoid explosion on low volume data errors
        participation_rate = min(participation_rate, 1.0)
        
        # e.g. 1% participation (0.01) -> sqrt(0.01)=0.1
        # 0.1 * 10000 * 0.01 (impact_factor) = 10 bps.
        impact_bps = self.impact_factor * (participation_rate ** 0.5) * 10000.0
        
        total_slippage_bps = self.base_spread_bps + vol_component_bps + impact_bps
        
        # Cap at reasonable limits (e.g. 500bps = 5%)
        total_slippage_bps = min(max(total_slippage_bps, 0.0), 500.0)
        
        return abs(order_value) * (total_slippage_bps / 10000.0)

    def get_commission_cost(self, order_value: float, quantity: float, price: float, **kwargs) -> float:
        comm = abs(quantity) * self.commission_per_share
        return max(comm, self.min_commission)
