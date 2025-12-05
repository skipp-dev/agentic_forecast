from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Order:
    symbol: str
    action: str  # BUY, SELL
    quantity: float
    order_type: str = "MARKET"
    price: Optional[float] = None

@dataclass
class Fill:
    order_id: str
    symbol: str
    action: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    status: str = "FILLED"

class BrokerInterface(ABC):
    """
    Abstract base class for broker interactions.
    """
    
    @abstractmethod
    def get_cash(self) -> float:
        """Get current cash balance."""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """Get current positions {symbol: quantity}."""
        pass
        
    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + equity)."""
        pass

    @abstractmethod
    def place_order(self, order: Order) -> Fill:
        """Place an order and return a fill (or rejection)."""
        pass
