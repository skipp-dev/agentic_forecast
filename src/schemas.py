
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, validator

class MarketData(BaseModel):
    """Standardized Market Data Structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None
    technicals: Optional[Dict[str, float]] = Field(default_factory=dict)

class ForecastRequest(BaseModel):
    """Request for a forecast generation"""
    symbol: str
    period: str = "1y"
    interval: str = "daily"
    include_technicals: bool = True
    forecast_horizon: int = 30

class ModelForecast(BaseModel):
    """Raw output from a specific model"""
    model_name: str
    forecast_values: List[float]
    confidence_interval: Optional[Tuple[List[float], List[float]]] = None
    metrics: Optional[Dict[str, float]] = None

class HorizonForecast(BaseModel):
    """Forecast for a specific time horizon"""
    horizon: int
    predicted_return: float
    confidence: str # "low", "medium", "high"
    comment: str
    selected_model_family: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ForecastResult(BaseModel):
    """Consolidated Forecast Result from ForecastAgent"""
    symbol: str
    generated_at: datetime = Field(default_factory=datetime.now)
    valid_until: float # Timestamp
    
    # Forecasts for different horizons
    horizon_forecasts: List[HorizonForecast]
    
    # Risk Assessment
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)
    
    # Narrative
    narrative_summary: Optional[str] = None

class PortfolioState(BaseModel):
    """Current state of the portfolio"""
    cash: float
    positions: Dict[str, float] # Symbol -> Quantity
    total_value: float
    last_updated: datetime = Field(default_factory=datetime.now)

class TradingSignal(BaseModel):
    """Actionable trading signal"""
    symbol: str
    action: str # BUY, SELL, HOLD
    quantity: Optional[float] = None
    weight: Optional[float] = None # Target weight
    reason: str
    confidence: float
    generated_at: datetime = Field(default_factory=datetime.now)
    valid_until: float

class PortfolioAllocation(BaseModel):
    """Target portfolio allocation"""
    target_weights: Dict[str, float]
    risk_metrics: Dict[str, float]
    risk_status: str
    rebalance_required: bool = False
    risk_events: List[Dict[str, Any]] = Field(default_factory=list)
