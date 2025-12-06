from typing import TypedDict, List, Dict, Any, Optional, Union
import pandas as pd

class PipelineGraphState(TypedDict):
    """
    State for the Agentic Forecast Pipeline Graph.
    
    Optimization Note:
    Large artifacts (DataFrames) should be stored as file paths (str) pointing to parquet files
    managed by StateManager, rather than raw DataFrames, to reduce memory overhead.
    """
    symbols: List[str]
    start_date: str
    end_date: str
    run_id: str
    config: Dict[str, Any]
    run_type: str
    
    # Data artifacts (Map symbol -> path_to_parquet or DataFrame)
    data: Dict[str, Union[str, pd.DataFrame]] 
    features: Dict[str, Union[str, pd.DataFrame]]
    macro_data: Dict[str, Any] 
    regimes: Dict[str, Any] 
    
    # Model artifacts
    best_models: Dict[str, Dict[str, Any]] 
    
    # Forecast artifacts
    forecasts: Dict[str, Union[str, pd.DataFrame]]
    
    # Analytics artifacts
    analytics_results: Dict[str, Dict[str, Any]] # Map symbol -> analytics dict (MAPE, etc.)
    
    # Retraining artifacts
    drift_detected: List[str] # List of symbols where drift was detected
    drift_metrics: Dict[str, Any] # Detailed drift metrics per symbol
    retrained_models: List[str] # List of symbols/models that were retrained
    
    # Orchestration flags
    hpo_triggered: bool
    hpo_results: Dict[str, Any] # Results from HPO
    errors: List[str]
    run_status: str
    next_step: str
    deep_research_conducted: bool
    horizon_forecasts: Dict[str, Any]
    interpreted_forecasts: bool
    
    # Reporting artifacts
    report_metadata: Dict[str, Any]
    report_generated: bool
    
    # Strategy artifacts
    signals: Dict[str, Any] # Map symbol -> trading signal details
    
    # Portfolio artifacts
    portfolio: Dict[str, float] # Map symbol -> target weight
    orders: List[Dict[str, Any]] # List of orders to execute
