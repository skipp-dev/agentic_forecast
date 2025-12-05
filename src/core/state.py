from typing import TypedDict, List, Dict, Any, Optional
import pandas as pd

class PipelineGraphState(TypedDict):
    """
    State for the Agentic Forecast Pipeline Graph.
    """
    symbols: List[str]
    start_date: str
    end_date: str
    run_id: str
    config: Dict[str, Any]
    run_type: str
    
    # Data artifacts
    data: Dict[str, pd.DataFrame]  # Map symbol -> DataFrame (raw/processed)
    features: Dict[str, pd.DataFrame] # Map symbol -> DataFrame with features
    
    # Model artifacts
    best_models: Dict[str, Dict[str, Any]] # Map symbol -> model_config/params
    
    # Forecast artifacts
    forecasts: Dict[str, pd.DataFrame] # Map symbol -> forecast DataFrame
    
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
