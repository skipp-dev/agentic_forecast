from typing import TypedDict, List, Dict, Any
import pandas as pd
import torch

class GraphState(TypedDict):
    """
    Represents the state of the forecasting graph.
    """
    symbols: List[str]
    config: Dict[str, Any]
    raw_data: Dict[str, pd.DataFrame]
    features: Dict[str, Any] # Serialized to dict
    forecasts: Dict[str, Dict[str, pd.DataFrame]]
    
    # Analytics Outputs
    performance_summary: pd.DataFrame
    drift_metrics: Dict[str, Any] # Serialized to dict
    risk_kpis: Dict[str, Any] # Serialized to dict
    anomalies: Dict[str, Any] # Serialized to dict
    
    # Decision & Action Outputs
    recommended_actions: List[Dict[str, Any]]
    executed_actions: List[Dict[str, Any]]
    retrained_models: Dict[str, str]
    best_models: Dict[str, Dict[str, Any]]
    errors: List[str]
    
    # HPO results
    hpo_results: Dict[str, Any]
    
    # Time Travel
    cutoff_date: str # Optional ISO date string for point-in-time simulation

    # SHAP explainability results
    shap_results: Dict[str, Dict[str, Any]]
    analytics_summary: pd.DataFrame
    hpo_decision: Dict[str, Any]

    # Graph construction outputs
    edge_index: torch.Tensor
    node_features: torch.Tensor
    symbol_to_idx: Dict[str, int]

    # Operational metadata
    retraining_history: List[Dict[str, Any]]
    guardrail_log: List[str]
    hpo_triggered: bool
    drift_detected: bool
    run_type: str
    
    # Supervisor Control
    run_status: str # INIT, RUNNING, COMPLETED, FAILED
    supervisor_iterations: int
    failure_reason: str

    # LLM & News Agents Outputs
    news_insights: Any
    market_sentiment: str
    
    # Market Calendar Status
    market_status: Dict[str, Any]
    key_news: List[Dict[str, Any]]
    enriched_news: List[Dict[str, Any]]
    llm_analytics_summary: str
    llm_actions: List[str]
    llm_notes: Dict[str, Any]
    llm_hpo_plan: Dict[str, Any]
    interpreted_forecasts: Dict[str, Any]
    
    # Continuous Learning
    continuous_learning_applied: bool
    learning_feedback: Dict[str, Any]
    
    # Trust Score
    trust_scores: Dict[str, float]

    # Macro & Regime
    macro_data: Dict[str, Any]
    regimes: Dict[str, str]
