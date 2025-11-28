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
    features: Dict[str, pd.DataFrame]
    forecasts: Dict[str, Dict[str, pd.DataFrame]]
    
    # Analytics Outputs
    performance_summary: pd.DataFrame
    drift_metrics: pd.DataFrame
    risk_kpis: pd.DataFrame
    anomalies: Dict[str, pd.DataFrame]
    
    # Decision & Action Outputs
    recommended_actions: List[Dict[str, Any]]
    executed_actions: List[Dict[str, Any]]
    retrained_models: Dict[str, str]
    best_models: Dict[str, Dict[str, Any]]
    errors: List[str]
    
    # HPO results
    hpo_results: Dict[str, Any]

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
