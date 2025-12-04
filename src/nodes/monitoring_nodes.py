from ..agents.drift_detection_agent import DriftDetectionAgent
from ..agents.risk_assessment_agent import RiskAssessmentAgent
from ..graphs.state import GraphState

def drift_detection_node(state: GraphState) -> GraphState:
    """
    Detects data drift in the raw data and updates the state.
    """
    print("--- Node: Drift Detection ---")
    
    drift_detection_agent = DriftDetectionAgent()
    drift_metrics = drift_detection_agent.detect_drift(state['raw_data'])
    
    # Convert DataFrame to serializable format for LangSmith tracing
    if not drift_metrics.empty:
        # Ensure index is symbol for efficient lookup
        if 'symbol' in drift_metrics.columns:
            drift_metrics = drift_metrics.set_index('symbol')
        
        drift_metrics.index = drift_metrics.index.astype(str)
        state['drift_metrics'] = drift_metrics.to_dict('index')
        
        if 'drift_detected' in drift_metrics.columns and drift_metrics['drift_detected'].any():
            state['drift_detected'] = True
            print("[ALERT] Drift detected!")
        else:
            state['drift_detected'] = False
            print("[OK] No significant drift detected.")
    else:
        state['drift_metrics'] = {}
        state['drift_detected'] = False
        print("[OK] No data for drift detection.")
        
    print(f"[OK] Drift detection complete.")
    return state

def risk_assessment_node(state: GraphState) -> GraphState:
    """
    Assesses the risk of the raw data and updates the state.
    """
    print("--- Node: Risk Assessment ---")
    
    risk_assessment_agent = RiskAssessmentAgent()
    risk_kpis = risk_assessment_agent.assess_risk(state['raw_data'])
    
    # Convert DataFrame to serializable format for LangSmith tracing
    if not risk_kpis.empty:
        # Ensure index is symbol for efficient lookup
        if 'symbol' in risk_kpis.columns:
            risk_kpis = risk_kpis.set_index('symbol')
            
        risk_kpis.index = risk_kpis.index.astype(str)
        state['risk_kpis'] = risk_kpis.to_dict('index')
    else:
        state['risk_kpis'] = {}
    
    print(f"[OK] Risk assessment complete.")
    return state
