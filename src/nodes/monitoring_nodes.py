from src.agents.drift_detection_agent import DriftDetectionAgent
from src.agents.risk_assessment_agent import RiskAssessmentAgent
from src.graphs.state import GraphState

def drift_detection_node(state: GraphState) -> GraphState:
    """
    Detects data drift in the raw data and updates the state.
    """
    print("--- Node: Drift Detection ---")
    
    drift_detection_agent = DriftDetectionAgent()
    drift_metrics = drift_detection_agent.detect_drift(state['raw_data'])
    
    state['drift_metrics'] = drift_metrics
    
    if not drift_metrics.empty and drift_metrics['drift_detected'].any():
        state['drift_detected'] = True
        print("🚨 Drift detected!")
    else:
        state['drift_detected'] = False
        print("✅ No significant drift detected.")
        
    print(f"✅ Drift detection complete.")
    return state

def risk_assessment_node(state: GraphState) -> GraphState:
    """
    Assesses the risk of the raw data and updates the state.
    """
    print("--- Node: Risk Assessment ---")
    
    risk_assessment_agent = RiskAssessmentAgent()
    risk_kpis = risk_assessment_agent.assess_risk(state['raw_data'])
    
    state['risk_kpis'] = risk_kpis
    print(f"✅ Risk assessment complete.")
    return state
