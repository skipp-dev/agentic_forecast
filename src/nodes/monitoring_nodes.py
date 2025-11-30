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
    
    # Convert DataFrame to serializable format for LangSmith tracing
    if not drift_metrics.empty:
        drift_metrics.index = drift_metrics.index.astype(str)
        state['drift_metrics'] = drift_metrics.to_dict('index')
    else:
        state['drift_metrics'] = {}
    
    if drift_metrics['drift_detected'].any():
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
    
    # Convert DataFrame to serializable format for LangSmith tracing
    if not risk_kpis.empty:
        risk_kpis.index = risk_kpis.index.astype(str)
        state['risk_kpis'] = risk_kpis.to_dict('index')
    else:
        state['risk_kpis'] = {}
    
    print(f"✅ Risk assessment complete.")
    return state
