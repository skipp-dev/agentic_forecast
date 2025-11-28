from ..graphs.state import GraphState
from ..agents.anomaly_detection_agent import AnomalyDetectionAgent

def anomaly_detection_node(state: GraphState) -> GraphState:
    """
    Detects anomalies in the raw data.
    """
    print("--- Node: Anomaly Detection ---")
    
    agent = AnomalyDetectionAgent()
    
    anomalies = agent.detect_anomalies(state['raw_data'])
    
    state['anomalies'] = anomalies

    # For now, just print the detected anomalies
    for symbol, anomaly_df in anomalies.items():
        if not anomaly_df.empty:
            print(f"🚨 Detected {len(anomaly_df)} anomalies for {symbol}:")
            print(anomaly_df)

    print("✅ Anomaly detection complete.")
    return state
