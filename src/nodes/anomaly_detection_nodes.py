from ..graphs.state import GraphState
from ..agents.anomaly_detection_agent import AnomalyDetectionAgent

def anomaly_detection_node(state: GraphState) -> GraphState:
    """
    Detects anomalies in the raw data.
    """
    print("--- Node: Anomaly Detection ---")
    
    agent = AnomalyDetectionAgent()
    
    anomalies = agent.detect_anomalies(state['raw_data'])
    
    # Convert DataFrames to serializable format for LangSmith tracing
    serializable_anomalies = {}
    for symbol, df in anomalies.items():
        if not df.empty:
            # Convert DatetimeIndex to string index for JSON serialization
            df_copy = df.copy()
            df_copy.index = df_copy.index.astype(str)
            serializable_anomalies[symbol] = df_copy.to_dict('index')
        else:
            serializable_anomalies[symbol] = {}
    
    state['anomalies'] = serializable_anomalies

    # For now, just print the detected anomalies
    for symbol, anomaly_df in anomalies.items():
        if not anomaly_df.empty:
            print(f"[ALERT] Detected {len(anomaly_df)} anomalies for {symbol}:")
            print(anomaly_df)

    print("[OK] Anomaly detection complete.")
    return state
