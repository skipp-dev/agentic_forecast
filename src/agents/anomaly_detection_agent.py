import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict

class AnomalyDetectionAgent:
    """
    Agent for detecting anomalies in time series data.
    """

    def detect_anomalies(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Detects anomalies in the 'close' price for each symbol using IsolationForest.
        """
        anomalies = {}
        for symbol, df in raw_data.items():
            # Work on a copy to avoid side effects
            df_copy = df.copy()
            
            # Using a fixed contamination rate is more robust for financial data
            model = IsolationForest(contamination=0.05, random_state=42)
            
            # Reshape data for the model
            data_to_fit = df_copy[['close']].values
            
            # Fit the model and predict anomalies
            df_copy['anomaly'] = model.fit_predict(data_to_fit)
            
            # Anomalies are marked as -1 by the model
            anomaly_df = df_copy[df_copy['anomaly'] == -1]
            anomalies[symbol] = anomaly_df
            
        return anomalies
