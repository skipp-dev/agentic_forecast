from typing import Dict
import pandas as pd
import numpy as np

class DriftDetectionAgent:
    def detect_drift(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Detects data drift by comparing the mean and standard deviation of the close price
        for the last 30 days with the previous 30 days.
        """
        print("🕵️‍ Running drift detection...")
        
        drift_results = []
        
        for symbol, df in data.items():
            if len(df) < 60:
                print(f"⚠️ Not enough data for drift detection on {symbol}. Skipping.")
                continue

            # Ensure the DataFrame is sorted by date (handle both index and column cases)
            if 'date' in df.columns:
                df = df.sort_values(by='date').reset_index(drop=True)
            else:
                # Assume datetime index
                df = df.sort_index().reset_index(drop=True)

            # Split data into two 30-day periods
            recent_data = df.tail(30)
            previous_data = df.iloc[-60:-30]

            # Calculate statistics for both periods
            recent_mean = recent_data['close'].mean()
            previous_mean = previous_data['close'].mean()
            
            recent_std = recent_data['close'].std()
            previous_std = previous_data['close'].std()

            # Calculate percentage change
            mean_drift = (recent_mean - previous_mean) / previous_mean * 100
            std_drift = (recent_std - previous_std) / previous_std * 100 if previous_std != 0 else 0

            # Define drift thresholds
            mean_threshold = 10  # 10% change in mean
            std_threshold = 20   # 20% change in standard deviation

            # Check for drift
            drift_detected = abs(mean_drift) > mean_threshold or abs(std_drift) > std_threshold
            
            drift_results.append({
                'symbol': symbol,
                'drift_detected': drift_detected,
                'mean_drift_percentage': round(mean_drift, 2),
                'std_drift_percentage': round(std_drift, 2),
                'recent_mean': round(recent_mean, 2),
                'previous_mean': round(previous_mean, 2),
                'recent_std': round(recent_std, 2),
                'previous_std': round(previous_std, 2),
            })

        return pd.DataFrame(drift_results)

