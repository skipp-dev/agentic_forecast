from typing import Dict
import pandas as pd
import numpy as np

class RiskAssessmentAgent:
    """
    Agent for assessing risk by calculating annualized volatility.
    """
    def assess_risk(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Assesses risk by calculating annualized volatility.
        """
        print("🕵️‍ Running risk assessment...")
        
        risk_results = []
        
        for symbol, df in data.items():
            if 'close' not in df.columns or df['close'].isnull().all():
                print(f"⚠️ No close price data for risk assessment on {symbol}. Skipping.")
                continue

            # Calculate daily returns
            daily_returns = df['close'].pct_change().dropna()

            if len(daily_returns) < 2:
                print(f"⚠️ Not enough data for risk assessment on {symbol}. Skipping.")
                continue

            # Calculate annualized volatility
            annualized_volatility = daily_returns.std() * np.sqrt(252) # Assuming 252 trading days in a year

            # Define risk thresholds
            high_risk_threshold = 0.4  # 40% annualized volatility
            
            # Assess risk level
            risk_level = "High" if annualized_volatility > high_risk_threshold else "Normal"
            
            risk_results.append({
                'symbol': symbol,
                'annualized_volatility': round(annualized_volatility, 4),
                'risk_level': risk_level,
            })

        return pd.DataFrame(risk_results)

