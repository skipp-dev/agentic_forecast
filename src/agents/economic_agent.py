"""
Economic Context Agent

Enriches risk assessment by incorporating macroeconomic data.
Analyzes inflation, interest rates, and other indicators to provide a broader economic context.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from src.agents.macro_data_agent import MacroDataAgent

logger = logging.getLogger(__name__)

class EconomicContextAgent:
    """
    Agent responsible for analyzing macroeconomic context for risk management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.macro_agent = MacroDataAgent(self.config)
        
    def assess_macro_risk(self, lookback_days: int = 90) -> Dict[str, Any]:
        """
        Assess macroeconomic risk based on recent data trends.
        
        Args:
            lookback_days: Number of days to look back for trend analysis.
            
        Returns:
            Dictionary containing risk assessment and key indicators.
        """
        logger.info("Assessing macroeconomic risk...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Fetch data
        macro_data = self.macro_agent.get_macro_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        raw_data = macro_data.get('raw_data', {})
        
        risk_score = 0.0
        risk_factors = []
        
        # 1. Interest Rate Risk (Rising rates = Higher Risk)
        if 'treasury_10y' in raw_data:
            ts = raw_data['treasury_10y']
            if not ts.empty:
                current_yield = ts.iloc[-1]
                start_yield = ts.iloc[0]
                change = current_yield - start_yield
                
                if change > 0.5: # Rates rose by > 50bps
                    risk_score += 0.3
                    risk_factors.append(f"Rising Rates (+{change:.2f}%)")
                elif change < -0.5:
                    risk_factors.append(f"Falling Rates ({change:.2f}%)")
                    
        # 2. Volatility Risk (VIX) - Assuming MacroDataAgent can fetch VIX or we add it
        # For now, let's check Oil as a proxy for commodity shock
        if 'oil' in raw_data:
            ts = raw_data['oil']
            if not ts.empty:
                current_price = ts.iloc[-1]
                start_price = ts.iloc[0]
                pct_change = (current_price - start_price) / start_price
                
                if abs(pct_change) > 0.2: # > 20% move
                    risk_score += 0.2
                    risk_factors.append(f"Commodity Volatility (Oil {pct_change*100:.1f}%)")

        # 3. Inflation Risk (if available)
        # CPI is usually monthly, might not be in the short lookback
        
        # Normalize Risk Score (0 to 1)
        risk_score = min(risk_score, 1.0)
        
        assessment = {
            'risk_score': risk_score,
            'risk_level': 'HIGH' if risk_score > 0.7 else 'MEDIUM' if risk_score > 0.3 else 'LOW',
            'risk_factors': risk_factors,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Macro Risk Assessment: {assessment['risk_level']} (Score: {risk_score:.2f})")
        return assessment

    def get_economic_summary(self) -> str:
        """
        Generate a text summary of the current economic context.
        """
        risk = self.assess_macro_risk()
        return f"Economic Risk Level: {risk['risk_level']}. Factors: {', '.join(risk['risk_factors'])}"
