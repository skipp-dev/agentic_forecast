
import pandas as pd
from typing import Dict, Any

# Mock RiskAssessmentAgent behavior
def mock_assess_risk():
    risk_results = []
    symbols = ['AAPL', 'GOOGL']
    for symbol in symbols:
        risk_results.append({
            'symbol': symbol,
            'annualized_volatility': 0.25,
            'risk_level': "Normal",
        })
    return pd.DataFrame(risk_results)

# Mock risk_assessment_node logic
def mock_node():
    risk_kpis = mock_assess_risk()
    if not risk_kpis.empty:
        if 'symbol' in risk_kpis.columns:
            risk_kpis = risk_kpis.set_index('symbol')
        risk_kpis.index = risk_kpis.index.astype(str)
        return risk_kpis.to_dict('index')
    return {}

# Mock TrustScoreCalculator logic
class TrustScoreCalculator:
    def __init__(self):
        self.volatility_high = 0.4
        self.volatility_medium = 0.2
        self.penalties = {}

    def _calculate_regime_score(self, symbol: str, risk_kpis: Dict[str, Any]) -> float:
        if symbol not in risk_kpis:
            print(f"Symbol {symbol} not in risk_kpis")
            return 0.5
            
        kpis = risk_kpis[symbol]
        print(f"kpis for {symbol}: {kpis} (type: {type(kpis)})")
        
        # This is where the error happens if kpis is a string
        try:
            volatility = kpis.get('volatility', 0.0)
            print(f"Volatility: {volatility}")
        except AttributeError as e:
            print(f"Error: {e}")
            return 0.0
            
        return 1.0

# Run test
risk_kpis = mock_node()
print(f"risk_kpis: {risk_kpis}")

calculator = TrustScoreCalculator()
calculator._calculate_regime_score('AAPL', risk_kpis)
