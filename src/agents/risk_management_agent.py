import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class RiskManagementAgent:
    """
    Specialized agent for portfolio risk assessment and management.
    Calculates VaR, CVaR, and monitors risk limits.
    """
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.max_portfolio_var = self.config.get('max_portfolio_var', 0.05)  # 5% max VaR
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.lookback_period = self.config.get('lookback_period', 252)  # 1 year

    def assess_portfolio_risk(self, positions: Dict[str, float], raw_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Assess the risk of the current or proposed portfolio.
        
        Args:
            positions: Dict mapping symbol to weight (0.0 to 1.0)
            raw_data: Dict mapping symbol to historical DataFrame (must contain 'close' or 'y')
            
        Returns:
            Dict containing risk metrics and approval status.
        """
        if not positions:
            return {'risk_approved': True, 'metrics': {}, 'reason': 'Empty portfolio'}

        # Align data
        returns_df = pd.DataFrame()
        for symbol, weight in positions.items():
            if weight > 0 and symbol in raw_data:
                df = raw_data[symbol].copy()
                if 'close' in df.columns:
                    price = df['close']
                elif 'y' in df.columns:
                    price = df['y']
                else:
                    continue
                
                # Calculate daily returns
                returns = price.pct_change().dropna()
                returns_df[symbol] = returns

        if returns_df.empty:
            return {'risk_approved': False, 'metrics': {}, 'reason': 'Insufficient data for risk calculation'}

        # Calculate Portfolio Returns
        weights = np.array([positions.get(col, 0) for col in returns_df.columns])
        weights = weights / np.sum(weights) # Normalize to 1 for calculation
        
        portfolio_returns = returns_df.dot(weights)
        
        # Calculate VaR (Value at Risk)
        var_95 = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        
        # Calculate CVaR (Conditional VaR / Expected Shortfall)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Calculate Volatility
        volatility = portfolio_returns.std() * np.sqrt(252) # Annualized
        
        metrics = {
            'VaR_95': float(var_95),
            'CVaR_95': float(cvar_95),
            'Annualized_Volatility': float(volatility),
            'Diversification_Ratio': self._calculate_diversification_ratio(returns_df, weights)
        }
        
        # Check limits
        # VaR is usually negative (loss), so we check if it's lower (more negative) than threshold
        # Or if we treat it as positive loss amount. Let's assume negative return.
        # If max_portfolio_var is 0.05 (5% loss), we check if var_95 < -0.05
        
        limit_breach = False
        reasons = []
        
        if var_95 < -self.max_portfolio_var:
            limit_breach = True
            reasons.append(f"VaR {var_95:.2%} exceeds limit {-self.max_portfolio_var:.2%}")
            
        if volatility > self.config.get('max_volatility', 0.30):
            limit_breach = True
            reasons.append(f"Volatility {volatility:.2%} exceeds limit {self.config.get('max_volatility', 0.30):.2%}")

        return {
            'risk_approved': not limit_breach,
            'metrics': metrics,
            'reason': "; ".join(reasons) if reasons else "Risk within limits"
        }

    def _calculate_diversification_ratio(self, returns_df: pd.DataFrame, weights: np.array) -> float:
        """
        Calculate the diversification ratio of the portfolio.
        Weighted average of volatilities / Portfolio volatility
        """
        asset_vols = returns_df.std()
        weighted_avg_vol = np.sum(asset_vols * weights)
        portfolio_vol = returns_df.dot(weights).std()
        
        if portfolio_vol == 0:
            return 0.0
            
        return float(weighted_avg_vol / portfolio_vol)
