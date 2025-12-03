import pandas as pd
import numpy as np
from typing import Dict, Any, List

class TrustScoreCalculator:
    """
    Calculates a deterministic Trust Score for each symbol/horizon.
    
    Trust Score Components:
    1. Forecast Accuracy (MAPE/SMAPE)
    2. Regime State (Normal vs Shock)
    3. Guardrail Flags
    4. Data Quality / Sanity
    """
    
    def __init__(self):
        # Weights for different components
        self.weights = {
            'accuracy': 0.4,
            'regime': 0.3,
            'guardrails': 0.2,
            'data_quality': 0.1
        }
        
        # Thresholds
        self.mape_threshold_good = 0.05
        self.mape_threshold_bad = 0.15
        
    def calculate_trust_scores(self, 
                             performance_summary: pd.DataFrame,
                             risk_kpis: Dict[str, Any],
                             guardrail_flags: List[Dict[str, Any]],
                             drift_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate trust scores for all symbols.
        
        Returns:
            Dict[symbol, float]: Trust score between 0.0 and 1.0
        """
        trust_scores = {}
        
        # Get list of all symbols from performance summary
        if performance_summary.empty:
            return {}
            
        symbols = performance_summary['symbol'].unique()
        
        for symbol in symbols:
            # 1. Accuracy Score (0.0 - 1.0)
            accuracy_score = self._calculate_accuracy_score(symbol, performance_summary)
            
            # 2. Regime Score (0.0 - 1.0)
            regime_score = self._calculate_regime_score(symbol, risk_kpis)
            
            # 3. Guardrail Score (0.0 - 1.0)
            guardrail_score = self._calculate_guardrail_score(symbol, guardrail_flags)
            
            # 4. Data Quality Score (0.0 - 1.0)
            data_score = self._calculate_data_score(symbol, drift_metrics)
            
            # Weighted Combination
            final_score = (
                accuracy_score * self.weights['accuracy'] +
                regime_score * self.weights['regime'] +
                guardrail_score * self.weights['guardrails'] +
                data_score * self.weights['data_quality']
            )
            
            # Clip to [0, 1] just in case
            trust_scores[symbol] = max(0.0, min(1.0, final_score))
            
        return trust_scores
    
    def _calculate_accuracy_score(self, symbol: str, performance_summary: pd.DataFrame) -> float:
        """
        Calculate score based on MAPE.
        MAPE < 5% -> 1.0
        MAPE > 15% -> 0.0
        Linear interpolation in between.
        """
        symbol_perf = performance_summary[performance_summary['symbol'] == symbol]
        if symbol_perf.empty:
            return 0.5 # Neutral if no data
            
        # Use the best model's MAPE
        best_mape = symbol_perf['mape'].min()
        
        if pd.isna(best_mape):
            return 0.0
            
        if best_mape <= self.mape_threshold_good:
            return 1.0
        elif best_mape >= self.mape_threshold_bad:
            return 0.0
        else:
            # Linear interpolation
            return 1.0 - (best_mape - self.mape_threshold_good) / (self.mape_threshold_bad - self.mape_threshold_good)

    def _calculate_regime_score(self, symbol: str, risk_kpis: Dict[str, Any]) -> float:
        """
        Calculate score based on market regime.
        Normal -> 1.0
        High Volatility / Shock -> 0.0 - 0.5
        """
        if symbol not in risk_kpis:
            return 0.5 # Default
            
        kpis = risk_kpis[symbol]
        volatility = kpis.get('volatility', 0.0)
        
        # Assuming volatility is annualized std dev. 
        # > 50% vol is high risk.
        if volatility > 0.5:
            return 0.2
        elif volatility > 0.3:
            return 0.5
        else:
            return 1.0

    def _calculate_guardrail_score(self, symbol: str, guardrail_flags: List[Dict[str, Any]]) -> float:
        """
        Calculate score based on active guardrails.
        No flags -> 1.0
        Active flags -> Penalize
        """
        # Filter flags for this symbol
        symbol_flags = [f for f in guardrail_flags if f.get('symbol') == symbol]
        
        if not symbol_flags:
            return 1.0
            
        # Simple penalty: 0.2 per flag
        penalty = len(symbol_flags) * 0.2
        return max(0.0, 1.0 - penalty)

    def _calculate_data_score(self, symbol: str, drift_metrics: Dict[str, Any]) -> float:
        """
        Calculate score based on data drift.
        Drift detected -> 0.0
        No drift -> 1.0
        """
        if symbol not in drift_metrics:
            return 1.0
            
        metrics = drift_metrics[symbol]
        if metrics.get('drift_detected', False):
            return 0.0
        return 1.0
