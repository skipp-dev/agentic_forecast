import pandas as pd
import numpy as np
from ..graphs.state import GraphState
from typing import Dict, List, Union

class GuardrailAgent:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.absolute_anomaly_threshold = self.config.get('anomaly_threshold', 15)
        self.anomaly_percentage_threshold = self.config.get('anomaly_percentage_threshold', 0.20)  # 20%
        self.adaptive_threshold_min = self.config.get('adaptive_threshold_min', 3)
        self.adaptive_threshold_max = self.config.get('adaptive_threshold_max', 20)
        self.market_volatility_sensitivity = self.config.get('market_volatility_sensitivity', 1.5)

    def _extract_symbol(self, action: str) -> str:
        if " for " in action:
            symbol_segment = action.split(" for ")[-1].strip()
            if "(" in symbol_segment:
                symbol_segment = symbol_segment.split("(")[0].strip()
            return symbol_segment
        return action.strip()

    def _calculate_adaptive_threshold(self, data_size: int) -> int:
        """
        Calculate adaptive anomaly threshold based on data size.
        Larger datasets can tolerate more anomalies.
        """
        # Base threshold scales with log of data size
        adaptive_threshold = int(max(self.adaptive_threshold_min,
                                   min(self.adaptive_threshold_max,
                                       self.absolute_anomaly_threshold + (data_size // 100))))

        return adaptive_threshold

    def _assess_anomaly_risk(self, anomaly_count: int, data_size: int, market_volatility: str = 'medium') -> Dict[str, Union[str, float]]:
        """
        Assess the risk level of anomalies using a weighted scoring system.
        """
        if data_size == 0:
            return {'level': 'unknown', 'confidence': 0.0, 'percentage': 0}

        anomaly_percentage = anomaly_count / data_size
        adaptive_threshold = self._calculate_adaptive_threshold(data_size)

        volatility_multiplier = {
            'low': 1.2,
            'medium': 1.0,
            'high': 0.75
        }.get(market_volatility, 1.0)

        adjusted_absolute_threshold = adaptive_threshold * volatility_multiplier
        adjusted_percentage_threshold = self.anomaly_percentage_threshold * volatility_multiplier

        # Weighted scoring system
        score = 0
        # Score based on absolute count
        if anomaly_count > adjusted_absolute_threshold:
            score += 50 * (anomaly_count / adjusted_absolute_threshold)
        
        # Score based on percentage
        if anomaly_percentage > adjusted_percentage_threshold:
            score += 50 * (anomaly_percentage / adjusted_percentage_threshold)

        # Determine risk level based on the final score
        if score > 250:
            risk_level = 'critical'
            confidence = 0.95
        elif score > 200:
            risk_level = 'high'
            confidence = 0.8
        elif score > 100:
            risk_level = 'moderate'
            confidence = 0.65
        else:
            risk_level = 'low'
            confidence = 0.9

        if anomaly_count == 0:
            risk_level = 'none'
            confidence = 1.0
            
        return {
            'level': risk_level,
            'confidence': confidence,
            'percentage': anomaly_percentage,
            'score': score,
            'adaptive_threshold': adjusted_absolute_threshold,
            'volatility_adjustment': volatility_multiplier
        }

    def run(self, state: GraphState) -> Dict:
        """
        Vets the recommended actions against risk metrics and data quality with intelligent anomaly handling.
        """
        print("---")
        print("Guardrail agent is running (enhanced logic)...")

        recommended_actions = state.get('recommended_actions', [])
        risk_kpis = state.get('risk_kpis', {})
        anomalies = state.get('anomalies', {})
        raw_data = state.get('raw_data', {})

        high_risk_symbols = set()
        # Handle risk_kpis as dict instead of DataFrame
        if isinstance(risk_kpis, dict):
            for kpi_data in risk_kpis.values():
                if isinstance(kpi_data, dict) and kpi_data.get('risk_level') == 'High':
                    high_risk_symbols.add(kpi_data.get('symbol'))
        elif hasattr(risk_kpis, 'empty') and not risk_kpis.empty and 'risk_level' in risk_kpis.columns:
            # Fallback for DataFrame format if it exists
            high_risk_symbols = set(
                risk_kpis.loc[risk_kpis['risk_level'] == 'High', 'symbol']
            )

        vetted_actions: List[str] = []
        guardrail_entries: List[str] = []

        print(f"Evaluating {len(recommended_actions)} recommended actions with intelligent anomaly assessment...")

        # Assess market volatility from data patterns
        market_volatility = self._assess_market_volatility(raw_data)

        for action in recommended_actions:
            symbol = self._extract_symbol(action)

            # Risk Check (existing logic)
            if symbol in high_risk_symbols:
                entry = (
                    f"Guardrail blocked '{action}' because {symbol} has elevated risk."
                )
                guardrail_entries.append(entry)
                continue

            # Enhanced Anomaly Check
            if symbol in anomalies:
                anomaly_count = len(anomalies[symbol])
                data_size = len(raw_data.get(symbol, pd.DataFrame()))

                risk_assessment = self._assess_anomaly_risk(anomaly_count, data_size, market_volatility)

                if risk_assessment['level'] == 'critical':
                    entry = (
                        f"Guardrail blocked '{action}' for {symbol}: critical anomaly risk "
                        f"({anomaly_count} anomalies, {risk_assessment['percentage']:.1%} of data)."
                    )
                    guardrail_entries.append(entry)
                    continue
                elif risk_assessment['level'] == 'high':
                    # For high risk, allow but with warning
                    entry = (
                        f"⚠️  Guardrail warning for '{action}' on {symbol}: high anomaly risk "
                        f"({anomaly_count} anomalies, {risk_assessment['percentage']:.1%} of data). "
                        f"Proceeding with caution."
                    )
                    guardrail_entries.append(entry)
                    # Still allow the action but log the warning
                elif risk_assessment['level'] == 'moderate':
                    # Allow moderate anomalies
                    entry = (
                        f"✅ Guardrail approved '{action}' for {symbol} with moderate anomaly risk "
                        f"({anomaly_count} anomalies, {risk_assessment['percentage']:.1%} of data)."
                    )
                    guardrail_entries.append(entry)

            vetted_actions.append(action)

        if not vetted_actions:
            placeholder = "Hold and monitor (no safe promotions)"
            guardrail_entries.append(
                "Guardrail defaulted to a safe-hold because no vetted actions remained."
            )
            vetted_actions.append(placeholder)

        print("[OK] Guardrail agent finished with enhanced anomaly assessment.")
        return {
            "recommended_actions": vetted_actions,
            "guardrail_log": guardrail_entries,
        }

    def _assess_market_volatility(self, raw_data: Dict) -> str:
        """
        Assess overall market volatility based on price movements across symbols.
        """
        if not raw_data:
            return 'medium'
        
        volatility_scores = []
        
        for symbol, data in raw_data.items():
            if 'close' in data.columns and len(data) > 10:
                # Calculate price volatility (standard deviation of returns)
                returns = data['close'].pct_change().dropna()
                if len(returns) > 0:
                    volatility = returns.std()
                    volatility_scores.append(volatility)
        
        if not volatility_scores:
            return 'medium'
        
        avg_volatility = np.mean(volatility_scores)
        
        # Classify volatility
        if avg_volatility < 0.02:  # Low volatility
            return 'low'
        elif avg_volatility > 0.05:  # High volatility
            return 'high'
        else:
            return 'medium'
