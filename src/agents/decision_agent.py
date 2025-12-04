import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional
from src.data.model_registry import ModelRegistry

class DecisionAgent:
    """
    Enhanced agent for making intelligent decisions based on comprehensive model performance analysis.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.trigger_mape_threshold = self.config.get('trigger_mape_threshold', 0.1)
        self.min_performance_improvement = self.config.get('min_performance_improvement', 0.05)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Enhanced parameters
        self.model_diversity_weight = self.config.get('model_diversity_weight', 0.1)
        self.risk_adjustment_factor = self.config.get('risk_adjustment_factor', 0.05)
        self.market_regime_sensitivity = self.config.get('market_regime_sensitivity', 0.8)
        
        # Initialize Model Registry
        self.model_registry = ModelRegistry()

    def select_best_model(self, performance_summary: pd.DataFrame, anomalies: Dict = None) -> Dict[str, Dict[str, any]]:
        """
        Selects the best model for each symbol based on comprehensive performance metrics.
        Returns detailed selection info including confidence scores.
        """
        if performance_summary.empty:
            return {}

        best_models = {}

        for symbol in performance_summary['symbol'].unique():
            symbol_performance = performance_summary[performance_summary['symbol'] == symbol].copy()

            if symbol_performance.empty:
                continue

            # Calculate composite score considering multiple metrics
            symbol_performance['composite_score'] = self._calculate_composite_score(symbol_performance, anomalies, symbol)

            # Handle cases where all scores are NaN
            if symbol_performance['composite_score'].isnull().all():
                # Fallback logic: select the model with the lowest MAPE if available
                if not symbol_performance['mape'].isnull().all():
                    best_idx = symbol_performance['mape'].idxmin()
                else:
                    # If all else fails, choose the first model
                    best_idx = symbol_performance.index[0]
            else:
                best_idx = symbol_performance['composite_score'].idxmax()
            
            best_model = symbol_performance.loc[best_idx]

            # Calculate confidence in the selection
            confidence = self._calculate_selection_confidence(symbol_performance, best_model)

            best_models[symbol] = {
                'model_family': best_model['model_family'],
                'mape': best_model['mape'],
                'composite_score': best_model['composite_score'],
                'confidence': confidence,
                'model_id': best_model.get('model_id'),
                'alternative_models': self._get_alternative_models(symbol_performance, best_model)
            }

        return best_models

    def _calculate_composite_score(self, performance_df: pd.DataFrame, anomalies: Dict = None, symbol: str = None) -> pd.Series:
        """
        Calculate a composite score considering MAPE, model diversity, and risk factors.
        Lower MAPE is better, so we invert it for scoring.
        """
        # Base score from inverted MAPE (lower MAPE = higher score)
        mape_scores = 1 / (1 + performance_df['mape'])  # Add 1 to avoid division by zero
        
        # Model diversity bonus (prefer different architectures)
        model_families = performance_df['model_family'].tolist()
        diversity_score = self._calculate_model_diversity_score(model_families)
        diversity_score.index = performance_df.index  # Align indices
        
        # Risk adjustment based on anomalies
        risk_adjustment = 0.0
        if anomalies and symbol and symbol in anomalies:
            anomaly_count = len(anomalies[symbol])
            # Penalize models when there are many anomalies (potential overfitting)
            risk_adjustment = -min(anomaly_count * self.risk_adjustment_factor, 0.2)
        
        # Combine scores
        composite_scores = (
            mape_scores * 0.8 +  # 80% weight on accuracy
            diversity_score * self.model_diversity_weight +  # Diversity bonus
            (1 + risk_adjustment) * 0.1  # Risk adjustment
        )
        
        return composite_scores

    def _calculate_model_diversity_score(self, model_families: List[str]) -> pd.Series:
        """
        Calculate diversity score based on model architecture variety.
        """
        # Define model categories for diversity calculation
        model_categories = {
            'linear': ['BaselineLinear', 'NLinear', 'DLinear'],
            'neural': ['AutoNHITS', 'AutoNBEATS', 'AutoTFT'],
            'deep': ['CNNLSTM', 'LSTM', 'BiTCN'],
            'ensemble': ['Ensemble']
        }
        
        diversity_scores = []
        for family in model_families:
            category = None
            for cat, models in model_categories.items():
                if family in models:
                    category = cat
                    break
            
            # Score based on category uniqueness (higher for unique categories)
            if category:
                category_count = sum(1 for f in model_families if any(f in models for models in model_categories.values() if f in models))
                diversity_scores.append(1.0 / (1 + category_count))  # Higher score for rarer categories
            else:
                diversity_scores.append(0.5)  # Default score
        
        return pd.Series(diversity_scores, index=range(len(model_families)))

    def _calculate_selection_confidence(self, performance_df: pd.DataFrame, best_model: pd.Series) -> float:
        """
        Calculate confidence in the model selection based on performance distribution.
        """
        if len(performance_df) <= 1:
            return 0.5  # Low confidence with only one model

        best_score = best_model['composite_score']
        mean_score = performance_df['composite_score'].mean()
        std_score = performance_df['composite_score'].std()

        if std_score == 0:
            return 1.0  # Perfect confidence if all models have same score

        # Confidence based on how much better the best model is compared to others
        z_score = (best_score - mean_score) / std_score

        # Convert z-score to confidence (sigmoid-like function)
        confidence = 1 / (1 + np.exp(-z_score))

        return min(confidence, 0.95)  # Cap at 95%

    def _get_alternative_models(self, performance_df: pd.DataFrame, best_model: pd.Series) -> List[Dict]:
        """
        Get alternative models ranked by performance.
        """
        alternatives = []
        sorted_df = performance_df.sort_values('composite_score', ascending=False)

        for idx, row in sorted_df.iterrows():
            if row['model_family'] != best_model['model_family']:
                alternatives.append({
                    'model_family': row['model_family'],
                    'mape': row['mape'],
                    'composite_score': row['composite_score']
                })

        return alternatives[:3]  # Top 3 alternatives

    def should_run_hpo(self, performance_summary: pd.DataFrame, anomalies: Dict = None, market_conditions: Dict = None, raw_data: Dict = None) -> Dict[str, any]:
        """
        Enhanced HPO triggering logic considering performance, anomalies, market conditions, and age.
        """
        # Check age-based trigger first
        if raw_data:
            for symbol in raw_data.keys():
                last_run = self.model_registry.get_last_hpo_run(symbol)
                if last_run is None:
                    return {
                        'should_run': True,
                        'reason': f"Initial HPO run required for {symbol}",
                        'confidence': 1.0,
                        'target_symbols': [symbol]
                    }
                
                max_age_seconds = 7 * 24 * 3600 # 7 days
                if (time.time() - last_run) > max_age_seconds:
                    return {
                        'should_run': True,
                        'reason': f"HPO age limit exceeded for {symbol} (Last run: {time.ctime(last_run)})",
                        'confidence': 1.0,
                        'target_symbols': [symbol]
                    }

        if performance_summary.empty:
            return {
                'should_run': False,
                'reason': 'No performance data available',
                'confidence': 0.0
            }

        # Analyze performance degradation
        performance_issues = self._analyze_performance_issues(performance_summary)
        
        # Analyze anomaly patterns
        anomaly_issues = self._analyze_anomaly_patterns(anomalies, raw_data) if anomalies and raw_data else []
        
        # Consider market regime
        market_pressure = self._assess_market_pressure(market_conditions)
        
        # Combine all factors
        all_issues = performance_issues + anomaly_issues
        if market_pressure:
            all_issues.append(market_pressure)
        
        if all_issues:
            # Prioritize issues by severity
            severe_issues = [issue for issue in all_issues if issue['severity'] == 'high']
            moderate_issues = [issue for issue in all_issues if issue['severity'] == 'medium']
            
            if severe_issues:
                return {
                    'should_run': True,
                    'reason': f"Critical issues detected: {[issue['description'] for issue in severe_issues]}",
                    'confidence': 0.9,
                    'target_symbols': list(set([issue['symbol'] for issue in severe_issues]))
                }
            elif moderate_issues:
                return {
                    'should_run': True,
                    'reason': f"Performance optimization opportunities: {[issue['description'] for issue in moderate_issues]}",
                    'confidence': 0.7,
                    'target_symbols': list(set([issue['symbol'] for issue in moderate_issues]))
                }
        
        return {
            'should_run': False,
            'reason': 'All models performing adequately with stable market conditions',
            'confidence': 0.8
        }

    def _analyze_performance_issues(self, performance_summary: pd.DataFrame) -> List[Dict]:
        """Analyze performance issues across symbols."""
        issues = []
        
        for symbol in performance_summary['symbol'].unique():
            symbol_performance = performance_summary[performance_summary['symbol'] == symbol]
            best_mape = symbol_performance['mape'].min()
            avg_mape = symbol_performance['mape'].mean()
            
            # Check for poor absolute performance
            if best_mape > self.trigger_mape_threshold:
                issues.append({
                    'symbol': symbol,
                    'description': f"Poor accuracy (MAPE: {best_mape:.3f})",
                    'severity': 'high'
                })
            
            # Check for high performance variance (inconsistent models)
            elif symbol_performance['mape'].std() > 0.05:
                issues.append({
                    'symbol': symbol,
                    'description': f"High model performance variance (std: {symbol_performance['mape'].std():.3f})",
                    'severity': 'medium'
                })
        
        return issues

    def _analyze_anomaly_patterns(self, anomalies: Dict, raw_data: Dict) -> List[Dict]:
        """Analyze anomaly patterns that might indicate model issues."""
        issues = []
        
        if not raw_data:
            return issues

        for symbol, anomaly_df in anomalies.items():
            if symbol not in raw_data or raw_data[symbol].empty:
                continue

            anomaly_count = len(anomaly_df)
            total_points = len(raw_data[symbol])
            anomaly_rate = anomaly_count / total_points if total_points > 0 else 0
            
            if anomaly_rate > 0.3:  # More than 30% anomalies
                issues.append({
                    'symbol': symbol,
                    'description': f"Critical anomaly rate ({anomaly_rate:.1%}) indicates potential model failure",
                    'severity': 'high'
                })
            elif anomaly_rate > 0.15:  # More than 15% anomalies
                issues.append({
                    'symbol': symbol,
                    'description': f"High anomaly rate ({anomaly_rate:.1%}) suggests model adaptation needed",
                    'severity': 'medium'
                })
        
        return issues

    def _assess_market_pressure(self, market_conditions: Dict = None) -> Optional[Dict]:
        """Assess if market conditions warrant HPO."""
        if not market_conditions:
            return None
            
        volatility = market_conditions.get('volatility', 'medium')
        if volatility == 'high':
            return {
                'symbol': 'all',
                'description': f"High market volatility ({volatility}) requires model adaptation",
                'severity': 'medium'
            }
        
        return None

    def should_promote_model(self, symbol: str, model_performance: Dict, market_conditions: Dict = None) -> Dict[str, any]:
        """
        Determines if a model should be promoted to production based on performance and conditions.
        """
        mape = model_performance.get('mape', float('inf'))
        confidence = model_performance.get('confidence', 0.0)

        # Base criteria
        meets_accuracy = mape <= self.trigger_mape_threshold
        meets_confidence = confidence >= self.confidence_threshold

        # Consider market conditions (if provided)
        market_risk = market_conditions.get('volatility', 'medium') if market_conditions else 'medium'
        market_adjustment = {'low': 1.2, 'medium': 1.0, 'high': 0.8}[market_risk]

        adjusted_threshold = self.trigger_mape_threshold * market_adjustment

        meets_adjusted_accuracy = mape <= adjusted_threshold

        if meets_accuracy and meets_confidence:
            return {
                'should_promote': True,
                'reason': f"Model meets accuracy ({mape:.3f} <= {self.trigger_mape_threshold}) and confidence ({confidence:.2f}) thresholds",
                'confidence': confidence
            }
        elif meets_adjusted_accuracy and meets_confidence:
            return {
                'should_promote': True,
                'reason': f"Model meets market-adjusted accuracy threshold ({mape:.3f} <= {adjusted_threshold:.3f}) under {market_risk} volatility",
                'confidence': confidence * 0.9  # Slight penalty for market adjustment
            }
        else:
            reasons = []
            if not meets_adjusted_accuracy:
                reasons.append(f"Accuracy {mape:.3f} above threshold {adjusted_threshold:.3f}")
            if not meets_confidence:
                reasons.append(f"Confidence {confidence:.2f} below threshold {self.confidence_threshold}")

            return {
                'should_promote': False,
                'reason': '; '.join(reasons),
                'confidence': confidence
            }

    def get_trading_decision(self, symbol: str, trust_score: float, forecast_confidence: str = 'low', regimes: Optional[Dict[str, str]] = None) -> Dict[str, any]:
        """
        Determine trading permissions and sizing based on trust score and market regimes.
        
        Trust Score Ranges:
        - 0.0 - 0.3: Do not auto-trade
        - 0.3 - 0.6: Small size only
        - 0.6 - 1.0: Normal size allowed
        """
        decision = {
            'symbol': symbol,
            'trust_score': trust_score,
            'auto_trade_allowed': False,
            'position_size_multiplier': 0.0,
            'reason': ''
        }
        
        if trust_score < 0.3:
            decision['auto_trade_allowed'] = False
            decision['position_size_multiplier'] = 0.0
            decision['reason'] = f"Trust score too low ({trust_score:.2f} < 0.3)"
            
        elif trust_score < 0.6:
            decision['auto_trade_allowed'] = True
            decision['position_size_multiplier'] = 0.5
            decision['reason'] = f"Moderate trust score ({trust_score:.2f}); reduced sizing"
            
        else:
            decision['auto_trade_allowed'] = True
            decision['position_size_multiplier'] = 1.0
            decision['reason'] = f"High trust score ({trust_score:.2f}); normal sizing"
            
        # Additional check: if forecast confidence is explicitly 'low', cap multiplier
        if forecast_confidence == 'low' and decision['position_size_multiplier'] > 0.5:
            decision['position_size_multiplier'] = 0.5
            decision['reason'] += " (Capped due to low forecast confidence)"

        # Regime-based adjustments
        if regimes and decision['auto_trade_allowed']:
            # High Rates Regime
            if regimes.get('rate_regime') == 'high_rates':
                decision['position_size_multiplier'] *= 0.8
                decision['reason'] += " (Reduced 20% due to High Rates regime)"
            
            # Oil Spike Regime
            if regimes.get('oil_regime') == 'spike':
                decision['position_size_multiplier'] *= 0.7
                decision['reason'] += " (Reduced 30% due to Oil Spike regime)"
                
            # Gold Rally (Risk-off signal)
            if regimes.get('gold_regime') == 'rally':
                decision['position_size_multiplier'] *= 0.9
                decision['reason'] += " (Reduced 10% due to Gold Rally/Risk-off)"
                
            # Seasonal Sell-off
            if regimes.get('seasonal_regime') == 'sell_in_may':
                decision['position_size_multiplier'] *= 0.8
                decision['reason'] += " (Reduced 20% due to Seasonal weakness)"
            
        return decision
