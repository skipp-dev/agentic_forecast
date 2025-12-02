"""
Strategy Selection Agent

Creates regime-aware trading strategies and selects optimal strategies based on:
- Current market regime
- Historical performance under similar regimes
- Risk-adjusted returns
- Cross-asset correlations
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

@dataclass
class Strategy:
    """Represents a trading strategy with its characteristics."""
    name: str
    description: str
    regime_affinity: Dict[str, float]  # Which regimes this strategy works best in
    risk_level: str  # 'low', 'medium', 'high'
    expected_return: float
    expected_volatility: float
    max_drawdown: float
    parameters: Dict[str, Any]

class StrategySelectionAgent:
    """
    Agent responsible for creating and selecting optimal trading strategies
    based on current market regimes and historical performance.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Define base strategies for different regimes
        self.base_strategies = self._define_base_strategies()

        # Strategy performance history
        self.performance_history = {}

    def _define_base_strategies(self) -> Dict[str, Strategy]:
        """Define the base set of trading strategies."""
        strategies = {}

        # Rate Easing Strategies (bullish environment)
        strategies['momentum_growth'] = Strategy(
            name='momentum_growth',
            description='Long momentum stocks with growth characteristics',
            regime_affinity={'rate_regime_easing': 0.9, 'labor_regime_expansion': 0.8},
            risk_level='medium',
            expected_return=0.12,
            expected_volatility=0.18,
            max_drawdown=0.15,
            parameters={'lookback_period': 252, 'momentum_threshold': 0.1}
        )

        # Rate Tightening Strategies (defensive)
        strategies['defensive_dividend'] = Strategy(
            name='defensive_dividend',
            description='High dividend yield defensive stocks',
            regime_affinity={'rate_regime_tightening': 0.9, 'commodity_regime_bear': 0.7},
            risk_level='low',
            expected_return=0.06,
            expected_volatility=0.12,
            max_drawdown=0.08,
            parameters={'min_dividend_yield': 0.03, 'max_beta': 0.8}
        )

        # Commodity Bull Strategies
        strategies['commodity_sensitive'] = Strategy(
            name='commodity_sensitive',
            description='Stocks sensitive to commodity price movements',
            regime_affinity={'commodity_regime_bull': 0.9, 'labor_regime_expansion': 0.6},
            risk_level='high',
            expected_return=0.15,
            expected_volatility=0.25,
            max_drawdown=0.20,
            parameters={'commodity_beta_threshold': 0.5}
        )

        # Labor Contraction Strategies (recession-resistant)
        strategies['recession_resistant'] = Strategy(
            name='recession_resistant',
            description='Consumer staples and healthcare stocks',
            regime_affinity={'labor_regime_contraction': 0.9, 'rate_regime_neutral': 0.7},
            risk_level='low',
            expected_return=0.05,
            expected_volatility=0.10,
            max_drawdown=0.06,
            parameters={'sector_focus': ['consumer_staples', 'healthcare']}
        )

        # Seasonal Strategies
        strategies['winter_defensive'] = Strategy(
            name='winter_defensive',
            description='Defensive positioning for winter months',
            regime_affinity={'seasonal_regime_winter': 0.8},
            risk_level='low',
            expected_return=0.04,
            expected_volatility=0.08,
            max_drawdown=0.05,
            parameters={'cash_allocation': 0.3}
        )

        strategies['summer_growth'] = Strategy(
            name='summer_growth',
            description='Growth-oriented for summer months',
            regime_affinity={'seasonal_regime_summer': 0.8},
            risk_level='medium',
            expected_return=0.10,
            expected_volatility=0.16,
            max_drawdown=0.12,
            parameters={'growth_focus': True}
        )

        # Cross-asset strategies
        strategies['btc_correlation'] = Strategy(
            name='btc_correlation',
            description='Stocks correlated with Bitcoin performance',
            regime_affinity={'clustered_regime_regime_1': 0.7},  # Assuming regime_1 is crypto-friendly
            risk_level='high',
            expected_return=0.18,
            expected_volatility=0.30,
            max_drawdown=0.25,
            parameters={'btc_correlation_threshold': 0.3}
        )

        return strategies

    def select_strategies(self, current_regimes: Dict[str, str],
                         historical_performance: Optional[Dict[str, pd.DataFrame]] = None,
                         risk_tolerance: str = 'medium') -> List[Dict[str, Any]]:
        """
        Select optimal strategies based on current regimes and historical performance.

        Args:
            current_regimes: Current market regime classifications
            historical_performance: Historical performance data by regime
            risk_tolerance: Risk tolerance level ('low', 'medium', 'high')

        Returns:
            List of selected strategies with scores and weights
        """
        logger.info(f"Selecting strategies for regimes: {current_regimes}")

        strategy_scores = {}

        # Score each strategy based on regime affinity
        for strategy_name, strategy in self.base_strategies.items():
            score = self._calculate_strategy_score(strategy, current_regimes, historical_performance)
            strategy_scores[strategy_name] = {
                'strategy': strategy,
                'score': score,
                'risk_level': strategy.risk_level
            }

        # Filter by risk tolerance
        risk_hierarchy = {'low': 1, 'medium': 2, 'high': 3}
        max_risk_level = risk_hierarchy.get(risk_tolerance, 2)

        filtered_strategies = {
            name: data for name, data in strategy_scores.items()
            if risk_hierarchy.get(data['risk_level'], 3) <= max_risk_level
        }

        # Sort by score and select top strategies
        sorted_strategies = sorted(filtered_strategies.items(),
                                 key=lambda x: x[1]['score'], reverse=True)

        # Select top 3-5 strategies and normalize weights
        selected_count = min(5, len(sorted_strategies))
        selected_strategies = sorted_strategies[:selected_count]

        # Calculate weights based on scores
        total_score = sum(data['score'] for _, data in selected_strategies)
        strategy_weights = []

        for strategy_name, data in selected_strategies:
            weight = data['score'] / total_score if total_score > 0 else 1.0 / selected_count
            strategy_weights.append({
                'strategy_name': strategy_name,
                'strategy': data['strategy'],
                'score': data['score'],
                'weight': weight,
                'regime_match': self._get_regime_match(data['strategy'], current_regimes)
            })

        logger.info(f"Selected {len(strategy_weights)} strategies")
        return strategy_weights

    def _calculate_strategy_score(self, strategy: Strategy, current_regimes: Dict[str, str],
                                historical_performance: Optional[Dict[str, pd.DataFrame]] = None) -> float:
        """
        Calculate a score for a strategy based on regime affinity and historical performance.
        """
        base_score = 0.0

        # Regime affinity score
        for regime_type, regime_value in current_regimes.items():
            regime_key = f"{regime_type}_{regime_value}"
            affinity = strategy.regime_affinity.get(regime_key, 0.0)
            base_score += affinity * 0.4  # 40% weight on regime affinity

        # Historical performance score (if available)
        if historical_performance:
            perf_score = self._calculate_performance_score(strategy, current_regimes, historical_performance)
            base_score += perf_score * 0.6  # 60% weight on historical performance

        return base_score

    def _calculate_performance_score(self, strategy: Strategy, current_regimes: Dict[str, str],
                                   historical_performance: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate performance score based on historical data.
        """
        # This would use actual historical performance data
        # For now, return a placeholder score
        return 0.5

    def _get_regime_match(self, strategy: Strategy, current_regimes: Dict[str, str]) -> List[str]:
        """
        Get list of regimes this strategy matches.
        """
        matches = []
        for regime_type, regime_value in current_regimes.items():
            regime_key = f"{regime_type}_{regime_value}"
            if strategy.regime_affinity.get(regime_key, 0.0) > 0.5:
                matches.append(regime_key)
        return matches

    def create_regime_playbook(self, regimes: Dict[str, pd.Series],
                             historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a comprehensive strategy playbook for different regime combinations.

        Args:
            regimes: Historical regime classifications
            historical_data: Historical market data for backtesting

        Returns:
            Strategy playbook with recommended strategies for each regime
        """
        logger.info("Creating regime strategy playbook")

        playbook = {}

        # Analyze each regime combination
        regime_combinations = self._get_regime_combinations(regimes)

        for combo_name, combo_mask in regime_combinations.items():
            # Get historical performance during this regime
            regime_data = historical_data[combo_mask]

            if not regime_data.empty:
                # Find best performing strategies for this regime
                best_strategies = self._find_best_strategies_for_regime(combo_name, regime_data)

                playbook[combo_name] = {
                    'regime_description': combo_name,
                    'recommended_strategies': best_strategies,
                    'historical_periods': len(regime_data),
                    'avg_performance': regime_data.mean(),
                    'volatility': regime_data.std()
                }

        return playbook

    def _get_regime_combinations(self, regimes: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Get unique regime combinations and their masks.
        """
        # Combine regime series into a single multi-index series
        combined_regimes = pd.DataFrame(regimes)

        # Create combination labels
        combination_labels = combined_regimes.apply(
            lambda row: '_'.join([f"{col}_{row[col]}" for col in combined_regimes.columns]), axis=1
        )

        # Group by combination
        combinations = {}
        for combo in combination_labels.unique():
            mask = combination_labels == combo
            combinations[combo] = mask

        return combinations

    def _find_best_strategies_for_regime(self, regime_name: str, regime_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find best performing strategies for a specific regime.
        """
        # This would implement backtesting logic
        # For now, return placeholder based on regime characteristics
        best_strategies = []

        if 'easing' in regime_name:
            best_strategies.append({
                'strategy': 'momentum_growth',
                'expected_return': 0.12,
                'confidence': 0.8
            })
        elif 'tightening' in regime_name:
            best_strategies.append({
                'strategy': 'defensive_dividend',
                'expected_return': 0.06,
                'confidence': 0.9
            })
        elif 'bull' in regime_name and 'commodity' in regime_name:
            best_strategies.append({
                'strategy': 'commodity_sensitive',
                'expected_return': 0.15,
                'confidence': 0.7
            })

        return best_strategies

    def get_strategy_recommendations(self, current_regimes: Dict[str, str],
                                   risk_tolerance: str = 'medium') -> Dict[str, Any]:
        """
        Main method to get strategy recommendations based on current market conditions.

        Args:
            current_regimes: Current market regime classifications
            risk_tolerance: Risk tolerance level

        Returns:
            Strategy recommendations and analysis
        """
        # Select optimal strategies
        selected_strategies = self.select_strategies(current_regimes, risk_tolerance=risk_tolerance)

        # Get current regime context
        regime_context = self._analyze_regime_context(current_regimes)

        return {
            'selected_strategies': selected_strategies,
            'regime_context': regime_context,
            'risk_tolerance': risk_tolerance,
            'recommendation_timestamp': datetime.now(),
            'total_weight': sum(s['weight'] for s in selected_strategies)
        }

    def _analyze_regime_context(self, current_regimes: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze the current regime context for additional insights.
        """
        context = {
            'regime_stability': 'unknown',
            'risk_environment': 'neutral',
            'market_bias': 'neutral'
        }

        # Analyze rate regime
        if current_regimes.get('rate_regime') == 'easing':
            context['risk_environment'] = 'bullish'
            context['market_bias'] = 'growth'
        elif current_regimes.get('rate_regime') == 'tightening':
            context['risk_environment'] = 'bearish'
            context['market_bias'] = 'defensive'

        # Analyze labor regime
        if current_regimes.get('labor_regime') == 'contraction':
            context['risk_environment'] = 'cautious'

        return context</content>
<parameter name="filePath">c:\Users\spreu\Documents\agentic_forecast\agents\strategy_selection_agent.py