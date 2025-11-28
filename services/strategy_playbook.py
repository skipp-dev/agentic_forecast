#!/usr/bin/env python3
"""
Strategy Playbook for regime-aware feature and model selection.
"""

import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.regime_agent import RegimeAgent
from src.config.config_loader import get_strategy_playbook

logger = logging.getLogger(__name__)

class StrategyPlaybook:
    """
    Regime-aware strategy playbook that selects features and models based on market conditions.

    This playbook implements conditional feature activation and model selection based on:
    - Interest rate regimes
    - Commodity price regimes
    - Seasonal patterns
    """

    def __init__(self, regime_agent: RegimeAgent = None, config_path: str = "config/strategy_config.yaml"):
        """Initialize the strategy playbook."""
        self.regime_agent = regime_agent or RegimeAgent()
        self.config_path = Path(project_root) / config_path

        # Default strategy configurations
        self.strategy_config = {
            'feature_groups': {
                'price_basic': ['price_returns', 'price_volatility', 'rolling_means'],
                'tech_basic': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands'],
                'seasonality_calendar': ['day_of_week', 'month_of_year', 'quarter'],
                'macro_rates': ['fed_funds_rate', 'rate_changes', 'rate_volatility'],
                'macro_labor': ['unemployment_rate', 'labor_changes'],
                'commodities_gold': ['gold_returns', 'gold_volatility', 'gold_trends'],
                'commodities_oil': ['oil_returns', 'oil_volatility', 'oil_trends'],
                'high_rate_features': ['rate_sensitive_momentum', 'defensive_assets'],
                'low_rate_features': ['growth_momentum', 'risk_assets'],
                'oil_spike_features': ['energy_correlation', 'inflation_hedges'],
                'oil_crash_features': ['energy_short', 'deflation_signals'],
                'gold_rally_features': ['safe_haven', 'inflation_protection'],
                'gold_selloff_features': ['risk_on', 'economic_growth'],
                'q1_features': ['winter_seasonal', 'tax_loss_harvest'],
                'q2_features': ['spring_rally', 'earnings_season'],
                'q3_features': ['summer_doldrums', 'back_to_school'],
                'q4_features': ['holiday_rally', 'year_end_positioning']
            },
            'model_strategies': {
                'default': ['naive', 'linear_regression', 'ridge_regression'],
                'high_rates': ['ridge_regression', 'random_forest', 'xgboost'],
                'low_rates': ['neural_network', 'lstm', 'xgboost'],
                'oil_spike': ['random_forest', 'xgboost', 'neural_network'],
                'oil_crash': ['linear_regression', 'ridge_regression', 'lstm'],
                'gold_rally': ['xgboost', 'random_forest', 'neural_network'],
                'gold_selloff': ['linear_regression', 'neural_network', 'lstm'],
                'q4': ['xgboost', 'neural_network', 'lstm']  # Year-end complexity
            }
        }

        # Load custom config using config_loader
        custom_config = get_strategy_playbook()
        self._merge_config(custom_config)

        logger.info("StrategyPlaybook initialized")

    def _merge_config(self, custom_config: Dict[str, Any]):
        """Merge custom configuration with defaults."""
        for key, value in custom_config.items():
            if key in self.strategy_config and isinstance(self.strategy_config[key], dict):
                self.strategy_config[key].update(value)
            else:
                self.strategy_config[key] = value

    def get_strategy_for_date(self, target_date: str) -> Dict[str, Any]:
        """
        Get complete strategy configuration for a target date.

        Args:
            target_date: Date string in YYYY-MM-DD format

        Returns:
            Dictionary with feature_groups and models to use
        """
        # Detect current regimes
        regimes = self.regime_agent.detect_regime(target_date)

        # Get feature selection
        feature_groups = self.regime_agent.get_regime_features(regimes)

        # Get model selection
        models = self._select_models(regimes)

        # Get specific features for active groups
        active_features = self._get_active_features(feature_groups)

        strategy = {
            'date': target_date,
            'regimes': regimes,
            'feature_groups': feature_groups,
            'active_features': active_features,
            'models': models,
            'confidence': self._calculate_strategy_confidence(regimes)
        }

        logger.info("Generated strategy for %s: %d features, %d models",
                   target_date, len(active_features), len(models))
        return strategy

    def _select_models(self, regimes: Dict[str, str]) -> List[str]:
        """Select appropriate models based on regimes."""
        selected_models = set(self.strategy_config['model_strategies']['default'])

        # Add regime-specific models
        for regime_type, regime_value in regimes.items():
            if regime_value in self.strategy_config['model_strategies']:
                selected_models.update(self.strategy_config['model_strategies'][regime_value])

        # Add advanced models for Phase 2
        selected_models.update(['neural_network', 'lstm'])

        return sorted(list(selected_models))

    def _get_active_features(self, feature_groups: Dict[str, bool]) -> List[str]:
        """Get list of active features based on enabled groups."""
        active_features = []

        for group_name, is_active in feature_groups.items():
            if is_active and group_name in self.strategy_config['feature_groups']:
                active_features.extend(self.strategy_config['feature_groups'][group_name])

        return sorted(list(set(active_features)))  # Remove duplicates

    def _calculate_strategy_confidence(self, regimes: Dict[str, str]) -> float:
        """Calculate confidence score for the strategy based on regime clarity."""
        known_regimes = sum(1 for regime in regimes.values() if regime != 'unknown')
        total_regimes = len(regimes)

        if total_regimes == 0:
            return 0.0

        # Base confidence on proportion of known regimes
        confidence = known_regimes / total_regimes

        # Boost confidence for extreme regimes
        extreme_regimes = ['high_rates', 'low_rates', 'oil_spike', 'oil_crash', 'gold_rally', 'gold_selloff']
        extreme_count = sum(1 for regime in regimes.values() if regime in extreme_regimes)

        if extreme_count > 0:
            confidence += 0.1 * extreme_count

        return min(confidence, 1.0)

    def validate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a strategy configuration for completeness and consistency.

        Args:
            strategy: Strategy dictionary to validate

        Returns:
            Validation results with any issues found
        """
        issues = []

        # Check required fields
        required_fields = ['regimes', 'feature_groups', 'active_features', 'models']
        for field in required_fields:
            if field not in strategy:
                issues.append(f"Missing required field: {field}")

        # Check regime validity
        if 'regimes' in strategy:
            expected_regime_types = ['rate_regime', 'oil_regime', 'gold_regime', 'seasonal_regime']
            for regime_type in expected_regime_types:
                if regime_type not in strategy['regimes']:
                    issues.append(f"Missing regime type: {regime_type}")

        # Check feature consistency
        if 'feature_groups' in strategy and 'active_features' in strategy:
            active_groups = [g for g, active in strategy['feature_groups'].items() if active]
            expected_features = set()
            for group in active_groups:
                if group in self.strategy_config['feature_groups']:
                    expected_features.update(self.strategy_config['feature_groups'][group])

            actual_features = set(strategy['active_features'])
            if not expected_features.issubset(actual_features):
                missing = expected_features - actual_features
                issues.append(f"Missing features for active groups: {missing}")

        # Check model availability
        if 'models' in strategy:
            available_models = ['naive', 'linear_regression', 'ridge_regression',
                              'random_forest', 'xgboost', 'neural_network', 'lstm']
            for model in strategy['models']:
                if model not in available_models:
                    issues.append(f"Unknown model: {model}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'strategy_summary': {
                'regime_count': len(strategy.get('regimes', {})),
                'feature_count': len(strategy.get('active_features', [])),
                'model_count': len(strategy.get('models', [])),
                'confidence': strategy.get('confidence', 0.0)
            }
        }