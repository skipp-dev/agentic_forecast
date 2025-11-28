#!/usr/bin/env python3
"""
Regime Detection Agent for identifying market regimes based on macro and commodity data.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.data.macro_data_agent import MacroDataAgent
from agents.data.commodity_data_agent import CommodityDataAgent

logger = logging.getLogger(__name__)

class RegimeAgent:
    """
    Agent for detecting market regimes based on macroeconomic and commodity data.

    Regimes detected:
    - Interest Rate Regimes: high_rates (>4%), low_rates (<2%), normal_rates (2-4%)
    - Oil Regimes: normal_oil, oil_spike (>10% increase), oil_crash (>10% decrease)
    - Gold Regimes: normal_gold, gold_rally (>5% increase), gold_selloff (>5% decrease)
    - Seasonal Regimes: q1, q2, q3, q4 (quarters)
    """

    def __init__(self, config_path: str = "config/regime_config.yaml"):
        """Initialize the regime detection agent."""
        self.config_path = Path(project_root) / config_path
        self.macro_agent = MacroDataAgent()
        self.commodity_agent = CommodityDataAgent()

        # Default regime thresholds
        self.regime_thresholds = {
            'rates': {
                'high': 0.04,  # 4%
                'low': 0.02    # 2%
            },
            'oil': {
                'spike': 0.10,  # 10% increase
                'crash': -0.10  # 10% decrease
            },
            'gold': {
                'rally': 0.05,  # 5% increase
                'selloff': -0.05 # 5% decrease
            }
        }

        # Load custom config if exists
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                self.regime_thresholds.update(custom_config.get('thresholds', {}))

        logger.info("RegimeAgent initialized with thresholds: %s", self.regime_thresholds)

    def detect_regime(self, target_date: str) -> Dict[str, str]:
        """
        Detect current market regime based on data up to target_date.

        Args:
            target_date: Date string in YYYY-MM-DD format

        Returns:
            Dictionary with regime classifications for each category
        """
        target_dt = pd.to_datetime(target_date)

        regimes = {}

        # Interest Rate Regime
        regimes['rate_regime'] = self._detect_rate_regime(target_dt)

        # Oil Regime
        regimes['oil_regime'] = self._detect_oil_regime(target_dt)

        # Gold Regime
        regimes['gold_regime'] = self._detect_gold_regime(target_dt)

        # Seasonal Regime
        regimes['seasonal_regime'] = self._detect_seasonal_regime(target_dt)

        logger.info("Detected regimes for %s: %s", target_date, regimes)
        return regimes

    def _detect_rate_regime(self, target_date: pd.Timestamp) -> str:
        """Detect interest rate regime."""
        try:
            # Load Fed Funds Rate data
            fed_data = self.macro_agent.load_macro_series('fed_funds_rate')
            if fed_data.empty:
                return 'unknown'

            # Get latest rate before target date
            recent_data = fed_data[fed_data.index <= target_date]
            if recent_data.empty:
                return 'unknown'

            current_rate = recent_data.iloc[-1]['fed_funds_rate'] / 100.0  # Convert from percent

            if current_rate > self.regime_thresholds['rates']['high']:
                return 'high_rates'
            elif current_rate < self.regime_thresholds['rates']['low']:
                return 'low_rates'
            else:
                return 'normal_rates'

        except Exception as e:
            logger.error("Error detecting rate regime: %s", e)
            return 'unknown'

    def _detect_oil_regime(self, target_date: pd.Timestamp) -> str:
        """Detect oil price regime."""
        try:
            # Load WTI Oil data
            oil_data = self.commodity_agent.load_commodity_series('wti_crude')
            if oil_data.empty:
                return 'unknown'

            # Get data for last 30 days before target date
            end_date = target_date
            start_date = end_date - timedelta(days=30)
            recent_data = oil_data[(oil_data.index >= start_date) & (oil_data.index <= end_date)]

            if len(recent_data) < 2:
                return 'unknown'

            # Calculate price change over the period
            start_price = recent_data.iloc[0]['wti_crude_close']
            end_price = recent_data.iloc[-1]['wti_crude_close']
            price_change = (end_price - start_price) / start_price

            if price_change > self.regime_thresholds['oil']['spike']:
                return 'oil_spike'
            elif price_change < self.regime_thresholds['oil']['crash']:
                return 'oil_crash'
            else:
                return 'normal_oil'

        except Exception as e:
            logger.error("Error detecting oil regime: %s", e)
            return 'unknown'

    def _detect_gold_regime(self, target_date: pd.Timestamp) -> str:
        """Detect gold price regime."""
        try:
            # Load Gold data
            gold_data = self.commodity_agent.load_commodity_series('gold_spot')
            if gold_data.empty:
                return 'unknown'

            # Get data for last 30 days before target date
            end_date = target_date
            start_date = end_date - timedelta(days=30)
            recent_data = gold_data[(gold_data.index >= start_date) & (gold_data.index <= end_date)]

            if len(recent_data) < 2:
                return 'unknown'

            # Calculate price change over the period
            start_price = recent_data.iloc[0]['gold_spot_close']
            end_price = recent_data.iloc[-1]['gold_spot_close']
            price_change = (end_price - start_price) / start_price

            if price_change > self.regime_thresholds['gold']['rally']:
                return 'gold_rally'
            elif price_change < self.regime_thresholds['gold']['selloff']:
                return 'gold_selloff'
            else:
                return 'normal_gold'

        except Exception as e:
            logger.error("Error detecting gold regime: %s", e)
            return 'unknown'

    def _detect_seasonal_regime(self, target_date: pd.Timestamp) -> str:
        """Detect seasonal regime based on quarter."""
        month = target_date.month
        if month <= 3:
            return 'q1'
        elif month <= 6:
            return 'q2'
        elif month <= 9:
            return 'q3'
        else:
            return 'q4'

    def get_regime_features(self, regimes: Dict[str, str]) -> Dict[str, bool]:
        """
        Get feature selection based on detected regimes.

        Args:
            regimes: Dictionary of detected regimes

        Returns:
            Dictionary with feature groups to activate
        """
        feature_groups = {
            'price_basic': True,       # Always include price features
            'tech_basic': True,        # Always include technical indicators
            'seasonality_calendar': True,  # Always include calendar features
            'macro_rates': True,       # Always include macro rates
            'macro_labor': True,       # Always include labor data
            'commodities_gold': True,  # Always include gold
            'commodities_oil': True,   # Always include oil
        }

        # Rate regime features
        rate_regime = regimes.get('rate_regime', 'unknown')
        if rate_regime == 'high_rates':
            feature_groups['high_rate_features'] = True
        elif rate_regime == 'low_rates':
            feature_groups['low_rate_features'] = True

        # Oil regime features
        oil_regime = regimes.get('oil_regime', 'unknown')
        if oil_regime == 'oil_spike':
            feature_groups['oil_spike_features'] = True
        elif oil_regime == 'oil_crash':
            feature_groups['oil_crash_features'] = True

        # Gold regime features
        gold_regime = regimes.get('gold_regime', 'unknown')
        if gold_regime == 'gold_rally':
            feature_groups['gold_rally_features'] = True
        elif gold_regime == 'gold_selloff':
            feature_groups['gold_selloff_features'] = True

        # Seasonal features
        seasonal_regime = regimes.get('seasonal_regime', 'unknown')
        feature_groups[f'{seasonal_regime}_features'] = True

        return feature_groups

    def get_regime_models(self, regimes: Dict[str, str]) -> List[str]:
        """
        Get recommended models based on detected regimes.

        Args:
            regimes: Dictionary of detected regimes

        Returns:
            List of recommended model names
        """
        models = ['naive', 'linear_regression']  # Base models

        # Add regime-specific models
        rate_regime = regimes.get('rate_regime', 'unknown')
        if rate_regime in ['high_rates', 'low_rates']:
            models.append('ridge_regression')

        oil_regime = regimes.get('oil_regime', 'unknown')
        if oil_regime in ['oil_spike', 'oil_crash']:
            models.append('random_forest')

        gold_regime = regimes.get('gold_regime', 'unknown')
        if gold_regime in ['gold_rally', 'gold_selloff']:
            models.append('xgboost')

        # Always include advanced models in Phase 2
        models.extend(['neural_network', 'lstm'])

        return list(set(models))  # Remove duplicates