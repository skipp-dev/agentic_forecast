#!/usr/bin/env python3
"""
Regime Detection Agent for identifying market regimes based on cross-asset features.

Updated to work with the new cross-asset data architecture.
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

from src.data.cross_asset_features import CrossAssetFeatures

logger = logging.getLogger(__name__)

class RegimeAgent:
    """
    Agent for detecting market regimes based on cross-asset features.

    Regimes detected:
    - Crypto Regimes: boom, crash, normal
    - FX Regimes: stress, normal
    - Equity Risk Regimes: risk_off, risk_on, neutral
    - Combined Market Regime: overall market assessment
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the regime detection agent."""
        self.config = config or {}
        self.cross_asset = CrossAssetFeatures(config=config)

        # Regime thresholds
        self.regime_thresholds = {
            'crypto': {
                'crash_drawdown': -0.2,  # 20% drawdown
                'boom_trend': 0.1,       # 10% uptrend
                'vol_percentile': 0.8    # 80th percentile for high vol
            },
            'fx': {
                'stress_move': 0.005,    # 0.5% daily move
                'vol_multiplier': 1.5    # 1.5x average vol
            },
            'equity': {
                'risk_off_trend': -0.05,  # 5% downtrend
                'risk_on_trend': 0.03,    # 3% uptrend
                'high_vol_percentile': 0.8
            }
        }

        # Data storage
        self.regimes_path = Path('data/features/regimes')
        self.regimes_path.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized Regime Detection Agent")

    def detect_regimes_from_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect regimes from cross-asset features DataFrame.

        Args:
            features_df: DataFrame with cross-asset features

        Returns:
            DataFrame with regime classifications
        """
        logger.info("Detecting regimes from cross-asset features...")

        regimes_df = pd.DataFrame(index=features_df.index)

        # Crypto regime
        regimes_df['crypto_regime'] = self._detect_crypto_regime(features_df)

        # FX regime
        regimes_df['fx_regime'] = self._detect_fx_regime(features_df)

        # Equity risk regime
        regimes_df['equity_risk_regime'] = self._detect_equity_regime(features_df)

        # Combined market regime
        regimes_df['market_regime'] = self._detect_combined_regime(regimes_df)

        logger.info(f"Detected regimes for {len(regimes_df)} periods")
        return regimes_df

    def _detect_crypto_regime(self, features_df: pd.DataFrame) -> pd.Series:
        """Detect crypto market regime."""
        # Use BTC features for crypto regime
        btc_trend = features_df.get('btc_trend_20d', 0)
        btc_vol = features_df.get('btc_vol_30d', 0)
        btc_drawdown = features_df.get('btc_drawdown_30d', 0)

        thresholds = self.regime_thresholds['crypto']

        conditions = [
            (btc_drawdown < thresholds['crash_drawdown']) & (btc_trend < -abs(thresholds['boom_trend'])),
            (btc_trend > thresholds['boom_trend']) & (btc_vol < btc_vol.quantile(thresholds['vol_percentile'])),
            (btc_vol > btc_vol.quantile(thresholds['vol_percentile'])),
        ]
        choices = ['crash', 'boom', 'volatile']
        default = 'normal'

        return pd.Series(np.select(conditions, choices, default=default), index=features_df.index)

    def _detect_fx_regime(self, features_df: pd.DataFrame) -> pd.Series:
        """Detect FX market regime."""
        # Use EURUSD features for FX regime
        eurusd_vol = features_df.get('eurusd_vol_20d', 0)
        fx_stress = features_df.get('fx_stress_flag', 0)
        fx_vol_spike = features_df.get('fx_vol_spike_flag', 0)

        thresholds = self.regime_thresholds['fx']

        conditions = [
            (fx_stress == 1) | (fx_vol_spike == 1),
            (eurusd_vol > eurusd_vol.rolling(50).mean() * thresholds['vol_multiplier']),
        ]
        choices = ['stress', 'high_vol']
        default = 'normal'

        return pd.Series(np.select(conditions, choices, default=default), index=features_df.index)

    def _detect_equity_regime(self, features_df: pd.DataFrame) -> pd.Series:
        """Detect equity risk regime."""
        # Use SPY features for equity regime (assuming SPY data is available)
        spy_trend = features_df.get('market_ret_1d', 0).rolling(20).mean()  # Approximate trend
        spy_vol = features_df.get('vol_10d', 0)

        # If SPY-specific features exist, use them
        if 'market_vol_10d' in features_df.columns:
            spy_vol = features_df['market_vol_10d']

        thresholds = self.regime_thresholds['equity']

        conditions = [
            (spy_trend < thresholds['risk_off_trend']) & (spy_vol > spy_vol.quantile(thresholds['high_vol_percentile'])),
            (spy_trend > thresholds['risk_on_trend']) & (spy_vol < spy_vol.quantile(thresholds['high_vol_percentile'])),
        ]
        choices = ['risk_off', 'risk_on']
        default = 'neutral'

        return pd.Series(np.select(conditions, choices, default=default), index=features_df.index)

    def _detect_combined_regime(self, regimes_df: pd.DataFrame) -> pd.Series:
        """Detect combined market regime from individual regimes."""
        # Simple combination logic
        crypto_regime = regimes_df['crypto_regime']
        fx_regime = regimes_df['fx_regime']
        equity_regime = regimes_df['equity_risk_regime']

        # Risk assessment based on combination
        risk_conditions = [
            (crypto_regime == 'crash') | (equity_regime == 'risk_off') | (fx_regime == 'stress'),
            (crypto_regime == 'boom') & (equity_regime == 'risk_on') & (fx_regime == 'normal'),
            (crypto_regime == 'volatile') | (fx_regime == 'high_vol'),
        ]
        risk_choices = ['high_risk', 'low_risk', 'moderate_risk']
        default = 'normal'

        return pd.Series(np.select(risk_conditions, risk_choices, default=default), index=regimes_df.index)

    def save_regimes(self, regimes_df: pd.DataFrame, filename: str = 'regimes.parquet'):
        """
        Save regime data to file.

        Args:
            regimes_df: Regimes DataFrame
            filename: Output filename
        """
        filepath = self.regimes_path / filename

        try:
            regimes_df.to_parquet(filepath)
            logger.info(f"Saved regimes to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save regimes: {e}")

    def load_regimes(self, filename: str = 'regimes.parquet') -> Optional[pd.DataFrame]:
        """
        Load regime data from file.

        Args:
            filename: Input filename

        Returns:
            Regimes DataFrame or None
        """
        filepath = self.regimes_path / filename

        if not filepath.exists():
            return None

        try:
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded regimes from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load regimes: {e}")
            return None

    def update_regimes(self, symbols: List[str], start_date: str, end_date: str,
                      force_refresh: bool = False) -> pd.DataFrame:
        """
        Update regime detection for given period.

        Args:
            symbols: List of symbols (for cross-asset features)
            start_date: Start date
            end_date: End date
            force_refresh: If True, refresh all data

        Returns:
            DataFrame with regime classifications
        """
        # Get cross-asset features
        features_df = self.cross_asset.build_cross_asset_features(
            symbols, start_date, end_date, force_refresh=force_refresh
        )

        if features_df.empty:
            logger.warning("No features available for regime detection")
            return pd.DataFrame()

        # Detect regimes
        regimes_df = self.detect_regimes_from_features(features_df)

        # Save regimes
        self.save_regimes(regimes_df)

        return regimes_df