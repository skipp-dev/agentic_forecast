"""
Feature Engineer Agent

Automated feature engineering with spectral analysis and GPU acceleration.
Extends existing feature engineering with advanced techniques.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression
import talib

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.feature_agent import FeatureAgent
from src.gpu_services import get_gpu_services
from src.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

class FeatureEngineerAgent(FeatureAgent):
    """
    Advanced feature engineering agent with spectral analysis and automation.

    Extends the existing FeatureAgent with:
    - GPU-accelerated spectral feature extraction
    - Automated feature selection
    - Advanced technical indicators
    - Feature importance analysis
    - Real-time feature engineering
    """

    def __init__(self, gpu_services=None, data_pipeline=None):
        """
        Initialize feature engineer agent.

        Args:
            gpu_services: GPU services instance for spectral analysis
            data_pipeline: Data pipeline instance
        """
        super().__init__()
        self.gpu_services = gpu_services or get_gpu_services()
        self.data_pipeline = data_pipeline or DataPipeline()

        # Feature engineering configuration
        self.feature_sets = {
            'basic': ['sma', 'ema', 'rsi', 'macd', 'bbands'],
            'advanced': ['stoch', 'williams_r', 'cci', 'mfi', 'adx'],
            'spectral': ['dominant_freq', 'spectral_entropy', 'spectral_centroid'],
            'volatility': ['parkinson', 'garman_klass', 'yang_zhang'],
            'momentum': ['roc', 'mom', 'trix', 'kama']
        }

        # Feature importance tracking
        self.feature_importance_history = {}
        self.selected_features = {}

        logger.info("Feature Engineer Agent initialized with GPU acceleration")

    def engineer_features(self, symbol: str, data: pd.DataFrame = None,
                         feature_sets: List[str] = None) -> pd.DataFrame:
        """
        Engineer comprehensive feature set for a symbol.

        Args:
            symbol: Stock symbol
            data: Raw market data (fetched if None)
            feature_sets: List of feature sets to include

        Returns:
            DataFrame with engineered features
        """
        if data is None:
            data = self._fetch_market_data(symbol)

        if feature_sets is None:
            feature_sets = ['basic', 'spectral']  # Default feature sets

        logger.info(f"Engineering features for {symbol} with sets: {feature_sets}")

        # Validate and correct OHLC data quality
        features_df = self._validate_ohlc_data(data.copy())

        # Add each requested feature set
        for feature_set in feature_sets:
            if feature_set == 'basic':
                features_df = self._add_basic_technical_features(features_df)
            elif feature_set == 'advanced':
                features_df = self._add_advanced_technical_features(features_df)
            elif feature_set == 'spectral':
                features_df = self._add_spectral_features(features_df, symbol)
            elif feature_set == 'volatility':
                features_df = self._add_volatility_features(features_df)
            elif feature_set == 'momentum':
                features_df = self._add_momentum_features(features_df)

        # Clean and normalize features
        features_df = self._clean_and_normalize_features(features_df)

        logger.info(f"Feature engineering complete: {features_df.shape[1]} features created")

        return features_df

    def _validate_ohlc_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and correct OHLC data quality issues.

        Args:
            data: Raw OHLC data

        Returns:
            DataFrame with validated OHLC data
        """
        df = data.copy()

        # Ensure required OHLC columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing OHLC columns: {missing_cols}")
            return df

        # Fix OHLC relationships
        # High should be >= max(open, close)
        # Low should be <= min(open, close)
        # High should be >= low

        # Calculate proper high/low bounds
        price_max = df[['open', 'close']].max(axis=1)
        price_min = df[['open', 'close']].min(axis=1)

        # Correct high values that are too low
        invalid_high = df['high'] < price_max
        if invalid_high.any():
            logger.warning(f"Correcting {invalid_high.sum()} invalid high values")
            df.loc[invalid_high, 'high'] = price_max[invalid_high]

        # Correct low values that are too high
        invalid_low = df['low'] > price_min
        if invalid_low.any():
            logger.warning(f"Correcting {invalid_low.sum()} invalid low values")
            df.loc[invalid_low, 'low'] = price_min[invalid_low]

        # Ensure high >= low
        invalid_range = df['high'] < df['low']
        if invalid_range.any():
            logger.warning(f"Correcting {invalid_range.sum()} high < low relationships")
            # Swap high and low for invalid ranges
            temp = df.loc[invalid_range, 'high'].copy()
            df.loc[invalid_range, 'high'] = df.loc[invalid_range, 'low']
            df.loc[invalid_range, 'low'] = temp

        return df

    def _add_basic_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators."""
        df = data.copy()

        # Price-based features
        df['returns'] = df['close'].pct_change(fill_method=None)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)

        # Oscillators
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    def _add_advanced_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators."""
        df = data.copy()

        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)

        # Commodity Channel Index
        df['cci_20'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)

        # Money Flow Index
        df['mfi_14'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)

        # Average Directional Movement Index
        df['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['di_plus'], df['di_minus'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14), \
                                       talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)

        # Chaikin Money Flow
        df['cmf_20'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])  # Accumulation/Distribution
        df['cmf_20'] = df['cmf_20'].rolling(20).mean()  # Simplified CMF

        # On Balance Volume
        df['obv'] = talib.OBV(df['close'], df['volume'])

        return df

    def _add_spectral_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add spectral features using GPU-accelerated cuFFT."""
        df = data.copy()

        if not self.gpu_services:
            logger.warning("GPU services not available for spectral features")
            return df

        try:
            # Extract price series for spectral analysis
            price_series = df['close'].dropna().values

            if len(price_series) >= 100:  # Minimum length for meaningful spectral analysis
                # Get spectral features
                spectral_features = self.gpu_services.spectral_service.extract_spectral_features(
                    price_series, normalize=True
                )

                # Add spectral features to dataframe
                for feature_name, value in spectral_features.items():
                    df[f'spectral_{feature_name}'] = value

                # Rolling spectral features (last 50 periods)
                rolling_spectra = []
                window_size = min(256, len(price_series))

                for i in range(window_size, len(price_series)):
                    window_data = price_series[i-window_size:i]
                    window_spectra = self.gpu_services.spectral_service.extract_spectral_features(
                        window_data, normalize=True
                    )
                    rolling_spectra.append(window_spectra)

                # Add rolling spectral features
                spectral_df = pd.DataFrame(rolling_spectra)
                spectral_df.index = df.index[-len(spectral_df):]

                for col in spectral_df.columns:
                    df[f'rolling_spectral_{col}'] = spectral_df[col]

                logger.info(f"Added {len(spectral_features)} spectral features")

        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {e}")

        return df

    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced volatility measures."""
        df = data.copy()

        # Parkinson volatility (high-low based)
        df['parkinson_vol'] = (1/(4*np.log(2))) * (np.log(df['high']/df['low']))**2
        df['parkinson_vol_20'] = df['parkinson_vol'].rolling(20).mean()

        # Garman-Klass volatility
        df['gk_vol'] = 0.5 * (np.log(df['high']/df['low']))**2 - \
                      (2*np.log(2)-1) * (np.log(df['close']/df['open']))**2
        df['gk_vol_20'] = df['gk_vol'].rolling(20).mean()

        # Realized volatility (daily returns)
        df['realized_vol_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['realized_vol_60'] = df['returns'].rolling(60).std() * np.sqrt(252)

        # ATR (Average True Range)
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        return df

    def _add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        df = data.copy()

        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)

        # Momentum
        for period in [5, 10, 20]:
            df[f'mom_{period}'] = talib.MOM(df['close'], timeperiod=period)

        # TRIX (Triple Exponential Average)
        df['trix_30'] = talib.TRIX(df['close'], timeperiod=30)

        # KAMA (Kaufman's Adaptive Moving Average)
        df['kama_30'] = talib.KAMA(df['close'], timeperiod=30)

        # TSI (True Strength Index)
        df['tsi_25_13'] = talib.TSI(df['close'], fastperiod=25, slowperiod=13)

        # Ultimate Oscillator
        df['ultosc'] = talib.ULTOSC(df['high'], df['low'], df['close'],
                                   timeperiod1=7, timeperiod2=14, timeperiod3=28)

        return df

    def _clean_and_normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize engineered features."""
        df = data.copy()

        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill then backward fill missing values
        df = df.ffill().bfill()

        # Remove columns with all NaN values
        df = df.dropna(axis=1, how='all')

        # Fill remaining NaN with 0
        df = df.fillna(0)

        # Preserve raw OHLC columns to maintain financial relationships
        protected_cols = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in df.columns]

        # Normalize numeric columns (except date/index) while skipping protected columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in protected_cols]
        scaler = RobustScaler()

        for col in numeric_cols:
            if df[col].std() > 0:  # Only scale if not constant
                df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()

        return df

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       method: str = 'mutual_info', k: int = 20) -> List[str]:
        """
        Select most important features using various methods.

        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('mutual_info', 'correlation', 'variance')
            k: Number of features to select

        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top {k} features using {method}")

        if method == 'mutual_info':
            # Mutual information
            mi_scores = mutual_info_regression(X, y)
            feature_scores = dict(zip(X.columns, mi_scores))

        elif method == 'correlation':
            # Correlation with target
            correlations = X.corrwith(y).abs()
            feature_scores = correlations.to_dict()

        elif method == 'variance':
            # Feature variance
            variances = X.var()
            feature_scores = variances.to_dict()

        else:
            raise ValueError(f"Unknown selection method: {method}")

        # Sort and select top k
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, score in sorted_features[:k]]

        logger.info(f"Selected features: {selected_features}")

        return selected_features

    def analyze_feature_importance(self, symbol: str, X: pd.DataFrame,
                                 y: pd.Series) -> Dict[str, float]:
        """
        Analyze feature importance for a symbol.

        Args:
            symbol: Stock symbol
            X: Feature matrix
            y: Target variable

        Returns:
            Feature importance scores
        """
        # Use mutual information as importance measure
        mi_scores = mutual_info_regression(X, y)
        importance_dict = dict(zip(X.columns, mi_scores))

        # Store in history
        self.feature_importance_history[symbol] = {
            'importance': importance_dict,
            'timestamp': datetime.now(),
            'n_features': len(X.columns)
        }

        return importance_dict

    def optimize_feature_set(self, symbol: str, base_features: List[str] = None) -> List[str]:
        """
        Optimize feature set for a symbol based on historical performance.

        Args:
            symbol: Stock symbol
            base_features: Base features to start with

        Returns:
            Optimized feature set
        """
        if base_features is None:
            base_features = ['basic', 'spectral']

        # Analyze historical feature performance
        if symbol in self.feature_importance_history:
            importance_data = self.feature_importance_history[symbol]

            # Get top performing features
            sorted_features = sorted(
                importance_data['importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Select top 80% of features
            n_select = int(len(sorted_features) * 0.8)
            optimized_features = [f[0] for f in sorted_features[:n_select]]

            logger.info(f"Optimized feature set for {symbol}: {len(optimized_features)} features")
            return optimized_features

        # Fallback to default
        return base_features

    def _fetch_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch market data for feature engineering."""
        try:
            # Use data pipeline to fetch data
            data = self.data_pipeline.av_client.get_daily_data(symbol, period='2y')
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")
            # Return mock data for testing
            dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
            mock_data = {
                'open': np.random.randn(500).cumsum() + 100,
                'high': np.random.randn(500).cumsum() + 105,
                'low': np.random.randn(500).cumsum() + 95,
                'close': np.random.randn(500).cumsum() + 100,
                'volume': np.random.randint(1000000, 10000000, 500)
            }
            return pd.DataFrame(mock_data, index=dates)

# Convenience functions
def create_feature_engineer_agent():
    """Create and configure feature engineer agent."""
    return FeatureEngineerAgent()

def engineer_features_for_symbol(symbol: str, feature_sets: List[str] = None):
    """Engineer features for a symbol with default settings."""
    agent = create_feature_engineer_agent()
    return agent.engineer_features(symbol, feature_sets=feature_sets)