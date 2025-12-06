"""
Feature Engineering Module

Decouples feature creation from data fetching.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Centralized feature engineering logic.
    """
    
    @staticmethod
    def add_technical_features(df: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume') -> pd.DataFrame:
        """
        Add technical indicators to the dataframe.
        """
        df = df.copy()
        
        # Returns
        df['returns'] = df[price_col].pct_change()
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Volatility (20-day rolling std of returns)
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Volume MA
        if volume_col in df.columns:
            df['volume_ma'] = df[volume_col].rolling(window=20).mean()
            
        # Momentum
        df['momentum_1m'] = df[price_col].pct_change(periods=20)
        df['momentum_3m'] = df[price_col].pct_change(periods=60)
        
        return df

    @staticmethod
    def add_spectral_features(df: pd.DataFrame, col: str = 'returns', window: int = 60) -> pd.DataFrame:
        """
        Add spectral features using FFT (CPU-based fallback).
        Calculates dominant frequency and spectral entropy over a rolling window.
        """
        df = df.copy()
        
        if col not in df.columns:
            logger.warning(f"Column {col} not found for spectral features")
            return df
            
        # Fill NaNs for FFT
        series = df[col].fillna(0).values
        
        # Pre-allocate arrays
        n = len(series)
        dominant_freq = np.zeros(n)
        spectral_entropy = np.zeros(n)
        
        # Rolling FFT
        # Note: This is slow on CPU for large datasets, hence why we want GPU usually.
        # For ablation study on small golden dataset, this is fine.
        for i in range(window, n):
            window_data = series[i-window:i]
            
            # Apply Hanning window to reduce leakage
            window_data = window_data * np.hanning(window)
            
            # FFT
            fft_vals = np.fft.rfft(window_data)
            power_spectrum = np.abs(fft_vals)**2
            
            # Dominant Frequency (index of max power, excluding DC component at 0)
            if len(power_spectrum) > 1:
                dominant_freq[i] = np.argmax(power_spectrum[1:]) + 1
            
            # Spectral Entropy
            # Normalize power spectrum to get probability distribution
            ps_sum = np.sum(power_spectrum)
            if ps_sum > 0:
                ps_norm = power_spectrum / ps_sum
                # Compute entropy (avoid log(0))
                ps_norm = ps_norm[ps_norm > 0]
                spectral_entropy[i] = -np.sum(ps_norm * np.log(ps_norm))
                
        df['spectral_dominant_freq'] = dominant_freq
        df['spectral_entropy'] = spectral_entropy
        
        return df
