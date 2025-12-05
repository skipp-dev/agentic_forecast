# analytics/evaluation_metrics.py
"""
Evaluation metrics computation for forecast performance.
Includes SMAPE, SWASE, and other forecast accuracy metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import json
import time


def evaluate_forecast_series(
    actual: np.ndarray,
    pred: np.ndarray,
    regime_flags: Optional[Dict[str, np.ndarray]] = None,
    shock_weight: float = 3.0,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Evaluate one forecast series (single symbol + horizon + model_family).

    Inputs:
        actual        : array of true values (prices or returns)
        pred          : array of predictions (same length as actual)
        regime_flags  : optional dict of flags, e.g. {"peer_shock_flag": np.array([...])}
        shock_weight  : how much more we penalize errors on shock days
        eps           : small constant to avoid division-by-zero

    Outputs:
        dict with mae, rmse, mape, smape, swase, directional_accuracy, n_samples
    """
    # 1. Basic checks
    if len(actual) == 0 or len(actual) != len(pred):
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "smape": np.nan,
            "swase": np.nan,
            "directional_accuracy": np.nan,
            "n_samples": 0,
        }

    # 2. Core arrays
    abs_error = np.abs(pred - actual)
    n = len(actual)

    # 3. Basic metrics (you already have this logic somewhere)
    mae = float(abs_error.mean())
    rmse = float(np.sqrt((abs_error ** 2).mean()))

    # MAPE
    mape_terms = abs_error / (np.abs(actual) + eps)
    mape = float(np.mean(mape_terms[np.isfinite(mape_terms)])) if np.isfinite(mape_terms).any() else 1.0

    # Directional accuracy (based on changes)
    if n > 1:
        actual_change = np.sign(np.diff(actual))
        pred_change = np.sign(np.diff(pred))
        directional_accuracy = float(np.mean(actual_change == pred_change))
    else:
        directional_accuracy = 0.0

    # 4. SMAPE
    # smape_t = 2 * |pred - actual| / (|pred| + |actual| + eps)
    smape_terms = 2.0 * abs_error / (np.abs(pred) + np.abs(actual) + eps)
    valid_smape = smape_terms[np.isfinite(smape_terms)]
    smape = float(valid_smape.mean()) if len(valid_smape) > 0 else 1.0

    # 5. SWASE (Shock-Weighted Absolute Scaled Error)
    # ase_t = |pred - actual| / (|actual| + eps)
    ase_terms = abs_error / (np.abs(actual) + eps)

    # Default: all weights = 1
    weights = np.ones_like(ase_terms, dtype=float)

    # Optionally: increase weight for shock days
    if regime_flags is not None:
        # Example: treat "peer_shock_flag" OR "has_macro_event_today" as shocks
        shock_mask = np.zeros_like(weights, dtype=bool)

        if "peer_shock_flag" in regime_flags:
            shock_mask |= (regime_flags["peer_shock_flag"].astype(bool))

        if "has_macro_event_today" in regime_flags:
            shock_mask |= (regime_flags["has_macro_event_today"].astype(bool))

        weights[shock_mask] = shock_weight

    # Weighted average
    weighted_sum = float(np.sum(weights * ase_terms))
    weight_total = float(np.sum(weights))
    swase = float(weighted_sum / weight_total) if weight_total > 0 else 1.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "swase": swase,
        "directional_accuracy": directional_accuracy,
        "n_samples": n,
    }


def detect_shock_days_from_returns(returns: np.ndarray, threshold: float = 0.03) -> np.ndarray:
    """
    Detect shock days based on return magnitude.
    A day is considered a shock if |return| > threshold.

    Args:
        returns: Array of daily returns
        threshold: Return magnitude threshold for shock detection

    Returns:
        Boolean array where True indicates shock days
    """
    return np.abs(returns) > threshold


def detect_shock_days_from_volatility(returns: np.ndarray, window: int = 20, threshold_std: float = 2.0) -> np.ndarray:
    """
    Detect shock days based on volatility spikes.
    A day is considered a shock if its return is more than threshold_std standard deviations
    from the rolling mean over the window.

    Args:
        returns: Array of daily returns
        window: Rolling window for volatility calculation
        threshold_std: Standard deviation threshold for shock detection

    Returns:
        Boolean array where True indicates shock days
    """
    if len(returns) < window:
        return np.zeros_like(returns, dtype=bool)

    # Calculate rolling mean and std
    # Use center=False to avoid look-ahead bias even in evaluation
    rolling_mean = pd.Series(returns).rolling(window=window, center=False).mean().fillna(method='ffill').values
    rolling_std = pd.Series(returns).rolling(window=window, center=False).std().fillna(method='ffill').values

    # Avoid division by zero
    rolling_std = np.where(rolling_std == 0, 1e-8, rolling_std)

    # Z-score
    z_scores = (returns - rolling_mean) / rolling_std

    return np.abs(z_scores) > threshold_std


def create_regime_flags_from_data(returns: np.ndarray, method: str = 'returns', **kwargs) -> Dict[str, np.ndarray]:
    """
    Create regime flags dictionary from returns data.

    Args:
        returns: Array of daily returns
        method: Method for shock detection ('returns', 'volatility', or 'combined')
        **kwargs: Additional parameters for shock detection methods

    Returns:
        Dictionary with regime flags
    """
    flags = {}

    if method == 'returns':
        flags['peer_shock_flag'] = detect_shock_days_from_returns(returns, **kwargs)
    elif method == 'volatility':
        flags['peer_shock_flag'] = detect_shock_days_from_volatility(returns, **kwargs)
    elif method == 'combined':
        # Combine both methods
        returns_shocks = detect_shock_days_from_returns(returns, kwargs.get('returns_threshold', 0.03))
        vol_shocks = detect_shock_days_from_volatility(returns, **{k: v for k, v in kwargs.items() if k != 'returns_threshold'})
        flags['peer_shock_flag'] = returns_shocks | vol_shocks
    else:
        # Default: no shocks
        flags['peer_shock_flag'] = np.zeros_like(returns, dtype=bool)

    # For now, no macro events
    flags['has_macro_event_today'] = np.zeros_like(returns, dtype=bool)

    return flags


def load_evaluation_metrics() -> pd.DataFrame:
    """
    Read the latest evaluation metrics file from disk.
    Adjust this if you rotate files or use JSON instead of CSV.
    """
    eval_file = Path('data/metrics/evaluation_results_baseline_latest.csv')
    if not eval_file.exists():
        return pd.DataFrame()

    return pd.read_csv(eval_file)


def update_evaluation_results_with_swase(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Update evaluation results DataFrame to include SWASE if missing.
    For now, this is a placeholder - in production you'd recompute with regime data.
    """
    if 'swase' not in eval_df.columns:
        # Placeholder: set SWASE equal to MAPE for now
        # In production, you'd recompute with actual regime flags
        eval_df['swase'] = eval_df.get('mape', 1.0)

    return eval_df


def get_forecast_performance_metrics(eval_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract forecast performance metrics from evaluation DataFrame.
    Returns aggregated metrics suitable for Prometheus export.
    """
    if eval_df.empty:
        return {}

    # Ensure we have SWASE
    eval_df = update_evaluation_results_with_swase(eval_df)

    metrics = {}

    # Global aggregates
    for metric in ['mae', 'mape', 'smape', 'swase', 'directional_accuracy']:
        if metric in eval_df.columns:
            values = eval_df[metric].dropna()
            if len(values) > 0:
                metrics[f'global_{metric}_mean'] = float(values.mean())
                metrics[f'global_{metric}_std'] = float(values.std())

    # Per-symbol aggregates (simplified - you might want more granularity)
    if 'symbol' in eval_df.columns:
        symbol_metrics = {}
        for symbol in eval_df['symbol'].unique():
            symbol_data = eval_df[eval_df['symbol'] == symbol]
            symbol_metrics[symbol] = {}
            for metric in ['mae', 'mape', 'smape', 'swase', 'directional_accuracy']:
                if metric in symbol_data.columns:
                    values = symbol_data[metric].dropna()
                    if len(values) > 0:
                        symbol_metrics[symbol][f'{metric}_mean'] = float(values.mean())

        metrics['per_symbol'] = symbol_metrics

    return metrics


# Example usage
if __name__ == "__main__":
    # Example with dummy data
    np.random.seed(42)
    actual = np.random.randn(100) + 100  # Stock prices around 100
    pred = actual + np.random.randn(100) * 2  # Predictions with some error

    # Example regime flags
    regime_flags = {
        "peer_shock_flag": np.random.choice([0, 1], size=100, p=[0.9, 0.1])
    }

    results = evaluate_forecast_series(actual, pred, regime_flags)
    print("Evaluation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")