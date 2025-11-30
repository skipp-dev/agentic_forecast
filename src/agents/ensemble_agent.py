import pandas as pd
import numpy as np
from typing import Dict, Any

class EnsembleAgent:
    """
    Agent for creating ensemble forecasts from multiple model predictions.
    """

    def create_ensemble_forecast(self, forecasts: Dict[str, Dict[str, pd.DataFrame]], performance_summary: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Creates an ensemble forecast using a weighted average based on model performance (MAPE).
        """
        if performance_summary.empty:
            return {}

        ensemble_forecasts: Dict[str, pd.DataFrame] = {}

        for symbol, model_outputs in forecasts.items():
            symbol_performance = performance_summary[
                performance_summary['symbol'] == symbol
            ].dropna(subset=['mape'])

            if symbol_performance.empty:
                continue

            # Avoid division-by-zero by adding a small epsilon and fall back to uniform weights
            epsilon = 1e-6
            weights = 1 / (symbol_performance['mape'] + epsilon)
            if not np.isfinite(weights).all() or weights.sum() == 0:
                weights = pd.Series(
                    1.0,
                    index=symbol_performance.index
                )

            weighted_sum = None
            weight_total = 0.0

            for idx, row in symbol_performance.iterrows():
                model_family = row['model_family']
                if model_family not in model_outputs:
                    continue

                forecast_df = model_outputs[model_family]
                if 'ds' not in forecast_df.columns or model_family not in forecast_df.columns:
                    continue

                series = forecast_df.set_index('ds')[model_family]
                weight = float(weights.loc[idx])
                if weight <= 0 or series.empty:
                    continue

                weighted_series = series * weight
                if weighted_sum is None:
                    weighted_sum = weighted_series
                else:
                    weighted_sum = weighted_sum.add(weighted_series, fill_value=0.0)
                weight_total += weight

            if weighted_sum is None or weight_total == 0:
                continue

            ensemble_series = weighted_sum / weight_total
            ensemble_df = ensemble_series.reset_index()
            ensemble_df.columns = ['ds', 'ensemble']
            ensemble_forecasts[symbol] = ensemble_df

        return ensemble_forecasts
