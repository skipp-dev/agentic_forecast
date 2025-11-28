
import pandas as pd
import numpy as np
from typing import Dict

class AnalyticsAgent:
    """
    Agent for performing analytics on model forecasts.
    """

    def calculate_performance_summary(
        self, 
        forecasts: Dict[str, Dict[str, pd.DataFrame]], 
        raw_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculates the Mean Absolute Percentage Error (MAPE) for each model and symbol.

        Args:
            forecasts: A dictionary of forecasts, keyed by symbol and then model family.
            raw_data: A dictionary of raw data, keyed by symbol.

        Returns:
            A DataFrame summarizing the performance metrics.
        """
        performance_data = []

        for symbol, model_forecasts in forecasts.items():
            if symbol not in raw_data:
                continue

            actuals_df = raw_data[symbol].copy()
            if not isinstance(actuals_df.index, pd.DatetimeIndex):
                actuals_df['ds'] = pd.to_datetime(actuals_df['ds'])
                actuals_df.set_index('ds', inplace=True)

            for model_family, forecast_df in model_forecasts.items():
                
                # Ensure forecast_df 'ds' column is datetime
                forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

                # Merge forecast with actuals
                merged_df = pd.merge(
                    forecast_df,
                    actuals_df[['close']],
                    left_on='ds',
                    right_index=True,
                    how='inner'
                )

                if merged_df.empty:
                    mape = np.nan
                else:
                    # Calculate MAPE
                    y_true = merged_df['close']
                    y_pred = merged_df[model_family]
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

                performance_data.append({
                    'symbol': symbol,
                    'model_family': model_family,
                    'mape': mape
                })

        return pd.DataFrame(performance_data)

