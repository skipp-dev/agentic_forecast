from ..graphs.state import GraphState
from ..agents.ensemble_agent import EnsembleAgent
from ..agents.analytics_agent import AnalyticsAgent
import pandas as pd

def ensemble_node(state: GraphState) -> GraphState:
    """
    Creates an ensemble forecast from the individual model forecasts.
    """
    print("--- Node: Ensemble Forecasting ---")
    
    agent = EnsembleAgent()
    
    # Convert forecasts back to DataFrames for ensemble creation
    forecasts_dfs = {}
    for symbol, model_forecasts in state.get('forecasts', {}).items():
        forecasts_dfs[symbol] = {}
        for model_family, forecast_dict in model_forecasts.items():
            if isinstance(forecast_dict, dict):
                forecasts_dfs[symbol][model_family] = pd.DataFrame.from_dict(forecast_dict, orient='index')
            else:
                forecasts_dfs[symbol][model_family] = forecast_dict
    
    ensemble_forecasts = agent.create_ensemble_forecast(
        forecasts_dfs,
        state['performance_summary']
    )
    
    # Update the main forecasts dictionary with the new ensemble forecast
    for symbol, forecast_df in ensemble_forecasts.items():
        if symbol not in state['forecasts']:
            continue
        # Convert back to dict for state storage
        state['forecasts'][symbol]['ensemble'] = forecast_df.to_dict('index')

    # After adding the ensemble, recalculate the performance summary to include it
    analytics_agent = AnalyticsAgent()
    
    # Convert forecasts back to DataFrames
    forecasts_dfs = {}
    for symbol, model_forecasts in state.get('forecasts', {}).items():
        forecasts_dfs[symbol] = {}
        for model_family, forecast_dict in model_forecasts.items():
            if isinstance(forecast_dict, dict):
                forecasts_dfs[symbol][model_family] = pd.DataFrame.from_dict(forecast_dict, orient='index')
            else:
                forecasts_dfs[symbol][model_family] = forecast_dict
    
    # Convert raw_data back to DataFrames
    raw_data_dfs = {}
    for symbol, raw_dict in state.get('raw_data', {}).items():
        if isinstance(raw_dict, dict):
            raw_data_dfs[symbol] = pd.DataFrame.from_dict(raw_dict, orient='index')
        else:
            raw_data_dfs[symbol] = raw_dict
    
    performance_summary = analytics_agent.calculate_performance_summary(
        forecasts_dfs,
        raw_data_dfs
    )
    state['performance_summary'] = performance_summary

    print("✅ Created ensemble forecasts and updated performance summary.")
    return state
