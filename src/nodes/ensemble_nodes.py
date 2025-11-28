from ..graphs.state import GraphState
from ..agents.ensemble_agent import EnsembleAgent
from ..agents.analytics_agent import AnalyticsAgent

def ensemble_node(state: GraphState) -> GraphState:
    """
    Creates an ensemble forecast from the individual model forecasts.
    """
    print("--- Node: Ensemble Forecasting ---")
    
    agent = EnsembleAgent()
    
    ensemble_forecasts = agent.create_ensemble_forecast(
        state['forecasts'],
        state['performance_summary']
    )
    
    # Update the main forecasts dictionary with the new ensemble forecast
    for symbol, forecast_df in ensemble_forecasts.items():
        if symbol not in state['forecasts']:
            continue
        state['forecasts'][symbol]['ensemble'] = forecast_df

    # After adding the ensemble, recalculate the performance summary to include it
    analytics_agent = AnalyticsAgent()
    performance_summary = analytics_agent.calculate_performance_summary(
        state.get('forecasts', {}),
        state.get('raw_data', {})
    )
    state['performance_summary'] = performance_summary

    print("✅ Created ensemble forecasts and updated performance summary.")
    return state
