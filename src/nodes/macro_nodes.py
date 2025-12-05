import logging
import pandas as pd
from datetime import datetime, timedelta
from src.core.state import PipelineGraphState
from src.agents.macro_data_agent import MacroDataAgent
from src.agents.regime_agent import RegimeAgent
from src.agents.regime_detection_agent import RegimeDetectionAgent

logger = logging.getLogger(__name__)

def macro_data_node(state: PipelineGraphState) -> PipelineGraphState:
    """
    Fetches macro economic and commodity data.
    """
    logger.info("--- Node: Macro Data Agent ---")
    
    try:
        agent = MacroDataAgent(state.get('config', {}))
        
        # Determine date range (e.g. last 180 days for regime detection and features)
        # Respect cutoff_date for backtesting
        cutoff_date_str = state.get('cutoff_date')
        if cutoff_date_str:
            end_dt = pd.to_datetime(cutoff_date_str)
        else:
            end_dt = datetime.now()
            
        end_date = end_dt.strftime('%Y-%m-%d')
        start_date = (end_dt - timedelta(days=180)).strftime('%Y-%m-%d')
        
        macro_data = agent.get_macro_data(start_date, end_date)
        
        state['macro_data'] = macro_data
        logger.info(f"Collected macro data: {list(macro_data.get('raw_data', {}).keys())}")
        
    except Exception as e:
        logger.error(f"Macro data collection failed: {e}")
        state['macro_data'] = {}
        state['errors'].append(f"Macro data error: {e}")
        
    return state

def regime_detection_node(state: PipelineGraphState) -> PipelineGraphState:
    """
    Detects market regimes based on macro data.
    """
    logger.info("--- Node: Regime Detection Agent ---")
    
    try:
        agent = RegimeAgent()
        
        macro_data_full = state.get('macro_data', {})
        raw_macro_data = macro_data_full.get('raw_data', {})
        
        if not raw_macro_data:
            logger.warning("No macro data available for regime detection")
            state['regimes'] = {}
            return state
            
        # Respect cutoff_date for backtesting
        cutoff_date_str = state.get('cutoff_date')
        if cutoff_date_str:
            target_date = cutoff_date_str
        else:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        regimes = agent.detect_regime(target_date, raw_macro_data)
        
        state['regimes'] = regimes
        logger.info(f"Detected regimes: {regimes}")
        
        # Also detect historical regimes for feature engineering
        try:
            complex_agent = RegimeDetectionAgent()
            processed_features = macro_data_full.get('processed_features')
            if processed_features is not None and not processed_features.empty:
                # Pass cutoff_date to enforce point-in-time correctness
                historical_regimes = complex_agent.detect_regimes(processed_features, cutoff_date=cutoff_date_str)
                
                # Convert Series to serializable format (Dict[str, Dict[date_str, value]])
                serializable_history = {}
                for name, series in historical_regimes.items():
                    # Ensure index is string (YYYY-MM-DD)
                    series_copy = series.copy()
                    series_copy.index = series_copy.index.strftime('%Y-%m-%d')
                    serializable_history[name] = series_copy.to_dict()
                
                state['historical_regimes'] = serializable_history
                logger.info(f"Detected historical regimes: {list(historical_regimes.keys())}")
        except Exception as e:
            logger.warning(f"Historical regime detection failed: {e}")
        
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        state['regimes'] = {}
        state['errors'].append(f"Regime detection error: {e}")
        
    return state
