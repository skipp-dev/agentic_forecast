import logging
from datetime import datetime, timedelta
from ..graphs.state import GraphState
from ..agents.macro_data_agent import MacroDataAgent
from ..agents.regime_agent import RegimeAgent

logger = logging.getLogger(__name__)

def macro_data_node(state: GraphState) -> GraphState:
    """
    Fetches macro economic and commodity data.
    """
    logger.info("--- Node: Macro Data Agent ---")
    
    try:
        agent = MacroDataAgent(state.get('config', {}))
        
        # Determine date range (e.g. last 180 days for regime detection and features)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        macro_data = agent.get_macro_data(start_date, end_date)
        
        state['macro_data'] = macro_data
        logger.info(f"Collected macro data: {list(macro_data.get('raw_data', {}).keys())}")
        
    except Exception as e:
        logger.error(f"Macro data collection failed: {e}")
        state['macro_data'] = {}
        state['errors'].append(f"Macro data error: {e}")
        
    return state

def regime_detection_node(state: GraphState) -> GraphState:
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
            
        target_date = datetime.now().strftime('%Y-%m-%d')
        
        regimes = agent.detect_regime(target_date, raw_macro_data)
        
        state['regimes'] = regimes
        logger.info(f"Detected regimes: {regimes}")
        
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        state['regimes'] = {}
        state['errors'].append(f"Regime detection error: {e}")
        
    return state
