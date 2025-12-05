from typing import Dict, Any, List
import logging
from ..agents.drift_monitor_agent import DriftMonitorAgent
from ..core.state import PipelineGraphState
from ..gpu_services import get_gpu_services

logger = logging.getLogger(__name__)

def monitoring_node(state: PipelineGraphState) -> PipelineGraphState:
    """
    Runs the drift monitor agent to detect drift in data, performance, and spectral features.
    """
    logger.info("--- Node: Monitoring Agent ---")
    
    # Initialize agent
    gpu_services = get_gpu_services()
    agent = DriftMonitorAgent(gpu_services=gpu_services)
    
    symbols = state.get('symbols', [])
    drift_metrics = {}
    drift_detected_symbols = []
    
    for symbol in symbols:
        try:
            # Run comprehensive drift check
            result = agent.comprehensive_drift_check(symbol)
            drift_metrics[symbol] = result
            
            # Determine if drift is detected based on the result
            # Check if any specific drift type was detected
            perf_drift = result.get('performance_drift', {}).get('drift_detected', False)
            data_drift = result.get('data_drift', {}).get('drift_detected', False)
            spec_drift = result.get('spectral_drift', {}).get('drift_detected', False)
            regime_change = result.get('regime_change', False)
            
            if perf_drift or data_drift or spec_drift or regime_change:
                drift_detected_symbols.append(symbol)
                logger.info(f"Drift detected for {symbol}: Perf={perf_drift}, Data={data_drift}, Spec={spec_drift}, Regime={regime_change}")
                
        except Exception as e:
            logger.error(f"Error monitoring drift for {symbol}: {e}")
            
    # Update state
    state['drift_metrics'] = drift_metrics
    state['drift_detected'] = drift_detected_symbols
    
    logger.info(f"Monitoring complete. Drift detected in: {drift_detected_symbols}")
    
    return state

