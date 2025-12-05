import logging
from typing import Dict, Any
from src.core.state import PipelineGraphState
from src.agents.portfolio_manager_agent import PortfolioManagerAgent
from dataclasses import asdict

logger = logging.getLogger(__name__)

def portfolio_construction_node(state: PipelineGraphState) -> PipelineGraphState:
    """
    Constructs a portfolio based on trading signals.
    """
    logger.info("--- Node: Portfolio Construction ---")
    
    signals = state.get('signals', {})
    raw_data = state.get('data', {})
    config = state.get('config', {})
    
    # Initialize Portfolio Manager
    portfolio_agent = PortfolioManagerAgent(config=config)
    
    # Construct Portfolio
    allocation = portfolio_agent.construct_portfolio_from_signals(signals, raw_data)
    
    # Update State
    state['portfolio'] = allocation.target_weights
    
    # Generate Orders (Simple diff for now, assuming starting from 0 or previous state if we had it)
    # In a real system, we'd compare against current holdings.
    # Here we just emit target orders.
    
    orders = []
    for symbol, weight in allocation.target_weights.items():
        orders.append({
            'symbol': symbol,
            'action': 'BUY' if weight > 0 else 'SELL', # Simplified
            'target_weight': weight,
            'type': 'MARKET'
        })
        
    state['orders'] = orders
    
    logger.info(f"Portfolio Constructed. Target Weights: {allocation.target_weights}")
    
    return state
