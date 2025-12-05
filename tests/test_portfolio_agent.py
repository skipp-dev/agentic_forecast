import pytest
from src.agents.portfolio_manager_agent import PortfolioManagerAgent

@pytest.fixture
def portfolio_agent():
    return PortfolioManagerAgent(config={
        'max_position_size': 0.2,
        'target_cash': 0.0,
        'risk': {'max_portfolio_var': 1.0} # Relax risk for these tests
    })

def test_total_weight_does_not_exceed_limit(portfolio_agent):
    """Test that total allocation does not exceed 1.0 (minus cash)."""
    # Mock inputs
    best_models = {f"SYM{i}": {} for i in range(10)}
    # Recommend promoting all of them
    actions = [f"Promote ModelX for SYM{i}" for i in range(10)]
    
    # Mock risk agent to always approve
    portfolio_agent.risk_agent.assess_portfolio_risk = lambda pos, data: {'risk_approved': True, 'metrics': {}}
    
    result = portfolio_agent.construct_portfolio(actions, best_models, {})
    weights = result.target_weights
    
    total_weight = sum(weights.values())
    assert total_weight <= 1.0 + 1e-6

def test_per_symbol_max_weight_respected(portfolio_agent):
    """Test that individual position sizes are capped."""
    # Only 1 symbol
    best_models = {"AAPL": {}}
    actions = ["Promote ModelX for AAPL"]
    
    # Mock risk agent
    portfolio_agent.risk_agent.assess_portfolio_risk = lambda pos, data: {'risk_approved': True, 'metrics': {}}
    
    result = portfolio_agent.construct_portfolio(actions, best_models, {})
    weights = result.target_weights
    
    # Should be capped at 0.2 (from config)
    assert weights["AAPL"] <= 0.2 + 1e-6

def test_empty_signals_yield_zero_allocation(portfolio_agent):
    """Test that no signals result in empty portfolio."""
    result = portfolio_agent.construct_portfolio([], {}, {})
    weights = result.target_weights
    
    assert sum(weights.values()) == 0.0

def test_normalization_of_excessive_signals(portfolio_agent):
    """Test that if signals imply > 100%, they are normalized."""
    # 20 symbols, all promoted. Base weight is 0.1. Sum would be 2.0.
    best_models = {f"SYM{i}": {} for i in range(20)}
    actions = [f"Promote ModelX for SYM{i}" for i in range(20)]
    
    portfolio_agent.risk_agent.assess_portfolio_risk = lambda pos, data: {'risk_approved': True, 'metrics': {}}
    
    result = portfolio_agent.construct_portfolio(actions, best_models, {})
    weights = result.target_weights
    
    total_weight = sum(weights.values())
    assert total_weight <= 1.0 + 1e-6
    # Each should be roughly 0.05
    assert weights["SYM0"] == pytest.approx(0.05, abs=0.01)
