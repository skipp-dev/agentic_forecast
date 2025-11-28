"""
Strategy Selector Agent

Maps symbols to appropriate trading strategies based on the strategy playbook.
Provides strategy configuration and selection logic.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml

from src.config.config_loader import get_strategy_playbook

logger = logging.getLogger(__name__)

class StrategySelector:
    """
    Strategy selector that maps symbols to trading strategies.

    For Phase 1, uses static mapping from strategy playbook.
    Future phases can implement dynamic regime-based selection.
    """

    def __init__(self, playbook_path: str = "config/strategy_playbook.yaml"):
        """
        Initialize strategy selector.

        Args:
            playbook_path: Path to strategy playbook configuration (deprecated - now uses config_loader)
        """
        self.playbook_path = Path(playbook_path)  # Keep for backward compatibility
        self.playbook = get_strategy_playbook()
        self.strategy_mapping = self.playbook.get('strategy_mapping', {})
        self.strategies = self.playbook.get('strategies', {})
        self.universes = self.playbook.get('universes', {})

        logger.info(f"Loaded {len(self.strategies)} strategies and {len(self.universes)} universes")



    def get_strategy_for_symbol(self, symbol: str) -> Optional[str]:
        """
        Get the strategy name for a given symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Strategy name or None if not found
        """
        return self.strategy_mapping.get(symbol.upper())

    def get_strategy_config(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Get full configuration for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy configuration dictionary
        """
        return self.strategies.get(strategy_name)

    def get_universe_for_strategy(self, strategy_name: str) -> List[str]:
        """
        Get the symbol universe for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            List of symbols in the strategy's universe
        """
        strategy_config = self.get_strategy_config(strategy_name)
        if not strategy_config:
            return []

        universe_name = strategy_config.get('universe')
        if not universe_name:
            return []

        universe_config = self.universes.get(universe_name, {})
        return universe_config.get('symbols', [])

    def get_all_strategies(self) -> List[str]:
        """Get list of all available strategy names."""
        return list(self.strategies.keys())

    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols across all strategies."""
        all_symbols = set()
        for strategy_name in self.strategies.keys():
            symbols = self.get_universe_for_strategy(strategy_name)
            all_symbols.update(symbols)
        return sorted(list(all_symbols))

    def validate_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """
        Validate a strategy configuration.

        Args:
            strategy_name: Name of the strategy to validate

        Returns:
            Validation results with any issues found
        """
        validation = {
            'valid': True,
            'issues': [],
            'warnings': []
        }

        config = self.get_strategy_config(strategy_name)
        if not config:
            validation['valid'] = False
            validation['issues'].append(f"Strategy '{strategy_name}' not found")
            return validation

        # Check required fields
        required_fields = ['universe', 'horizons', 'features', 'models']
        for field in required_fields:
            if field not in config:
                validation['valid'] = False
                validation['issues'].append(f"Missing required field: {field}")

        # Validate universe exists
        universe_name = config.get('universe')
        if universe_name and universe_name not in self.universes:
            validation['valid'] = False
            validation['issues'].append(f"Universe '{universe_name}' not found")

        # Validate horizons are reasonable
        horizons = config.get('horizons', [])
        if horizons:
            for horizon in horizons:
                if not isinstance(horizon, int) or horizon <= 0:
                    validation['issues'].append(f"Invalid horizon: {horizon} (must be positive integer)")

        # Check ensemble weights sum to 1
        weights = config.get('ensemble_weights', {})
        if weights:
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
                validation['warnings'].append(f"Ensemble weights sum to {total_weight:.3f}, should sum to 1.0")

        return validation

    def get_strategy_summary(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get a summary of strategy configuration.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Summary dictionary with key strategy information
        """
        config = self.get_strategy_config(strategy_name)
        if not config:
            return {'error': f'Strategy {strategy_name} not found'}

        universe_symbols = self.get_universe_for_strategy(strategy_name)
        universe_info = self.universes.get(config.get('universe', {}), {})

        return {
            'name': config.get('name', strategy_name),
            'description': config.get('description', ''),
            'universe': config.get('universe', ''),
            'universe_size': len(universe_symbols),
            'symbols': universe_symbols,
            'horizons': config.get('horizons', []),
            'features': config.get('features', []),
            'models': config.get('models', []),
            'ensemble_weights': config.get('ensemble_weights', {}),
            'risk_limits': config.get('risk_limits', {}),
            'sector': universe_info.get('sector', 'unknown')
        }

    def get_portfolio_config(self) -> Dict[str, Any]:
        """Get global portfolio configuration."""
        return self.playbook.get('global_config', {})

# Convenience function
def create_strategy_selector(playbook_path: str = "config/strategy_playbook.yaml") -> StrategySelector:
    """Create and return a strategy selector instance."""
    return StrategySelector(playbook_path)