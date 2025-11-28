"""
Configuration loader for the agentic forecasting system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

def get_feature_config() -> Dict[str, Any]:
    """
    Load the feature configuration from feature_config.yaml.

    Returns:
        Dictionary containing the feature configuration
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "feature_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Feature config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def get_data_sources_config() -> Dict[str, Any]:
    """
    Load the data sources configuration from config.yaml.

    Returns:
        Dictionary containing the data sources configuration
    """
    config = get_main_config()
    return config.get('data_source', {})

def get_strategy_playbook() -> Dict[str, Any]:
    """
    Load the strategy playbook configuration from config/strategy_config.yaml.

    Returns:
        Dictionary containing the strategy playbook configuration
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "strategy_config.yaml"

    if not config_path.exists():
        # Return default strategy config if file doesn't exist
        return {
            'feature_groups': {
                'price_basic': ['price_returns', 'price_volatility', 'rolling_means'],
                'tech_basic': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands'],
                'seasonality_calendar': ['day_of_week', 'month_of_year', 'quarter'],
                'macro_rates': ['fed_funds_rate', 'rate_changes', 'rate_volatility'],
                'macro_labor': ['unemployment_rate', 'labor_changes'],
                'commodities_gold': ['gold_returns', 'gold_volatility', 'gold_trends'],
                'commodities_oil': ['oil_returns', 'oil_volatility', 'oil_trends'],
                'high_rate_features': ['rate_sensitive_momentum', 'defensive_assets'],
                'low_rate_features': ['growth_momentum', 'risk_assets'],
            },
            'model_strategies': {
                'high_volatility': ['Ensemble', 'AutoNHITS', 'AutoNBEATS'],
                'low_volatility': ['AutoARIMA', 'AutoETS', 'AutoTheta'],
                'trending': ['AutoNHITS', 'AutoNBEATS', 'CNNLSTM'],
                'sideways': ['AutoARIMA', 'AutoETS', 'AutoTheta'],
            }
        }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config