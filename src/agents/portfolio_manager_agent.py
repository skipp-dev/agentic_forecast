import pandas as pd
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from .risk_management_agent import RiskManagementAgent
from src.data.model_registry import ModelRegistry
from src.schemas import PortfolioAllocation

try:
    from ..risk.events import portfolio_rejection_event
except ImportError:
    # Fallback for when running tests or different path structure
    from src.risk.events import portfolio_rejection_event

logger = logging.getLogger(__name__)

class PortfolioManagerAgent:
    """
    Specialized agent for portfolio construction and optimization.
    Takes trading signals and constructs a target portfolio.
    """
    def __init__(self, config: Dict = None, registry: ModelRegistry = None):
        self.config = config or {}
        self.risk_agent = RiskManagementAgent(self.config.get('risk', {}))
        self.max_position_size = self.config.get('max_position_size', 0.2) # Max 20% per asset
        self.target_cash = self.config.get('target_cash', 0.0) # Fully invested by default
        self.registry = registry or ModelRegistry()

    def construct_portfolio(self, recommended_actions: List[str], best_models: Dict, raw_data: Dict) -> PortfolioAllocation:
        """
        Construct a target portfolio based on recommendations and risk constraints.
        Enforces TTL (Time-To-Live) checks on forecasts.
        """
        logger.info("Constructing portfolio from recommendations...")
        
        # 0. Kill Switch Check
        if not self.registry.get_trading_status():
            logger.critical("KILL SWITCH ACTIVE: Trading is disabled globally.")
            return PortfolioAllocation(
                target_weights={},
                risk_metrics={},
                risk_status='KILL_SWITCH_ACTIVE',
                risk_events=[
                    asdict(portfolio_rejection_event(
                        reason="Kill Switch Active",
                        details={'message': "Trading disabled by administrator"}
                    ))
                ]
            )
        
        # Parse recommendations to get initial weights
        # Expected action format: "Promote {model} for {symbol} (Trust: {score}, Size: {mult}x)"
        # or "Hold ..."
        
        target_weights = {}
        generated_events = []
        current_time = time.time()
        
        for symbol, model_info in best_models.items():
            # TTL / Staleness Check
            champion_details = self.registry.get_champion_details(symbol)
            valid_until = champion_details.get('valid_until')
            
            if valid_until and current_time > valid_until:
                msg = f"Blocking trade for {symbol}: Forecast expired at {valid_until} (Current: {current_time})"
                logger.warning(msg)
                
                # Emit risk event for staleness
                event = portfolio_rejection_event(
                    reason="Stale Forecast",
                    details={
                        'symbol': symbol,
                        'valid_until': valid_until,
                        'current_time': current_time
                    }
                )
                generated_events.append(asdict(event))
                continue # Skip this symbol

            # Find matching action
            action = next((a for a in recommended_actions if symbol in a), None)
            
            weight = 0.0
            if action and "Promote" in action:
                # Extract size multiplier if present
                mult = 1.0
                if "Size:" in action:
                    try:
                        part = action.split("Size:")[1].split("x")[0].strip()
                        mult = float(part)
                    except:
                        pass
                
                # Base weight (equal weight assumption for now, could be optimized)
                weight = 0.1 * mult # Start with 10% base
            
            target_weights[symbol] = weight

        # Normalize weights to sum to (1 - target_cash)
        total_weight = sum(target_weights.values())
        if total_weight > 0:
            scale_factor = (1.0 - self.target_cash) / total_weight
            # Apply max position size constraint
            for s in target_weights:
                target_weights[s] = min(target_weights[s] * scale_factor, self.max_position_size)
            
            # Re-normalize if we hit caps (simple pass)
            current_sum = sum(target_weights.values())
            if current_sum < (1.0 - self.target_cash):
                # Distribute remaining? Or keep as cash?
                # For safety, keep as cash
                pass
        
        # Risk Check
        risk_assessment = self.risk_agent.assess_portfolio_risk(target_weights, raw_data)
        
        if not risk_assessment['risk_approved']:
            logger.warning(f"Portfolio rejected by Risk Agent: {risk_assessment['reason']}")
            
            # Create risk event
            event = portfolio_rejection_event(
                reason=risk_assessment['reason'],
                details={
                    'initial_weights': target_weights.copy(),
                    'metrics': risk_assessment.get('metrics', {})
                }
            )
            generated_events.append(asdict(event))

            # Simple fallback: Reduce exposure by 50%
            for s in target_weights:
                target_weights[s] *= 0.5
            
            risk_assessment = self.risk_agent.assess_portfolio_risk(target_weights, raw_data)
            risk_assessment['note'] = "Portfolio scaled down due to risk limits"

        return PortfolioAllocation(
            target_weights=target_weights,
            risk_metrics=risk_assessment.get('metrics', {}),
            risk_status=risk_assessment.get('reason', 'OK'),
            risk_events=generated_events
        )
