import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from .risk_management_agent import RiskManagementAgent

logger = logging.getLogger(__name__)

class PortfolioManagerAgent:
    """
    Specialized agent for portfolio construction and optimization.
    Takes trading signals and constructs a target portfolio.
    """
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.risk_agent = RiskManagementAgent(self.config.get('risk', {}))
        self.max_position_size = self.config.get('max_position_size', 0.2) # Max 20% per asset
        self.target_cash = self.config.get('target_cash', 0.0) # Fully invested by default

    def construct_portfolio(self, recommended_actions: List[str], best_models: Dict, raw_data: Dict) -> Dict[str, Any]:
        """
        Construct a target portfolio based on recommendations and risk constraints.
        """
        logger.info("Constructing portfolio from recommendations...")
        
        # Parse recommendations to get initial weights
        # Expected action format: "Promote {model} for {symbol} (Trust: {score}, Size: {mult}x)"
        # or "Hold ..."
        
        target_weights = {}
        
        for symbol, model_info in best_models.items():
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
            # Simple fallback: Reduce exposure by 50%
            for s in target_weights:
                target_weights[s] *= 0.5
            
            risk_assessment = self.risk_agent.assess_portfolio_risk(target_weights, raw_data)
            risk_assessment['note'] = "Portfolio scaled down due to risk limits"

        return {
            'target_weights': target_weights,
            'risk_metrics': risk_assessment.get('metrics', {}),
            'risk_status': risk_assessment.get('reason', 'OK')
        }
