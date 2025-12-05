from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import json
import yaml
import os
from langsmith import traceable
from src.utils.llm_utils import extract_json_from_llm_output

logger = logging.getLogger(__name__)

@dataclass
class StrategyPlanningInput:
    strategy_backtests: Dict[str, Any]
    current_regime: Dict[str, Any]
    risk_constraints: Dict[str, Any]
    portfolio_requirements: Dict[str, Any]

@dataclass
class StrategyPlan:
    strategy_rankings: List[Dict[str, Any]]
    portfolio_recommendations: Dict[str, Any]
    regime_specific_notes: List[Dict[str, Any]]
    tactical_adjustments: List[Dict[str, Any]]
    risk_considerations: Dict[str, Any]
    experiments_to_run: List[str]
    implementation_considerations: Dict[str, Any]

class LLMStrategyPlannerAgent:
    """
    Agent that provides strategic recommendations for portfolio construction and strategy allocation.
    Maintains advisory-only posture - recommendations, not commands.
    """
    def __init__(self, settings=None, llm_client=None):
        if llm_client is not None:
            self.llm = llm_client
        else:
            # Use the new LLM factory
            from src.llm.llm_factory import create_strategy_planner_llm
            self.llm = create_strategy_planner_llm()
        self.settings = settings or {}
        self.strategy_config = self._load_strategy_config()

    def _load_strategy_config(self) -> Dict[str, Any]:
        """Load valid strategies from config."""
        config_path = os.path.join("config", "strategies.yaml")
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    return config.get("strategies", {})
            else:
                logger.warning(f"Strategy config not found at {config_path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load strategy config: {e}")
            return {}

    @traceable(
        name="strategy_planner_generate_plan",
        tags=["strategy", "llm", "planning"],
        metadata={"role": "strategy_planner"}
    )
    def generate_strategy_plan(self, planning_input: StrategyPlanningInput) -> StrategyPlan:
        """
        Generate strategic portfolio recommendations based on backtest performance and market conditions.
        This call is traced to LangSmith.
        """
        from src.configs.llm_prompts import PROMPTS, build_strategy_planner_user_prompt

        system_prompt = PROMPTS["strategy_planner"]
        # Inject valid strategies into prompt context if possible, or just validate output
        valid_strategy_names = list(self.strategy_config.keys())
        
        user_prompt = build_strategy_planner_user_prompt(
            strategy_backtests=planning_input.strategy_backtests,
            current_regime=planning_input.current_regime,
            risk_constraints=planning_input.risk_constraints,
            portfolio_requirements=planning_input.portfolio_requirements
        )
        
        if valid_strategy_names:
            user_prompt += f"\n\nIMPORTANT: You must ONLY recommend strategies from this allowed list: {', '.join(valid_strategy_names)}."

        logger.info("Calling LLM for strategy planning (LangSmith tracing enabled)")

        raw = self.llm.complete(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.3,  # Moderate temperature for creative but reasonable planning
            max_tokens=2000,
        )

        logger.info(f"Raw LLM response (first 500 chars): {raw[:500]}")

        try:
            json_str = extract_json_from_llm_output(raw)
            data = json.loads(json_str)
            logger.info("Successfully parsed LLM response as JSON")
            
            # Validate strategies
            if self.strategy_config:
                validated_rankings = []
                for strat in data.get("strategy_rankings", []):
                    name = strat.get("strategy_name")
                    if name in self.strategy_config:
                        validated_rankings.append(strat)
                    else:
                        logger.warning(f"LLM suggested invalid strategy '{name}'. Dropping.")
                
                if not validated_rankings:
                    logger.warning("No valid strategies returned. Falling back to default_balanced.")
                    validated_rankings.append({
                        "strategy_name": "default_balanced",
                        "overall_rank": 1,
                        "allocation_recommendation": 1.0,
                        "rationale": "Fallback due to invalid LLM output"
                    })
                
                data["strategy_rankings"] = validated_rankings

        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}. Raw response: {raw}")
            # Fallback: create basic strategy plan structure matching prompt schema
            data = {
                "strategy_rankings": [
                    {
                        "strategy_name": "default_balanced",
                        "overall_rank": 1,
                        "performance_score": 0.5,
                        "risk_score": 0.5,
                        "regime_performance": {
                            "bull": "neutral",
                            "bear": "neutral",
                            "high_volatility": "neutral",
                            "low_volatility": "neutral"
                        },
                        "strengths": ["Stable performance"],
                        "weaknesses": ["Unable to parse detailed analysis"],
                        "allocation_recommendation": 1.0
                    }
                ],
                "portfolio_recommendations": {
                    "suggested_allocation": {
                        "default_balanced": 1.0
                    },
                    "rationale": "Fallback allocation due to parsing error",
                    "expected_performance": {
                        "annual_return": 0.05,
                        "annual_volatility": 0.10,
                        "sharpe_ratio": 0.5,
                        "max_drawdown": 0.10,
                        "sortino_ratio": 0.5
                    },
                    "diversification_metrics": {
                        "correlation_matrix": "N/A",
                        "concentration_risk": "high",
                        "regime_coverage": "poor"
                    }
                },
                "regime_specific_notes": [
                    {
                        "regime": "current",
                        "strategies_to_prefer": ["Default Strategy"],
                        "strategies_to_avoid": [],
                        "comment": "Fallback due to parse error."
                    }
                ],
                "tactical_adjustments": [
                    {
                        "adjustment": "Maintain current allocation",
                        "condition": "Until detailed analysis available",
                        "rationale": "Conservative approach during system issues",
                        "time_horizon": "Immediate",
                        "expected_impact": "Stability"
                    }
                ],
                "risk_considerations": {
                    "portfolio_risk_metrics": {
                        "value_at_risk_95": 0.05,
                        "expected_shortfall_95": 0.07,
                        "stress_test_results": "N/A"
                    },
                    "risk_mitigation_suggestions": ["Monitor manually"],
                    "tail_risk_assessment": "Unknown"
                },
                "experiments_to_run": [
                    "Verify strategy analysis pipeline"
                ],
                "implementation_considerations": {
                    "transition_costs": "Minimal",
                    "monitoring_requirements": "Standard",
                    "risk_limits": "Conservative",
                    "backtest_period": "N/A"
                }
            }

        # Validate and filter
        valid_keys = StrategyPlan.__annotations__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        
        # Ensure required fields
        if "risk_considerations" not in filtered:
            filtered["risk_considerations"] = {}
        if "experiments_to_run" not in filtered:
            filtered["experiments_to_run"] = []

        return StrategyPlan(**filtered)

    def format_strategy_report(self, strategy_plan: StrategyPlan) -> str:
        """
        Format the strategy plan as a comprehensive report.
        """
        md = []

        # Title
        md.append("# Strategic Portfolio Recommendations")
        md.append("")

        # Strategy Rankings
        md.append("## Strategy Rankings")
        for strategy in strategy_plan.strategy_rankings:
            md.append(f"### {strategy.get('strategy_name', 'Unknown')} (Rank: {strategy.get('overall_rank', 'N/A')})")
            md.append(f"- **Performance Score:** {strategy.get('performance_score', 0):.2f}")
            md.append(f"- **Risk Score:** {strategy.get('risk_score', 0):.2f}")
            md.append(f"- **Allocation:** {strategy.get('allocation_recommendation', 0):.1%}")
            md.append("")

            md.append("**Regime Performance:**")
            for regime, perf in strategy.get('regime_performance', {}).items():
                md.append(f"  - {regime.title()}: {perf}")
            md.append("")

            md.append("**Strengths:**")
            for strength in strategy.get('strengths', []):
                md.append(f"  - {strength}")
            md.append("")

            md.append("**Weaknesses:**")
            for weakness in strategy.get('weaknesses', []):
                md.append(f"  - {weakness}")
            md.append("")

        # Portfolio Recommendations
        md.append("## Portfolio Recommendations")
        portfolio = strategy_plan.portfolio_recommendations
        md.append("### Suggested Allocation")
        for strategy_name, weight in portfolio.get('suggested_allocation', {}).items():
            md.append(f"- {strategy_name}: {weight:.1%}")
        md.append("")

        md.append(f"**Rationale:** {portfolio.get('rationale', '')}")
        md.append("")

        md.append("### Expected Performance")
        perf = portfolio.get('expected_performance', {})
        md.append(f"- Annual Return: {perf.get('annual_return', 0):.1%}")
        md.append(f"- Max Drawdown: {perf.get('max_drawdown', 0):.1%}")
        md.append(f"- Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
        md.append("")

        # Regime-Specific Notes
        md.append("## Regime-Specific Notes")
        for note in strategy_plan.regime_specific_notes:
            md.append(f"### {note.get('regime', 'Unknown').title()} Market Conditions")
            md.append("**Strategies to Prefer:**")
            for s in note.get('strategies_to_prefer', []):
                md.append(f"  - {s}")
            md.append("**Strategies to Avoid:**")
            for s in note.get('strategies_to_avoid', []):
                md.append(f"  - {s}")
            md.append(f"**Comment:** {note.get('comment', '')}")
            md.append("")

        # Tactical Adjustments
        md.append("## Tactical Adjustments")
        for adjustment in strategy_plan.tactical_adjustments:
            md.append(f"### {adjustment.get('adjustment', 'Adjustment')}")
            md.append(f"- **Condition:** {adjustment.get('condition', '')}")
            md.append(f"- **Rationale:** {adjustment.get('rationale', '')}")
            md.append(f"- **Time Horizon:** {adjustment.get('time_horizon', '')}")
            md.append("")

        # Risk Considerations
        md.append("## Risk Considerations")
        risk = strategy_plan.risk_considerations
        metrics = risk.get('portfolio_risk_metrics', {})
        md.append("### Portfolio Risk Metrics")
        md.append(f"- VaR (95%): {metrics.get('value_at_risk_95', 'N/A')}")
        md.append(f"- Expected Shortfall (95%): {metrics.get('expected_shortfall_95', 'N/A')}")
        md.append("")
        
        md.append("### Mitigation Suggestions")
        for sugg in risk.get('risk_mitigation_suggestions', []):
            md.append(f"- {sugg}")
        md.append("")

        # Experiments to Run
        md.append("## Recommended Experiments")
        for experiment in strategy_plan.experiments_to_run:
            # experiment is a string in the new schema
            md.append(f"- {experiment}")
        md.append("")

        # Implementation Considerations
        md.append("## Implementation Considerations")
        impl = strategy_plan.implementation_considerations
        md.append(f"- **Transition Costs:** {impl.get('transition_costs', 'N/A')}")
        md.append(f"- **Monitoring Requirements:** {impl.get('monitoring_requirements', 'N/A')}")
        md.append(f"- **Risk Limits:** {impl.get('risk_limits', 'N/A')}")
        md.append(f"- **Backtest Period:** {impl.get('backtest_period', 'N/A')}")
        md.append("")

        md.append("---")
        md.append("*This is an advisory analysis. All recommendations should be validated through backtesting and risk assessment before implementation.*")

        return "\n".join(md)