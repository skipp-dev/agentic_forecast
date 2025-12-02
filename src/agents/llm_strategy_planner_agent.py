from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
import json
from langsmith import traceable

logger = logging.getLogger(__name__)

@dataclass
class StrategyPlanningInput:
    strategy_backtests: Dict[str, Any]
    current_regime: Dict[str, Any]
    risk_constraints: Dict[str, Any]
    portfolio_requirements: Dict[str, Any]

@dataclass
class StrategyPlan:
    strategy_rankings: list
    portfolio_recommendations: Dict[str, Any]
    regime_specific_notes: list
    tactical_adjustments: list
    experiments_to_run: list
    implementation_considerations: Dict[str, Any]

class LLMStrategyPlannerAgent:
    """
    Agent that provides strategic recommendations for portfolio construction and strategy allocation.
    Maintains advisory-only posture - recommendations, not commands.
    """
    def __init__(self, settings=None):
        # Use the new LLM factory
        from src.llm.llm_factory import create_strategy_planner_llm
        self.llm = create_strategy_planner_llm()
        self.settings = settings or {}

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
        from src.prompts.llm_prompts import PROMPTS, build_strategy_planner_user_prompt

        system_prompt = PROMPTS["strategy_planner"]
        user_prompt = build_strategy_planner_user_prompt(
            strategy_backtests=planning_input.strategy_backtests,
            current_regime=planning_input.current_regime,
            risk_constraints=planning_input.risk_constraints,
            portfolio_requirements=planning_input.portfolio_requirements
        )

        logger.info("Calling LLM for strategy planning (LangSmith tracing enabled)")

        raw = self.llm.complete(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.3,  # Moderate temperature for creative but reasonable planning
            max_tokens=2000,
        )

        logger.info(f"Raw LLM response (first 500 chars): {raw[:500]}")

        try:
            data = json.loads(raw)
            logger.info("Successfully parsed LLM response as JSON")
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}. Raw response: {raw}")
            # Fallback: create basic strategy plan structure
            data = {
                "strategy_rankings": [
                    {
                        "strategy_name": "Default Strategy",
                        "overall_rank": 1,
                        "performance_score": 0.5,
                        "risk_score": 0.5,
                        "regime_performance": {
                            "bull": "neutral",
                            "bear": "neutral",
                            "sideways": "neutral"
                        },
                        "strengths": ["Stable performance"],
                        "weaknesses": ["Unable to parse detailed analysis"]
                    }
                ],
                "portfolio_recommendations": {
                    "suggested_allocation": {
                        "default_strategy": 1.0
                    },
                    "rationale": "Fallback allocation due to parsing error",
                    "expected_performance": {
                        "annual_return": 0.05,
                        "max_drawdown": 0.10,
                        "sharpe_ratio": 1.0
                    }
                },
                "regime_specific_notes": [
                    {
                        "regime": "current",
                        "recommendations": ["Monitor system performance", "Review strategy backtests"],
                        "expected_impact": "Improved decision making"
                    }
                ],
                "tactical_adjustments": [
                    {
                        "adjustment": "Maintain current allocation",
                        "condition": "Until detailed analysis available",
                        "rationale": "Conservative approach during system issues",
                        "time_horizon": "Immediate"
                    }
                ],
                "experiments_to_run": [
                    {
                        "experiment": "Verify strategy analysis pipeline",
                        "hypothesis": "System can provide detailed strategy recommendations",
                        "expected_impact": "Enhanced portfolio optimization",
                        "resource_requirement": "low"
                    }
                ],
                "implementation_considerations": {
                    "transition_costs": "Minimal during fallback",
                    "monitoring_requirements": "Standard system monitoring",
                    "risk_limits": "Conservative limits applied",
                    "backtest_period": "Current backtest data"
                }
            }

        return StrategyPlan(**data)

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
            md.append(f"### {strategy['strategy_name']} (Rank: {strategy['overall_rank']})")
            md.append(f"- **Performance Score:** {strategy['performance_score']:.2f}")
            md.append(f"- **Risk Score:** {strategy['risk_score']:.2f}")
            md.append("")

            md.append("**Regime Performance:**")
            for regime, perf in strategy['regime_performance'].items():
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
        for strategy_name, weight in portfolio['suggested_allocation'].items():
            md.append(f"- {strategy_name}: {weight:.1%}")
        md.append("")

        md.append(f"**Rationale:** {portfolio['rationale']}")
        md.append("")

        md.append("### Expected Performance")
        perf = portfolio['expected_performance']
        md.append(f"- Annual Return: {perf['annual_return']:.1%}")
        md.append(f"- Max Drawdown: {perf['max_drawdown']:.1%}")
        md.append(f"- Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        md.append("")

        # Regime-Specific Notes
        md.append("## Regime-Specific Notes")
        for note in strategy_plan.regime_specific_notes:
            md.append(f"### {note['regime'].title()} Market Conditions")
            md.append("**Recommendations:**")
            for rec in note.get('recommendations', []):
                md.append(f"  - {rec}")
            md.append(f"**Expected Impact:** {note.get('expected_impact', 'N/A')}")
            md.append("")

        # Tactical Adjustments
        md.append("## Tactical Adjustments")
        for adjustment in strategy_plan.tactical_adjustments:
            md.append(f"### {adjustment['adjustment']}")
            md.append(f"- **Condition:** {adjustment['condition']}")
            md.append(f"- **Rationale:** {adjustment['rationale']}")
            md.append(f"- **Time Horizon:** {adjustment['time_horizon']}")
            md.append("")

        # Experiments to Run
        md.append("## Recommended Experiments")
        for experiment in strategy_plan.experiments_to_run:
            md.append(f"### {experiment['experiment']}")
            md.append(f"- **Hypothesis:** {experiment['hypothesis']}")
            md.append(f"- **Expected Impact:** {experiment['expected_impact']}")
            md.append(f"- **Resource Requirement:** {experiment['resource_requirement']}")
            md.append("")

        # Implementation Considerations
        md.append("## Implementation Considerations")
        impl = strategy_plan.implementation_considerations
        md.append(f"- **Transition Costs:** {impl['transition_costs']}")
        md.append(f"- **Monitoring Requirements:** {impl['monitoring_requirements']}")
        md.append(f"- **Risk Limits:** {impl['risk_limits']}")
        md.append(f"- **Backtest Period:** {impl['backtest_period']}")
        md.append("")

        md.append("---")
        md.append("*This is an advisory analysis. All recommendations should be validated through backtesting and risk assessment before implementation.*")

        return "\n".join(md)