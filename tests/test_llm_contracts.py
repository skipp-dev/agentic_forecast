
import json
from typing import Any, Dict, List
import pytest
from dataclasses import asdict

from src.agents.llm_hpo_planner_agent import (
    LLMHPOPlannerAgent,
    HPOPlanInput,
    HPORun,
    HPOPlan,
)
from src.agents.analytics_explainer import (
    LLMAnalyticsExplainerAgent,
    AnalyticsExplanation,
)
from src.agents.llm_news_agent import (
    LLMNewsFeatureAgent,
    RawNewsItem,
    EnrichedNewsFeature
)
from src.agents.llm_strategy_planner_agent import (
    LLMStrategyPlannerAgent,
    StrategyPlan,
    StrategyPlanningInput
)

class StubLLMClient:
    """
    Minimal stub LLM client that mimics the interface used by your agents.
    It always returns the given raw_response string, regardless of the call.
    """
    def __init__(self, raw_response: str):
        self.raw_response = raw_response

    def complete(self, *args, **kwargs) -> str:
        return self.raw_response

    def generate(self, *args, **kwargs) -> str:
        return self.raw_response

    def chat(self, *args, **kwargs) -> str:
        # Some agents might expect an object with .content or .text
        class Response:
            def __init__(self, content):
                self.content = content
                self.text = content
            def __str__(self):
                return self.content
        return Response(self.raw_response)

# ---------------------------------------------------------------------------
# HPO PLANNER CONTRACT TESTS
# ---------------------------------------------------------------------------

def _make_sample_hpo_input() -> HPOPlanInput:
    past_runs: List[HPORun] = [
        HPORun(
            model_family="neuralforecast_nhits",
            trial_id="trial_1",
            params={"learning_rate": 1e-3},
            metric=0.015,
            directional_accuracy=0.62,
            status="completed",
        )
    ]
    performance_summary: List[Dict[str, Any]] = [
        {"symbol": "AAPL", "horizon": 5, "family": "neuralforecast_nhits", "mape": 0.015}
    ]
    return HPOPlanInput(
        past_runs=past_runs,
        performance_summary=performance_summary,
        total_hpo_budget=40,
        per_family_min_trials=5,
        per_family_max_trials=20,
    )


def test_hpo_planner_valid_json():
    plan_input = _make_sample_hpo_input()

    valid_plan = {
        "symbols_to_focus": ["AAPL", "NVDA"],
        "horizons_to_focus": [1, 5],
        "families_to_prioritize": ["neuralforecast_nhits", "neuralforecast_tft"],
        "per_family_search_spaces": {
            "neuralforecast_nhits": {"learning_rate": [1e-4, 1e-3, 3e-3]},
            "neuralforecast_tft": {"learning_rate": [1e-4, 3e-4, 1e-3]},
        },
        "budget_allocation": {
            "neuralforecast_nhits": 20,
            "neuralforecast_tft": 20,
        },
        "symbol_family_overrides": [
            {"symbol": "AAPL", "horizon": 5, "families": ["neuralforecast_nhits"]}
        ],
        "notes": "Test HPO plan from stub.",
    }

    stub = StubLLMClient(json.dumps(valid_plan))
    agent = LLMHPOPlannerAgent(llm_client=stub)

    plan: HPOPlan = agent.plan(plan_input)

    assert isinstance(plan, HPOPlan)
    assert plan.symbols_to_focus == ["AAPL", "NVDA"]
    assert 5 in plan.horizons_to_focus
    assert "neuralforecast_nhits" in plan.families_to_prioritize
    assert "neuralforecast_nhits" in plan.per_family_search_spaces
    assert plan.budget_allocation.get("neuralforecast_nhits") == 20
    assert "Test HPO plan" in plan.notes


def test_hpo_planner_missing_fields_defaults():
    plan_input = _make_sample_hpo_input()

    # Only one field returned; others should be filled from dataclass defaults
    partial_plan = {
        "symbols_to_focus": ["AAPL"],
    }

    stub = StubLLMClient(json.dumps(partial_plan))
    agent = LLMHPOPlannerAgent(llm_client=stub)

    plan: HPOPlan = agent.plan(plan_input)

    assert isinstance(plan, HPOPlan)
    assert plan.symbols_to_focus == ["AAPL"]
    # Everything else should fall back to defaults
    assert plan.horizons_to_focus == []
    assert plan.families_to_prioritize == []
    assert plan.per_family_search_spaces == {}
    assert plan.budget_allocation == {}
    assert plan.symbol_family_overrides == []
    # notes may be "" here, depending on your parser
    assert isinstance(plan.notes, str)


def test_hpo_planner_invalid_json_safe_fallback():
    plan_input = _make_sample_hpo_input()

    # Completely invalid JSON
    stub = StubLLMClient("NOT_JSON >>> <<<")
    agent = LLMHPOPlannerAgent(llm_client=stub)

    plan: HPOPlan = agent.plan(plan_input)

    assert isinstance(plan, HPOPlan)
    # Fallback plan is a safe no-op
    assert plan.symbols_to_focus == []
    assert plan.horizons_to_focus == []
    assert plan.families_to_prioritize == []
    assert plan.per_family_search_spaces == {}
    assert plan.budget_allocation == {}
    assert "Failed to parse plan" in plan.notes

# ---------------------------------------------------------------------------
# ANALYTICS EXPLAINER CONTRACT TESTS
# ---------------------------------------------------------------------------

def _make_dummy_metrics_payload() -> Dict[str, Any]:
    return {
        "run_metadata_json": "{}",
        "metrics_global_json": '{"mape": 0.02, "mae": 1.5}',
        "per_symbol_metrics_json": "[]",
        "regime_metrics_json": "[]",
        "feature_importance_json": "[]",
        "guardrail_summary_json": "{}",
    }


def test_analytics_explainer_valid_json():
    metrics_payload = _make_dummy_metrics_payload()

    valid_explanation = {
        "global_summary": "Models look healthy overall; no major regressions.",
        "metric_explanations": {
            "mae": "MAE is stable across most symbols.",
            "mape": "MAPE remains below 2% for the majority.",
            "smape": "SMAPE shows similar behaviour to MAPE.",
            "swase": "SWASE slightly elevated in shock regime.",
            "directional_accuracy": "Directional accuracy around 65–70%.",
        },
        "regime_insights": [
            {
                "regime": "peer_shock_flag == 1",
                "performance_comment": "Errors higher in shock regime.",
                "risk_comment": "Forecasts should be used cautiously in shock periods.",
            }
        ],
        "symbol_outliers": [
            {
                "symbol": "AAPL",
                "horizon": 5,
                "issue": "Very high MAPE in shock regime",
                "comment": "Likely driven by earnings gap.",
            }
        ],
        "feature_insights": {
            "overall_top_features": [
                {"name": "rsi_14", "importance_comment": "RSI 14 seems highly predictive overall."}
            ],
            "shock_regime_top_features": [
                {"name": "vix_level", "importance_comment": "Volatility index matters in shocks."}
            ],
        },
        "recommendations": [
            {
                "category": "HPO",
                "action": "Run HPO for [AAPL, NVDA] on 5d horizon.",
                "reason": "Underperformance on medium-term horizon.",
            }
        ],
    }

    stub = StubLLMClient(json.dumps(valid_explanation))
    agent = LLMAnalyticsExplainerAgent(settings={}, llm_client=stub)

    expl: AnalyticsExplanation = agent.explain_metrics(metrics_payload)

    assert isinstance(expl, AnalyticsExplanation)
    assert isinstance(expl.metric_explanations, type(expl.metric_explanations))
    assert expl.global_summary.startswith("Models look healthy")
    assert expl.metric_explanations.mape != ""
    assert isinstance(expl.regime_insights, list)
    assert isinstance(expl.feature_insights, dict)


def test_analytics_explainer_missing_fields_defaults():
    metrics_payload = _make_dummy_metrics_payload()

    # Only global_summary returned
    partial = {
        "global_summary": "Only summary provided.",
    }

    stub = StubLLMClient(json.dumps(partial))
    agent = LLMAnalyticsExplainerAgent(settings={}, llm_client=stub)

    expl: AnalyticsExplanation = agent.explain_metrics(metrics_payload)

    assert isinstance(expl, AnalyticsExplanation)
    assert expl.global_summary == "Only summary provided."
    # metric_explanations must be a MetricExplanations instance with defaults
    assert hasattr(expl.metric_explanations, "mape")
    assert expl.metric_explanations.mape == ""
    assert isinstance(expl.regime_insights, list)
    assert isinstance(expl.feature_insights, dict)
    assert isinstance(expl.recommendations, list)


def test_analytics_explainer_invalid_json_safe_fallback():
    metrics_payload = _make_dummy_metrics_payload()

    stub = StubLLMClient("THIS IS NOT JSON")
    agent = LLMAnalyticsExplainerAgent(settings={}, llm_client=stub)

    expl: AnalyticsExplanation = agent.explain_metrics(metrics_payload)

    assert isinstance(expl, AnalyticsExplanation)
    # In your current fallback, global_summary is raw text → check it’s non-empty
    assert expl.global_summary != ""
    assert isinstance(expl.metric_explanations, type(expl.metric_explanations))
    assert isinstance(expl.regime_insights, list)

# ---------------------------------------------------------------------------
# NEWS AGENT CONTRACT TESTS
# ---------------------------------------------------------------------------

def test_news_agent_contract_valid_json():
    valid = {
        "categories": ["earnings", "macro"],
        "directional_impact": "bullish",
        "impact_horizon": "1-5d",
        "volatility_impact": "medium",
        "confidence": 0.8,
        "notes": "Strong earnings beat.",
        "sentiment_score": 0.75
    }

    stub = StubLLMClient(json.dumps(valid))
    agent = LLMNewsFeatureAgent(llm_client=stub)

    item = RawNewsItem(
        symbol="AAPL",
        timestamp="2025-12-03T10:00:00Z",
        headline="AAPL beats earnings",
        body="Apple reported strong earnings...",
        provider="Bloomberg"
    )

    enriched = agent.enrich_item(item)
    
    assert isinstance(enriched, EnrichedNewsFeature)
    assert enriched.symbol == "AAPL"
    assert enriched.directional_impact == "bullish"
    assert enriched.sentiment_score == 0.75
    assert "earnings" in enriched.categories

def test_news_agent_contract_invalid_json():
    stub = StubLLMClient("NOT JSON")
    agent = LLMNewsFeatureAgent(llm_client=stub)

    item = RawNewsItem(
        symbol="AAPL",
        timestamp="2025-12-03T10:00:00Z",
        headline="AAPL beats earnings",
        body="Apple reported strong earnings...",
        provider="Bloomberg"
    )

    enriched = agent.enrich_item(item)
    
    assert isinstance(enriched, EnrichedNewsFeature)
    assert enriched.directional_impact == "neutral" # Default
    assert enriched.sentiment_score == 0.0 # Default

# ---------------------------------------------------------------------------
# STRATEGY PLANNER CONTRACT TESTS
# ---------------------------------------------------------------------------

def _make_strategy_input() -> StrategyPlanningInput:
    return StrategyPlanningInput(
        strategy_backtests={},
        current_regime={},
        risk_constraints={},
        portfolio_requirements={}
    )

def test_strategy_planner_contract_valid_json():
    valid = {
        "strategy_rankings": [
            {"strategy_name": "MeanReversion", "overall_rank": 1, "performance_score": 0.8, "risk_score": 0.7, "allocation_recommendation": 0.6},
            {"strategy_name": "TrendFollower", "overall_rank": 2, "performance_score": 0.7, "risk_score": 0.6, "allocation_recommendation": 0.4},
        ],
        "portfolio_recommendations": {
            "suggested_allocation": {"MeanReversion": 0.6, "TrendFollower": 0.4},
            "rationale": "Diversification",
            "expected_performance": {"annual_return": 0.15}
        },
        "regime_specific_notes": [],
        "tactical_adjustments": [],
        "risk_considerations": {"max_drawdown_focus": True},
        "experiments_to_run": ["Test new signal"],
        "implementation_considerations": {},
    }

    stub = StubLLMClient(json.dumps(valid))
    agent = LLMStrategyPlannerAgent(settings={}, llm_client=stub)

    plan: StrategyPlan = agent.generate_strategy_plan(_make_strategy_input())
    assert isinstance(plan, StrategyPlan)
    assert len(plan.strategy_rankings) == 2
    assert plan.portfolio_recommendations["suggested_allocation"]["MeanReversion"] == 0.6
    assert "Test new signal" in plan.experiments_to_run

def test_strategy_planner_contract_invalid_json():
    stub = StubLLMClient("NOT JSON")
    agent = LLMStrategyPlannerAgent(settings={}, llm_client=stub)

    plan: StrategyPlan = agent.generate_strategy_plan(_make_strategy_input())
    assert isinstance(plan, StrategyPlan)
    # Check fallback values
    assert len(plan.strategy_rankings) == 1
    assert plan.strategy_rankings[0]["strategy_name"] == "Default Strategy"
    assert "Fallback" in plan.portfolio_recommendations["rationale"]
