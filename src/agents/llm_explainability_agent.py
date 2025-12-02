from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import json
from langsmith import traceable

logger = logging.getLogger(__name__)

@dataclass
class ExplainabilityInput:
    symbol: str
    horizon: int
    forecast_return: float
    actual_return: Optional[float]
    model_family: str
    feature_importance: Dict[str, float]
    regime_context: Dict[str, Any]
    guardrails_active: List[str]

@dataclass
class ForecastExplanation:
    symbol: str
    horizon: int
    prediction_summary: str
    key_drivers: List[Dict[str, Any]]
    regime_influence: str
    uncertainty_factors: List[str]
    confidence_assessment: Dict[str, Any]
    practical_insights: str
    monitoring_notes: str
    caveats: List[str]

class LLMExplainabilityAgent:
    """
    Agent that provides human-understandable explanations for individual forecasts.
    """
    def __init__(self, settings=None):
        # Use the new LLM factory
        from src.llm.llm_factory import create_explainability_agent_llm
        self.llm = create_explainability_agent_llm()
        self.settings = settings or {}

    @traceable(
        name="explainability_agent_explain_forecast",
        tags=["explainability", "llm", "forecast"],
        metadata={"role": "explainability_agent"}
    )
    def explain_forecast(self, explanation_input: ExplainabilityInput) -> ForecastExplanation:
        """
        Generate a human-understandable explanation for a specific forecast.
        This call is traced to LangSmith.
        """
        from src.prompts.llm_prompts import PROMPTS, build_explainability_agent_user_prompt

        system_prompt = PROMPTS["explainability_agent"]
        user_prompt = build_explainability_agent_user_prompt(
            symbol=explanation_input.symbol,
            horizon=explanation_input.horizon,
            forecast_return=explanation_input.forecast_return,
            actual_return=explanation_input.actual_return,
            model_family=explanation_input.model_family,
            feature_importance=explanation_input.feature_importance,
            regime_context=explanation_input.regime_context,
            guardrails_active=explanation_input.guardrails_active
        )

        logger.info(f"Calling LLM for forecast explanation: {explanation_input.symbol} (LangSmith tracing enabled)")

        raw = self.llm.complete(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.1,  # Lower temperature for more consistent explanations
            max_tokens=1500,
        )

        logger.info(f"Raw LLM response (first 500 chars): {raw[:500]}")

        try:
            data = json.loads(raw)
            logger.info("Successfully parsed LLM response as JSON")
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}. Raw response: {raw}")
            # Fallback: create a basic explanation
            data = {
                "symbol": explanation_input.symbol,
                "horizon": explanation_input.horizon,
                "prediction_summary": f"The {explanation_input.model_family} model predicts a {explanation_input.forecast_return:.1%} return for {explanation_input.symbol} over {explanation_input.horizon} days.",
                "key_drivers": [
                    {
                        "feature": "Unable to parse feature data",
                        "importance": 0.0,
                        "explanation": "Feature importance data could not be processed"
                    }
                ],
                "regime_influence": f"Current regime context: {explanation_input.regime_context}",
                "uncertainty_factors": ["Unable to parse uncertainty analysis"],
                "confidence_assessment": {
                    "level": "unknown",
                    "score": 0.5,
                    "rationale": "Unable to parse confidence assessment"
                },
                "practical_insights": "Unable to generate practical insights due to parsing error",
                "monitoring_notes": "Monitor for changes in key drivers and regime conditions",
                "caveats": ["This explanation was generated due to a parsing error and may be incomplete"]
            }

        return ForecastExplanation(**data)

    def explain_multiple_forecasts(self, forecasts: List[ExplainabilityInput]) -> List[ForecastExplanation]:
        """
        Explain multiple forecasts in batch.
        Useful for generating explanations for a dashboard or report.
        """
        explanations = []
        for forecast_input in forecasts:
            try:
                explanation = self.explain_forecast(forecast_input)
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Failed to explain forecast for {forecast_input.symbol}: {e}")
                # Create a minimal fallback explanation
                fallback = ForecastExplanation(
                    symbol=forecast_input.symbol,
                    horizon=forecast_input.horizon,
                    prediction_summary=f"Unable to generate explanation due to error: {e}",
                    key_drivers=[],
                    regime_influence="Unknown",
                    uncertainty_factors=["Error occurred"],
                    confidence_assessment={"level": "unknown", "score": 0.0, "rationale": "Error"},
                    practical_insights="Unable to analyze",
                    monitoring_notes="Monitor system health",
                    caveats=["Explanation generation failed"]
                )
                explanations.append(fallback)

        return explanations

    def format_explanation_as_html(self, explanation: ForecastExplanation) -> str:
        """
        Format the explanation as an HTML snippet for UI display.
        """
        html = []

        html.append(f"<div class='forecast-explanation'>")
        html.append(f"<h4>{explanation.symbol} - {explanation.horizon} Day Forecast</h4>")

        html.append(f"<div class='prediction-summary'>")
        html.append(f"<strong>Prediction:</strong> {explanation.prediction_summary}")
        html.append(f"</div>")

        html.append(f"<div class='confidence'>")
        conf = explanation.confidence_assessment
        html.append(f"<strong>Confidence:</strong> {conf['level'].title()} ({conf['score']:.0%})")
        html.append(f"<br><em>{conf['rationale']}</em>")
        html.append(f"</div>")

        html.append(f"<div class='key-drivers'>")
        html.append(f"<strong>Key Drivers:</strong>")
        html.append(f"<ul>")
        for driver in explanation.key_drivers[:3]:  # Top 3 drivers
            html.append(f"<li>{driver['feature']}: {driver['explanation']}</li>")
        html.append(f"</ul>")
        html.append(f"</div>")

        html.append(f"<div class='regime-influence'>")
        html.append(f"<strong>Regime Influence:</strong> {explanation.regime_influence}")
        html.append(f"</div>")

        html.append(f"<div class='practical-insights'>")
        html.append(f"<strong>Practical Insights:</strong> {explanation.practical_insights}")
        html.append(f"</div>")

        if explanation.caveats:
            html.append(f"<div class='caveats'>")
            html.append(f"<strong>Caveats:</strong>")
            html.append(f"<ul>")
            for caveat in explanation.caveats:
                html.append(f"<li>{caveat}</li>")
            html.append(f"</ul>")
            html.append(f"</div>")

        html.append(f"</div>")

        return "\n".join(html)