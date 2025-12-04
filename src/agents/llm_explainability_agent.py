from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import json
from langsmith import traceable
from src.utils.llm_utils import extract_json_from_llm_output

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
class LocalExplanation:
    symbol: str
    horizon: int
    forecast_comment: str
    top_feature_drivers: List[Dict[str, Any]] = field(default_factory=list)
    regime_and_risk_comment: str = ""
    limitations: List[str] = field(default_factory=list)

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
    def explain_forecast(self, explanation_input: ExplainabilityInput) -> LocalExplanation:
        """
        Generate a human-understandable explanation for a specific forecast.
        This call is traced to LangSmith.
        """
        from src.configs.llm_prompts import PROMPTS, build_explainability_agent_user_prompt

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
            json_str = extract_json_from_llm_output(raw)
            data = json.loads(json_str)
            logger.info("Successfully parsed LLM response as JSON")
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}. Raw response: {raw}")
            # Fallback: create a basic explanation matching schema
            data = {
                "symbol": explanation_input.symbol,
                "horizon": explanation_input.horizon,
                "forecast_comment": f"The {explanation_input.model_family} model predicts a {explanation_input.forecast_return:.2%} move.",
                "top_feature_drivers": [],
                "regime_and_risk_comment": "Explanation unavailable due to parse error; using basic summary.",
                "limitations": ["LLM output could not be parsed; this explanation is generic."]
            }

        # Filter and validate
        valid_keys = LocalExplanation.__annotations__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        
        # Ensure required fields
        if "forecast_comment" not in filtered:
             filtered["forecast_comment"] = "Forecast comment missing."
        if "symbol" not in filtered:
             filtered["symbol"] = explanation_input.symbol
        if "horizon" not in filtered:
             filtered["horizon"] = explanation_input.horizon

        return LocalExplanation(**filtered)

    def explain_multiple_forecasts(self, forecasts: List[ExplainabilityInput]) -> List[LocalExplanation]:
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
                fallback = LocalExplanation(
                    symbol=forecast_input.symbol,
                    horizon=forecast_input.horizon,
                    forecast_comment=f"Unable to generate explanation due to error: {e}",
                    top_feature_drivers=[],
                    regime_and_risk_comment="Error occurred",
                    limitations=["Explanation generation failed"]
                )
                explanations.append(fallback)

        return explanations

    def format_explanation_as_html(self, explanation: LocalExplanation) -> str:
        """
        Format the explanation as an HTML snippet for UI display.
        """
        html = []

        html.append(f"<div class='forecast-explanation'>")
        html.append(f"<h4>{explanation.symbol} - {explanation.horizon} Day Forecast</h4>")

        html.append(f"<div class='prediction-summary'>")
        html.append(f"<strong>Forecast:</strong> {explanation.forecast_comment}")
        html.append(f"</div>")

        html.append(f"<div class='key-drivers'>")
        html.append(f"<strong>Top Feature Drivers:</strong>")
        html.append(f"<ul>")
        for driver in explanation.top_feature_drivers[:3]:  # Top 3 drivers
            name = driver.get('name', 'Unknown')
            comment = driver.get('comment', '')
            html.append(f"<li>{name}: {comment}</li>")
        html.append(f"</ul>")
        html.append(f"</div>")

        html.append(f"<div class='regime-influence'>")
        html.append(f"<strong>Regime & Risk:</strong> {explanation.regime_and_risk_comment}")
        html.append(f"</div>")

        if explanation.limitations:
            html.append(f"<div class='caveats'>")
            html.append(f"<strong>Limitations:</strong>")
            html.append(f"<ul>")
            for limit in explanation.limitations:
                html.append(f"<li>{limit}</li>")
            html.append(f"</ul>")
            html.append(f"</div>")

        html.append(f"</div>")

        return "\n".join(html)