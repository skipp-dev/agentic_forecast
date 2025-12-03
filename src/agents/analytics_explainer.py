import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import json
from langsmith import traceable

logger = logging.getLogger(__name__)

class AnalyticsAgent:
    """
    Agent for performing analytics on model forecasts.
    """

    def calculate_performance_summary(
        self, 
        forecasts: Dict[str, Dict[str, pd.DataFrame]], 
        raw_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculates the Mean Absolute Percentage Error (MAPE) for each model and symbol.

        Args:
            forecasts: A dictionary of forecasts, keyed by symbol and then model family.
            raw_data: A dictionary of raw data, keyed by symbol.

        Returns:
            A DataFrame summarizing the performance metrics.
        """
        performance_data = []

        for symbol, model_forecasts in forecasts.items():
            if symbol not in raw_data:
                continue

            actuals_df = raw_data[symbol].copy()
            if not isinstance(actuals_df.index, pd.DatetimeIndex):
                actuals_df.index = pd.to_datetime(actuals_df.index)

            for model_family, forecast_df in model_forecasts.items():
                
                # Handle dict (serialized DataFrame)
                if isinstance(forecast_df, dict):
                    forecast_df = pd.DataFrame.from_dict(forecast_df, orient='index')

                # Ensure forecast_df 'ds' column is datetime
                if 'ds' in forecast_df.columns:
                    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
                else:
                    # If ds is not a column, it might be the index or missing
                    # Try to recover from index if it looks like a date
                    pass # For now assume ds is a column as per contract

                # Merge forecast with actuals
                merged_df = pd.merge(
                    forecast_df,
                    actuals_df[['close']],
                    left_on='ds',
                    right_index=True,
                    how='inner'
                )

                if merged_df.empty:
                    mape = np.nan
                else:
                    # Calculate MAPE
                    y_true = merged_df['close']
                    y_pred = merged_df[model_family]
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

                performance_data.append({
                    'symbol': symbol,
                    'model_family': model_family,
                    'mape': mape
                })

        return pd.DataFrame(performance_data)

@dataclass
class AnalyticsInput:
    performance_summary: List[Dict[str, Any]]
    drift_events: List[Dict[str, Any]]
    risk_kpis: Optional[List[Dict[str, Any]]] = None
    top_n: int = 20

@dataclass
class AnalyticsRecommendation:
    summary_text: str
    actions: List[Dict[str, Any]]
    notes_for_humans: str

class LLMAnalyticsExplainerAgent:
    """
    Agent that uses OpenAILLMClient to explain performance metrics and drift.
    """
    def __init__(self, settings=None):
        # Use the new SmithLLMClient via factory
        from src.llm.llm_factory import create_llm_for_role
        self.llm = create_llm_for_role("analytics_explainer")
        self.settings = settings or {}

    @traceable(
        name="analytics_explainer_explain_metrics",
        tags=["analytics", "llm", "explainer"],
        metadata={"role": "analytics_explainer"}
    )
    def explain_metrics(self, metrics_payload: dict) -> dict:
        """
        Explain metrics using the structured prompt and JSON output schema.
        This call is traced to LangSmith.
        """
        from src.configs.llm_prompts import PROMPTS, build_analytics_summary_user_prompt

        system_prompt = PROMPTS["analytics_explainer"]
        user_prompt = build_analytics_summary_user_prompt(metrics_payload)

        logger.info("Calling LLM for analytics explanation (LangSmith tracing enabled)")

        raw = self.llm.complete(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.2,
            max_tokens=1200,
        )

        logger.info(f"Raw LLM response (first 500 chars): {raw[:500]}")

        try:
            data = json.loads(raw)
            logger.info("Successfully parsed LLM response as JSON")
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}. Raw response: {raw}")
            # Fallback: wrap the raw text if model messed up
            data = {
                "global_summary": raw,
                "metric_explanations": {},
                "regime_insights": [],
                "symbol_outliers": [],
                "feature_insights": {"overall_top_features": [], "shock_regime_top_features": []},
                "recommendations": [],
            }

        return data

    def analyze(self, analytics_input: AnalyticsInput) -> AnalyticsRecommendation:
        """
        Legacy method for backward compatibility.
        Converts old AnalyticsInput format to new metrics_payload format.
        """
        # Convert old format to new structured payload
        metrics_payload = self._convert_to_metrics_payload(analytics_input)

        # Use new method
        explanation_data = self.explain_metrics(metrics_payload)

        # Convert back to old AnalyticsRecommendation format for compatibility
        return AnalyticsRecommendation(
            summary_text=explanation_data.get("global_summary", "Analysis completed"),
            actions=explanation_data.get("recommendations", []),
            notes_for_humans=json.dumps(explanation_data, indent=2)
        )

    def _convert_to_metrics_payload(self, analytics_input: AnalyticsInput) -> dict:
        """
        Convert AnalyticsInput to the structured metrics_payload format.
        This is a bridge for backward compatibility.
        """
        # Extract basic metrics from performance_summary
        global_metrics: Dict[str, Any] = {}
        per_symbol_metrics: List[Dict[str, Any]] = []
        regime_metrics: Dict[str, Any] = {"peer_shock_flag": {}}

        if analytics_input.performance_summary:
            # Calculate global averages
            mapes = [row.get("mape", 0) for row in analytics_input.performance_summary if "mape" in row]
            maes = [row.get("mae", 0) for row in analytics_input.performance_summary if "mae" in row]
            accuracies = [row.get("directional_accuracy", 0) for row in analytics_input.performance_summary if "directional_accuracy" in row]

            if mapes:
                global_metrics["mape"] = {"mean": sum(mapes) / len(mapes), "trend": "unknown"}
            if maes:
                global_metrics["mae"] = {"mean": sum(maes) / len(maes), "trend": "unknown"}
            if accuracies:
                global_metrics["directional_accuracy"] = {"mean": sum(accuracies) / len(accuracies), "trend": "unknown"}

            # Convert to per-symbol format
            for row in analytics_input.performance_summary[:analytics_input.top_n]:
                per_symbol_metrics.append({
                    "symbol": row.get("symbol", "UNKNOWN"),
                    "target_horizon": row.get("horizon", 1),
                    "mae": row.get("mae", 0),
                    "mape": row.get("mape", 0),
                    "directional_accuracy": row.get("directional_accuracy", 0)
                })

        # Basic payload structure
        return {
            "run_metadata": {
                "run_type": "LEGACY_CONVERSION",
                "date": "unknown",
                "universe_size": len(per_symbol_metrics),
                "cross_asset_v2_enabled": False
            },
            "metrics_global": global_metrics,
            "per_symbol_metrics": per_symbol_metrics,
            "regime_metrics": regime_metrics,
            "feature_importance": {
                "overall": [],
                "shock_regime": []
            },
            "guardrail_summary": {
                "status": "unknown",
                "issues": []
            }
        }

