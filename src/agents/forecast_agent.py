#!/usr/bin/env python3
"""
Forecast Agent

Risk-aware interpreter that translates raw model forecasts into actionable,
human-readable insights with uncertainty quantification and scenario analysis.
"""

import os
import sys
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple, Literal
import logging
from datetime import datetime
import pandas as pd

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.policy_optimizer import PolicyOptimizer
from src.schemas import ForecastResult, HorizonForecast
from src.services.model_registry_service import ModelRegistryService

logger = logging.getLogger(__name__)

# Guardrail flag categories
CRITICAL_GUARDRAILS = {
    "data_drift_suspected",
    "missing_feature_values",
    "pipeline_failure_recently",
}

NON_CRITICAL_GUARDRAILS = {
    "high_error_recently",
    "shock_regime_active",
    "news_shock_active",
}


class ForecastAgent:
    """
    Forecast Agent that interprets raw model outputs and produces risk-aware JSON summaries.

    This agent acts as the "cautious translator" between technical forecasts and
    human/trading decisions, providing structured insights with uncertainty quantification.
    """

    def __init__(self, llm_client=None, config_path: str = "src/configs/model_families.yaml", model_registry: Optional[ModelRegistryService] = None):
        """
        Initialize the Forecast Agent.

        Args:
            llm_client: LLM client for generating narrative comments (optional)
            config_path: Path to model family policy configuration
            model_registry: ModelRegistryService instance
        """
        self.llm_client = llm_client
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.policy_optimizer = PolicyOptimizer(default_policy=self.config)
        self.model_registry = model_registry or ModelRegistryService()
        logger.info("Forecast Agent initialized")

    def update_policy_from_performance(self, performance_summary: List[Dict[str, Any]]) -> None:
        """
        Update model selection policy based on recent performance metrics.
        
        Args:
            performance_summary: List of dicts with keys ['symbol', 'model_family', 'mape']
        """
        if not performance_summary:
            return
            
        try:
            df = pd.DataFrame(performance_summary)
            new_policy = self.policy_optimizer.optimize_policy(df)
            
            # Update in-memory config
            self.config = new_policy
            logger.info("Updated model family policy based on performance metrics")
            
            # Optional: Persist to disk? 
            # For now, we keep it in memory for the session. 
            # If we wanted to persist:
            # with open(self.config_path, 'w') as f:
            #     yaml.dump(new_policy, f)
                
        except Exception as e:
            logger.error(f"Failed to update policy from performance: {e}")

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load model family policy configuration."""
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Config file not found at {path}, using defaults")
                return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _apply_model_policy(
        self,
        symbol: str,
        horizon: int,
        available_forecasts: Dict[str, float],
        error_metrics: Dict[str, Any]
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Apply model family policy to select champion or compute ensemble.
        
        Returns:
            Tuple of (predicted_return, selected_family, metadata)
        """
        # Determine policy bucket (default for now, could be extended for low_data etc.)
        policy = self.config.get('default_policy', {})
        
        # Determine horizon category
        if horizon <= 2:
            horizon_policy = policy.get('short_horizon', {})
        elif horizon <= 10:
            horizon_policy = policy.get('medium_horizon', {})
        else:
            horizon_policy = policy.get('long_horizon', {})
            
        primary_families = horizon_policy.get('primary', [])
        secondary_families = horizon_policy.get('secondary', [])
        baseline_families = horizon_policy.get('baseline', [])
        
        # Find available primary models
        available_primary = [f for f in primary_families if f in available_forecasts]
        available_secondary = [f for f in secondary_families if f in available_forecasts]
        
        champion_family = "Unknown"
        final_prediction = 0.0
        metadata: Dict[str, Any] = {}

        # 1. Check ModelRegistry for best model (Priority)
        try:
            best_model_meta = self.model_registry.get_best_model(symbol, metric='mae', mode='min')
            if best_model_meta:
                best_family = best_model_meta['model_type']
                if best_family in available_forecasts:
                    champion_family = best_family
                    final_prediction = available_forecasts[champion_family]
                    metadata['source'] = 'ModelRegistry'
                    metadata['model_id'] = best_model_meta.get('model_id')
                    metadata['best_metric'] = best_model_meta.get('metrics', {}).get('mae')
                    metadata['selected_family'] = champion_family
                    return final_prediction, champion_family, metadata
        except Exception as e:
            logger.warning(f"Failed to query ModelRegistry for {symbol}: {e}")
        
        # Champion selection logic (Fallback to Policy)
        if available_primary:
            # Pick best primary based on recent metrics if available, else first
            # For now, simple logic: pick first available primary
            champion_family = available_primary[0]
            final_prediction = available_forecasts[champion_family]
        elif available_secondary:
            champion_family = available_secondary[0]
            final_prediction = available_forecasts[champion_family]
        elif baseline_families:
             # Fallback to baseline
             available_baseline = [f for f in baseline_families if f in available_forecasts]
             if available_baseline:
                 champion_family = available_baseline[0]
                 final_prediction = available_forecasts[champion_family]
             else:
                 # Last resort: any available model
                 if available_forecasts:
                     champion_family = list(available_forecasts.keys())[0]
                     final_prediction = available_forecasts[champion_family]
        else:
             if available_forecasts:
                 champion_family = list(available_forecasts.keys())[0]
                 final_prediction = available_forecasts[champion_family]

        # Ensemble logic
        if self.config.get('enable_ensembling', False) and available_primary and available_secondary:
            weights = horizon_policy.get('ensemble_weights', {'primary': 1.0, 'secondary': 0.0})
            w_p = weights.get('primary', 1.0)
            w_s = weights.get('secondary', 0.0)
            
            # Simple ensemble: average of available primaries * w_p + average of available secondaries * w_s
            # Normalize weights if needed
            total_w = w_p + w_s
            if total_w > 0:
                pred_p = sum(available_forecasts[f] for f in available_primary) / len(available_primary)
                pred_s = sum(available_forecasts[f] for f in available_secondary) / len(available_secondary)
                
                ensemble_pred = (pred_p * w_p + pred_s * w_s) / total_w
                final_prediction = ensemble_pred
                champion_family = f"Ensemble({champion_family}+{available_secondary[0]})"
                metadata['ensemble_components'] = available_primary + available_secondary

        metadata['selected_family'] = champion_family
        return final_prediction, champion_family, metadata

    def _compare_with_baselines(
        self,
        symbol: str,
        horizon: int,
        forecasts: Dict[str, float],
        champion_family: str,
        champion_value: float
    ) -> Dict[str, Any]:
        """
        Compare champion forecast with baselines (AutoDLinear, BaselineLinear).
        """
        comparison: Dict[str, Any] = {}
        baselines = ["AutoDLinear", "BaselineLinear"]
        
        for baseline in baselines:
            if baseline in forecasts:
                baseline_val = forecasts[baseline]
                gap = champion_value - baseline_val
                comparison[f"gap_vs_{baseline}"] = gap
                
                # Check for significant deviation (e.g. > 5% difference in return prediction)
                if abs(gap) > 0.05:
                    comparison[f"warning_vs_{baseline}"] = f"Significant deviation from {baseline}: {gap:.2%}"
                    
                # Specific check: Deep model vs AutoDLinear
                if baseline == "AutoDLinear" and champion_family not in baselines:
                    # If deep model predicts huge move but DLinear is flat, flag it
                    if abs(champion_value) > 0.03 and abs(baseline_val) < 0.01:
                         comparison["deep_vs_linear_divergence"] = "Deep model predicts large move not seen by linear baseline"
                    
                    # Direction mismatch check
                    # Only flag if both predict a move > 1% in opposite directions
                    if (champion_value * baseline_val < 0) and (abs(champion_value) > 0.01) and (abs(baseline_val) > 0.01):
                        comparison["deep_vs_linear_direction_mismatch"] = f"Direction mismatch: Deep {champion_value:.1%} vs Linear {baseline_val:.1%}"

                # Specific check: Risk Baseline (BaselineLinear)
                if baseline == "BaselineLinear":
                    # If champion is way off from simple regression, it's a risk flag
                    if abs(gap) > 0.10: # 10% divergence is huge
                        comparison["risk_baseline_divergence"] = "CRITICAL: Champion model diverges >10% from risk baseline"

        return comparison

    def interpret_forecasts(
        self,
        symbol: str,
        forecasts: List[Dict[str, Any]],
        error_metrics: Dict[str, Any],
        regime_and_guardrail_info: Optional[Dict[str, Any]] = None
    ) -> ForecastResult:
        """
        Interpret forecasts and produce risk-aware JSON summary.

        Args:
            symbol: Stock symbol
            forecasts: List of forecast objects with horizon and predicted_return. 
            error_metrics: Historical error metrics (MAE, MAPE, SMAPE, SWASE, directional_accuracy)
            regime_and_guardrail_info: Current regime flags and guardrail statuses

        Returns:
            ForecastResult object with interpreted forecasts, confidence levels, and scenario notes
        """
        logger.info(f"Interpreting forecasts for {symbol}")

        # Extract guardrail flags
        guardrail_flags = []
        if regime_and_guardrail_info:
            guardrail_flags = regime_and_guardrail_info.get("guardrail_flags", [])

        # Group forecasts by horizon to apply policy
        forecasts_by_horizon: Dict[int, Dict[str, float]] = {}
        for f in forecasts:
            h = f.get("horizon", 1)
            if h not in forecasts_by_horizon:
                forecasts_by_horizon[h] = {}
            
            # If input has model_family, use it. If not, assume it's the only one or "Unknown"
            family = f.get("model_family", "Unknown")
            forecasts_by_horizon[h][family] = f.get("predicted_return", 0.0)

        # Build horizon forecasts with confidence
        horizon_forecasts_objs = []
        
        # Iterate over horizons found in input
        for horizon, family_forecasts in forecasts_by_horizon.items():
            
            # Apply policy to select champion/ensemble
            predicted_return, selected_family, metadata = self._apply_model_policy(
                symbol, horizon, family_forecasts, error_metrics
            )
            
            # Compare with baselines
            baseline_comparison = self._compare_with_baselines(
                symbol, horizon, family_forecasts, selected_family, predicted_return
            )
            metadata.update(baseline_comparison)
            
            # Compute confidence for this horizon
            confidence, confidence_reasons = self._compute_confidence_level(
                error_metrics, guardrail_flags
            )

            # Generate comment
            comment = self._generate_forecast_comment(
                horizon, predicted_return, confidence, regime_and_guardrail_info
            )

            horizon_forecasts_objs.append(HorizonForecast(
                horizon=horizon,
                predicted_return=predicted_return,
                confidence=confidence,
                comment=comment,
                selected_model_family=selected_family,
                metadata=metadata
            ))

        # Build risk assessment
        trust_score, trust_reasons = self._compute_trust_score(error_metrics, guardrail_flags)
        
        market_regimes = regime_and_guardrail_info.get("market_regimes", {}) if regime_and_guardrail_info else {}
        
        risk_assessment = {
            "model_confidence_comment": self._build_model_confidence_comment(
                error_metrics, confidence_reasons
            ),
            "regime_comment": self._build_regime_comment(guardrail_flags, market_regimes),
            "guardrail_flags": guardrail_flags,
            "trust_score": trust_score,
            "trust_reasons": trust_reasons
        }

        # Generate scenario notes
        scenario_notes_list = self._generate_scenario_notes(
            symbol, [h.dict() for h in horizon_forecasts_objs], risk_assessment, regime_and_guardrail_info
        )
        scenario_notes = "\n".join(scenario_notes_list)

        result = ForecastResult(
            symbol=symbol,
            valid_until=datetime.now().timestamp() + 86400, # 24h TTL default
            horizon_forecasts=horizon_forecasts_objs,
            risk_assessment=risk_assessment,
            narrative_summary=scenario_notes
        )

        # Save to Prometheus metrics file
        try:
            from services.metrics_exporter import save_forecast_agent_output
            save_forecast_agent_output(result.dict())
        except Exception as e:
            logger.warning(f"Failed to save forecast agent output for Prometheus: {e}")

        # Log the full JSON output for audit/backtesting
        self._log_forecast_output_for_audit(result.dict())

        return result

    def _compute_trust_score(
        self,
        metrics: Dict[str, Any],
        guardrail_flags: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Compute a deterministic trust score (0.0 - 1.0) based on metrics and guardrails.
        
        Score components:
        - Base score: 1.0
        - Critical guardrails: -1.0 (forces 0.0)
        - Non-critical guardrails: -0.2 each
        - Poor directional accuracy (< 0.55): -0.3
        - High SMAPE (> 0.20): -0.2
        - Missing metrics: -0.5
        
        Returns:
            Tuple of (trust_score, reasons)
        """
        score = 1.0
        reasons = []
        flags = set(guardrail_flags or [])
        
        # 1. Critical Guardrails (Immediate zero)
        if flags & CRITICAL_GUARDRAILS:
            active = sorted(flags & CRITICAL_GUARDRAILS)
            reasons.append(f"Critical guardrails active: {', '.join(active)} (-1.0)")
            return 0.0, reasons
            
        # 2. Non-Critical Guardrails
        non_critical_active = flags & NON_CRITICAL_GUARDRAILS
        for flag in non_critical_active:
            penalty = 0.2
            score -= penalty
            reasons.append(f"Guardrail '{flag}' active (-{penalty})")
            
        # 3. Metric Sanity & Performance
        da = metrics.get("directional_accuracy")
        smape = metrics.get("smape")
        
        if da is None or smape is None:
            penalty = 0.5
            score -= penalty
            reasons.append(f"Missing key metrics (DA/SMAPE) (-{penalty})")
        else:
            # Directional Accuracy penalties
            if da < 0.50:
                penalty = 0.4
                score -= penalty
                reasons.append(f"Very poor directional accuracy ({da:.2f} < 0.50) (-{penalty})")
            elif da < 0.55:
                penalty = 0.2
                score -= penalty
                reasons.append(f"Poor directional accuracy ({da:.2f} < 0.55) (-{penalty})")
                
            # SMAPE penalties
            if smape > 0.30:
                penalty = 0.3
                score -= penalty
                reasons.append(f"Very high SMAPE ({smape:.2f} > 0.30) (-{penalty})")
            elif smape > 0.20:
                penalty = 0.1
                score -= penalty
                reasons.append(f"High SMAPE ({smape:.2f} > 0.20) (-{penalty})")

        # 4. Regime Adjustments (if not already covered by flags)
        # If we have specific regime info in metrics that isn't a flag, use it here.
        # Currently relying on flags like 'shock_regime_active'.
        
        # Clamp score to [0.0, 1.0]
        final_score = max(0.0, min(1.0, score))
        
        return final_score, reasons

    def _compute_confidence_level(
        self,
        metrics: Dict[str, Any],
        guardrail_flags: List[str],
    ) -> Tuple[str, List[str]]:
        """
        Map error metrics + guardrail flags to a discrete confidence level.

        Returns:
            Tuple of (confidence_level, reasons_list)
        """
        reasons = []
        flags = set(guardrail_flags or [])

        # 1) Critical guardrails immediately cap to LOW
        if flags & CRITICAL_GUARDRAILS:
            reasons.append(
                f"Critical guardrails active: {', '.join(sorted(flags & CRITICAL_GUARDRAILS))}."
            )
            reasons.append("Confidence forced to LOW due to potential data/pipeline issues.")
            return "low", reasons

        # 2) Metric-based initial confidence
        directional_acc = metrics.get("directional_accuracy")
        smape = metrics.get("smape")
        mae_vs_baseline = metrics.get("mae_vs_baseline")

        confidence = "low"

        # High confidence: good directional accuracy, low smape, not worse than baseline
        if directional_acc is not None and smape is not None:
            if (
                directional_acc >= 0.60
                and smape <= 0.15
                and (mae_vs_baseline is None or mae_vs_baseline <= 1.0)
            ):
                confidence = "high"
                reasons.append(
                    f"Directional accuracy {directional_acc:.2f} and SMAPE {smape:.2f} indicate strong past performance."
                )
                if mae_vs_baseline is not None:
                    reasons.append(
                        f"MAE vs baseline {mae_vs_baseline:.2f} (≤ 1.0 is good) supports using this model."
                    )
            # Medium confidence: slightly weaker but still acceptable
            elif directional_acc >= 0.55 and smape <= 0.20:
                confidence = "medium"
                reasons.append(
                    f"Directional accuracy {directional_acc:.2f} and SMAPE {smape:.2f} indicate moderate reliability."
                )
            else:
                confidence = "low"
                reasons.append(
                    f"Directional accuracy {directional_acc:.2f} and/or SMAPE {smape:.2f} suggest limited reliability."
                )
        else:
            confidence = "low"
            reasons.append("Missing key metrics (directional_accuracy and/or SMAPE). Set to LOW.")

        # 3) Adjust for non-critical guardrails
        non_critical_active = flags & NON_CRITICAL_GUARDRAILS
        if non_critical_active:
            reasons.append(
                f"Non-critical guardrails active: {', '.join(sorted(non_critical_active))}."
            )
            if confidence == "high":
                confidence = "medium"
                reasons.append("Downgraded from HIGH to MEDIUM due to elevated risk regime.")
            elif confidence == "medium":
                confidence = "low"
                reasons.append("Downgraded from MEDIUM to LOW due to elevated risk regime.")
            else:
                reasons.append("Confidence already LOW; no further downgrade.")

        return confidence, reasons

    def _generate_forecast_comment(
        self,
        horizon: int,
        predicted_return: float,
        confidence: str,
        regime_info: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate a short, natural language comment for a forecast.
        """
        # Magnitude description
        if abs(predicted_return) < 0.005:  # < 0.5%
            magnitude = "minimal"
        elif abs(predicted_return) < 0.015:  # < 1.5%
            magnitude = "modest"
        elif abs(predicted_return) < 0.03:  # < 3%
            magnitude = "moderate"
        else:
            magnitude = "significant"

        direction = "positive" if predicted_return > 0 else "negative"

        # Horizon context
        if horizon == 1:
            horizon_desc = "tomorrow"
        elif horizon <= 5:
            horizon_desc = f"next {horizon} days"
        else:
            horizon_desc = f"next {horizon} days"

        # Base comment
        comment = f"Model expects a {magnitude} {direction} move ({predicted_return:.2%}) over the {horizon_desc} with {confidence} confidence."

        # Add regime context if relevant
        if regime_info:
            if regime_info.get("guardrail_flags"):
                flags = set(regime_info["guardrail_flags"])
                if "shock_regime_active" in flags:
                    comment += " Shock regime may amplify volatility."
                elif "high_error_recently" in flags:
                    comment += " Recent forecast errors suggest caution."
            
            # Add market regime context
            market_regimes = regime_info.get("market_regimes", {})
            if market_regimes.get("rate_regime") == "high_rates":
                comment += " High interest rate environment detected."
            if market_regimes.get("oil_regime") == "spike":
                comment += " Oil price spike may impact costs."

        return comment

    def _build_model_confidence_comment(
        self,
        metrics: Dict[str, Any],
        confidence_reasons: List[str]
    ) -> str:
        """
        Build the model confidence comment for risk assessment.
        """
        if confidence_reasons:
            return " ".join(confidence_reasons)
        else:
            return "Model confidence assessment not available."

    def _build_regime_comment(self, guardrail_flags: List[str], market_regimes: Dict[str, str] = None) -> str:
        """
        Build regime comment based on active guardrail flags and market regimes.
        """
        comments = []
        flags = set(guardrail_flags or [])

        if "shock_regime_active" in flags:
            comments.append("Shock regime is active; volatility and gap risk are elevated.")
        elif "news_shock_active" in flags:
            comments.append("Recent strong news flow may cause sudden moves beyond the model's typical error profile.")
        elif "high_error_recently" in flags:
            comments.append("Recent forecast errors were elevated; recent behavior may be less predictable than usual.")
            
        if market_regimes:
            if market_regimes.get("rate_regime") == "high_rates":
                comments.append("Market is in a High Rates regime.")
            if market_regimes.get("oil_regime") == "spike":
                comments.append("Oil prices are spiking.")
            if market_regimes.get("gold_regime") == "rally":
                comments.append("Gold rally suggests risk-off sentiment.")
                
        if not comments:
            return "No active guardrail flags; regime appears normal from the model's perspective."
            
        return " ".join(comments)

    def _generate_scenario_notes(
        self,
        symbol: str,
        horizon_forecasts: List[Dict[str, Any]],
        risk_assessment: Dict[str, Any],
        regime_info: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate scenario-style notes for human consideration.
        """
        notes = []

        # Check for shock regime
        guardrail_flags = set(risk_assessment.get("guardrail_flags", []))
        has_shock_regime = "shock_regime_active" in guardrail_flags
        has_high_error = "high_error_recently" in guardrail_flags

        # Basic positioning note
        notes.append(
            "If someone wanted exposure to this symbol, position sizing should reflect the current confidence levels and guardrail flags."
        )

        # Shock regime specific notes
        if has_shock_regime:
            notes.append(
                "Shock regime is active; sudden adverse moves are more likely, so wider stops or reduced leverage may be appropriate if seeking exposure."
            )

        # High error notes
        if has_high_error:
            notes.append(
                "Recent forecast errors have been elevated, suggesting the model may be in a less predictable regime than usual."
            )

        # Forecast consistency notes
        if len(horizon_forecasts) > 1:
            directions = [f.get("predicted_return", 0) > 0 for f in horizon_forecasts]
            if all(directions) or not any(directions):
                consistency = "consistent"
            else:
                consistency = "mixed"

            if consistency == "consistent":
                direction_word = "positive" if directions[0] else "negative"
                notes.append(
                    f"Forecasts across horizons are {consistency} in their {direction_word} outlook."
                )
            else:
                notes.append(
                    "Forecasts across horizons show mixed signals; short-term and longer-term outlooks differ."
                )

        # Large move caveat
        large_moves = [f for f in horizon_forecasts if abs(f.get("predicted_return", 0)) > 0.03]
        if large_moves:
            notes.append(
                "Some forecasts suggest relatively large moves; these should be treated with extra caution given model uncertainty."
            )

        # Low confidence warning
        low_confidence_forecasts = [f for f in horizon_forecasts if f.get("confidence") == "low"]
        if low_confidence_forecasts:
            horizons = [str(f["horizon"]) for f in low_confidence_forecasts]
            notes.append(
                f"Forecasts for horizon(s) {', '.join(horizons)} have low confidence and should be treated as highly uncertain."
            )

        return notes

    def _log_forecast_output_for_audit(self, output_model: Dict[str, Any]) -> None:
        """
        Log the full ForecastAgentOutput JSON for backtesting/audit purposes.
        
        This creates structured logs that can be scraped by Loki or similar
        log aggregation systems for later analysis.
        """
        import json
        from datetime import datetime
        
        payload = {
            "ts": datetime.utcnow().isoformat(),
            "agent": "forecast_agent",
            "data": output_model
        }
        
        # Log as structured JSON
        logger.info("forecast_agent_output %s", json.dumps(payload, default=str))

    def interpret_forecast_bundle(
        self,
        forecast_bundle: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convenience method to interpret a complete forecast bundle.

        Expected bundle structure:
        {
            "symbol": "AAPL",
            "forecasts": [...],  # List of forecast objects
            "error_metrics": {...},  # Error metrics dict
            "regime_and_guardrail_info": {...}  # Optional regime info
        }
        """
        return self.interpret_forecasts(
            symbol=forecast_bundle["symbol"],
            forecasts=forecast_bundle["forecasts"],
            error_metrics=forecast_bundle["error_metrics"],
            regime_and_guardrail_info=forecast_bundle.get("regime_and_guardrail_info")
        )


# Convenience functions for integration
def create_forecast_agent(llm_client=None):
    """Create and configure forecast agent."""
    return ForecastAgent(llm_client=llm_client)


def interpret_symbol_forecasts(
    symbol: str,
    forecasts: List[Dict[str, Any]],
    error_metrics: Dict[str, Any],
    regime_info: Optional[Dict[str, Any]] = None,
    llm_client=None
) -> Dict[str, Any]:
    """
    Convenience function to interpret forecasts for a symbol.
    """
    agent = create_forecast_agent(llm_client=llm_client)
    return agent.interpret_forecasts(symbol, forecasts, error_metrics, regime_info)


def interpret_forecast_bundle(
    forecast_bundle: Dict[str, Any],
    llm_client=None
) -> Dict[str, Any]:
    """
    Convenience function to interpret a complete forecast bundle.
    """
    agent = create_forecast_agent(llm_client=llm_client)
    return agent.interpret_forecast_bundle(forecast_bundle)