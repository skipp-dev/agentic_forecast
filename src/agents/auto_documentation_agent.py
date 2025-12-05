from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class MetricsSnapshot:
    model_performance: Dict[str, Any]  # per horizon/family metrics
    guardrail_counts: Dict[str, int]
    news_health: Dict[str, Any]
    llm_usage: Dict[str, Any]
    risk_events: List[Any] = None
    run_health: Dict[str, Any] = None

@dataclass
class AutoDocReport:
    markdown_content: str
    json_summary: Dict[str, Any]

class AutoDocumentationAgent:
    """
    Agent that generates human-readable reports from run metrics and configuration.
    """
    
    def __init__(self, llm_client=None, config: Dict[str, Any] = None):
        self.llm_client = llm_client
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
    def generate_run_report(
        self,
        run_context: Dict[str, Any],
        metrics_snapshot: MetricsSnapshot,
        config_snapshot: Dict[str, Any],
    ) -> AutoDocReport:
        """
        Generate a run report based on the provided context and metrics.
        """
        if not self.enabled:
            return AutoDocReport(markdown_content="", json_summary={})
            
        logger.info("Generating auto-documentation report...")
        
        # Prepare context for LLM
        context_str = self._prepare_context(run_context, metrics_snapshot, config_snapshot)
        
        # Generate report using LLM
        if self.llm_client:
            try:
                report_content = self._generate_with_llm(context_str)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                report_content = self._generate_fallback_report(run_context, metrics_snapshot)
        else:
            report_content = self._generate_fallback_report(run_context, metrics_snapshot)
            
        # Create JSON summary
        json_summary = {
            "run_id": run_context.get("run_id"),
            "date": datetime.now().isoformat(),
            "metrics": metrics_snapshot.model_performance,
            "guardrails": metrics_snapshot.guardrail_counts
        }
        
        return AutoDocReport(markdown_content=report_content, json_summary=json_summary)
        
    def _prepare_context(
        self,
        run_context: Dict[str, Any],
        metrics: MetricsSnapshot,
        config: Dict[str, Any]
    ) -> str:
        """Format the context for the LLM prompt."""
        # Convert risk events to serializable format if they are objects
        serializable_risk_events = []
        if metrics.risk_events:
            for e in metrics.risk_events:
                if hasattr(e, '__dict__'):
                    serializable_risk_events.append(e.__dict__)
                else:
                    serializable_risk_events.append(str(e))

        return json.dumps({
            "run_info": {
                "type": run_context.get("run_type"),
                "id": run_context.get("run_id"),
                "date": str(datetime.now()),
                "health": metrics.run_health
            },
            "metrics": {
                "performance": metrics.model_performance,
                "guardrails": metrics.guardrail_counts,
                "news": metrics.news_health,
                "risk_events": serializable_risk_events
            },
            "config_highlights": {
                "news_enabled": config.get("news", {}).get("enabled", False),
                "models": config.get("models", {}).get("primary", [])
            }
        }, indent=2)
        
    def _generate_with_llm(self, context_str: str) -> str:
        """Call LLM to generate the markdown report."""
        system_prompt = """
        You are the AutoDocumentation Agent for an advanced agentic forecasting system.
        Your goal is to write a concise, professional Markdown report summarizing the latest run.
        
        Include sections:
        1. **Run Metadata**: Type, ID, Date.
        2. **Model Health**: Which models performed best? Any failures?
        3. **Data Health**: News coverage, shock detection, fundamentals.
        4. **Risk & Guardrails**: Notable guardrail activations.
        5. **Strategic Recommendations**: Suggested next steps (e.g. HPO, risk reduction).
        
        Be objective and data-driven. Use bullet points for readability.
        """
        
        user_prompt = f"Here is the run context and metrics:\n\n{context_str}\n\nPlease generate the run report."
        
        # Assuming llm_client has a chat completion method
        # This might need adjustment based on the actual LLMClient interface
        if hasattr(self.llm_client, "chat_completion"):
             response = self.llm_client.chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
             return response
        elif hasattr(self.llm_client, "generate"):
             # Fallback for simple interface
             return self.llm_client.generate(f"{system_prompt}\n\n{user_prompt}")
        else:
             raise NotImplementedError("LLM client interface not supported")

    def _generate_fallback_report(self, run_context: Dict[str, Any], metrics: MetricsSnapshot) -> str:
        """Generate a basic template-based report if LLM fails or is missing."""
        health_status = "UNKNOWN"
        if metrics.run_health:
            health_status = metrics.run_health.get('status', 'UNKNOWN')
            
        return f"""
# Run Report: {run_context.get('run_id', 'Unknown')}

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Type**: {run_context.get('run_type', 'Unknown')}
**Status**: {health_status}

## Model Performance
- Performance metrics available in JSON summary.

## Risk & Guardrails
- Run Health: {health_status}
- Active flags: {json.dumps(metrics.guardrail_counts, indent=2)}
- Risk Events: {len(metrics.risk_events) if metrics.risk_events else 0}

## News Health
- News metrics: {json.dumps(metrics.news_health, indent=2)}

*(Generated via fallback template)*
"""
