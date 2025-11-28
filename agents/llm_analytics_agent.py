from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

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
    Agent that uses an LLM to explain performance metrics and drift.
    """
    def __init__(self, llm_client):
        self.llm = llm_client
        from src.prompts.llm_prompts import get_prompt
        self.get_prompt = get_prompt

    def analyze(self, analytics_input: AnalyticsInput) -> AnalyticsRecommendation:
        """
        Analyze performance and drift data to generate recommendations.
        """
        # Prepare data for prompt
        perf_rows = analytics_input.performance_summary[:analytics_input.top_n]
        drift_rows = analytics_input.drift_events[:analytics_input.top_n]
        
        # Build prompt
        prompt = self.get_prompt(
            "drift_analysis",
            drift_events_json=str(drift_rows) # In real usage, use json.dumps
        )
        
        # Call LLM
        logger.info("Calling LLM for analytics explanation")
        response = self.llm.chat(prompt)
        
        # Parse response (assuming LLM returns JSON-compatible structure)
        # This is a simplified placeholder
        data = response.json() if hasattr(response, 'json') else {}
        
        return AnalyticsRecommendation(
            summary_text=data.get("summary_text", "Analysis complete."),
            actions=data.get("actions", []),
            notes_for_humans=data.get("notes_for_humans", "")
        )
