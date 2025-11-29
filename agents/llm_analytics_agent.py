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
            "analytics_summary",
            performance_summary_json=str(perf_rows)
        )
        
        # Call LLM
        logger.info("Calling LLM for analytics explanation")
        response = self.llm.generate(prompt, temperature=0.1)
        
        # For now, return a simple recommendation since LLM returns text
        return AnalyticsRecommendation(
            summary_text=response,
            actions=[],
            notes_for_humans="LLM analysis completed"
        )
