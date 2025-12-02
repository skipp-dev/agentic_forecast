import logging
import json
from typing import Dict, Any
from langsmith import traceable
from src.llm.llm_factory import create_llm_for_role
from src.configs.llm_prompts import PROMPTS, build_research_agent_user_prompt, extract_json_from_response

logger = logging.getLogger(__name__)

class OpenAIResearchAgent:
    """
    Agent responsible for conducting deep research on market trends and anomalies.
    """
    def __init__(self):
        self.llm = create_llm_for_role("research_agent")
        
    @traceable(name="research_agent_run", tags=["research", "llm"])
    def run(self, metrics_and_regimes: Dict[str, Any], external_context: str = "") -> Dict[str, Any]:
        """
        Analyze metrics and context to generate research insights.
        """
        system_prompt = PROMPTS["research_agent"]
        user_prompt = build_research_agent_user_prompt(metrics_and_regimes, external_context)
        
        logger.info("Calling LLM for research analysis")
        
        raw_response = self.llm.complete(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        try:
            data = extract_json_from_response(raw_response)
            return data
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {"error": str(e), "raw_response": raw_response}
