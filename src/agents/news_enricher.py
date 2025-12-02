import logging
import json
from typing import List, Dict, Any
from langsmith import traceable
from src.llm.llm_factory import create_llm_for_role
from src.configs.llm_prompts import PROMPTS, build_news_enrichment_user_prompt, extract_json_from_response

logger = logging.getLogger(__name__)

class LLMNewsFeatureAgent:
    """
    Agent responsible for enriching forecast data with news sentiment and features.
    """
    def __init__(self):
        self.llm = create_llm_for_role("news_enricher")
    
    @traceable(name="news_enricher_run", tags=["news", "llm"])
    def run(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a list of news items and return structured features.
        """
        if not news_items:
            logger.info("No news items to process.")
            return {"per_item_annotations": [], "daily_aggregates": []}

        system_prompt = PROMPTS["news_enrichment"]
        user_prompt = build_news_enrichment_user_prompt(news_items)
        
        logger.info(f"Calling LLM to enrich {len(news_items)} news items")
        
        raw_response = self.llm.complete(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.2,
            max_tokens=2000
        )
        
        try:
            data = extract_json_from_response(raw_response)
            return data
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {"per_item_annotations": [], "daily_aggregates": [], "error": str(e), "raw_response": raw_response}
