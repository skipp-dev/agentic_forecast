from typing import List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RawNewsItem:
    symbol: str
    timestamp: str
    headline: str
    body: str
    provider: str

@dataclass
class EnrichedNewsFeature:
    symbol: str
    timestamp: str
    headline: str
    categories: List[str]
    directional_impact: str
    impact_horizon: str
    volatility_impact: str
    confidence: float
    notes: str

class LLMNewsFeatureAgent:
    """
    Agent that uses an LLM to enrich news data with structured features.
    """
    def __init__(self, llm_client):
        self.llm = llm_client
        from src.prompts.llm_prompts import get_prompt
        self.get_prompt = get_prompt

    def enrich_item(self, item: RawNewsItem) -> EnrichedNewsFeature:
        """
        Enrich a single news item.
        """
        prompt = self.get_prompt(
            "news_enrichment",
            symbol=item.symbol,
            timestamp=item.timestamp,
            headline=item.headline,
            body=item.body[:800]
        )
        
        logger.info(f"Enriching news for {item.symbol}")
        response = self.llm.chat(prompt)
        data = response.json() if hasattr(response, 'json') else {}
        
        return EnrichedNewsFeature(
            symbol=item.symbol,
            timestamp=item.timestamp,
            headline=item.headline,
            categories=data.get("categories", []),
            directional_impact=data.get("directional_impact", "neutral"),
            impact_horizon=data.get("impact_horizon", "intraday"),
            volatility_impact=data.get("volatility_impact", "low"),
            confidence=float(data.get("confidence", 0.0)),
            notes=data.get("notes", "")
        )
