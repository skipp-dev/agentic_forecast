from typing import List, Dict, Any
from dataclasses import dataclass
import logging
import pandas as pd

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
    sentiment_score: float = 0.0

class LLMNewsFeatureAgent:
    """
    Agent that uses an LLM to enrich news data with structured features.
    """
    def __init__(self, llm_client):
        self.llm = llm_client
        try:
            from src.prompts.llm_prompts import get_prompt
            self.get_prompt = get_prompt
        except ImportError:
            self.get_prompt = lambda x, **kwargs: f"Analyze news for {kwargs.get('symbol')}: {kwargs.get('headline')}"

    def enrich_item(self, item: RawNewsItem) -> EnrichedNewsFeature:
        """
        Enrich a single news item.
        """
        prompt = self.get_prompt(
            "news_enrichment",
            symbol=item.symbol,
            timestamp=item.timestamp,
            headline=item.headline,
            body=item.body[:800] if item.body else ""
        )
        
        # logger.info(f"Enriching news for {item.symbol}")
        if hasattr(self.llm, 'chat'):
            try:
                response = self.llm.chat(prompt)
                data = response.json() if hasattr(response, 'json') else {}
            except Exception as e:
                logger.warning(f"LLM enrichment failed: {e}")
                data = {}
        else:
            data = {}
        
        return EnrichedNewsFeature(
            symbol=item.symbol,
            timestamp=item.timestamp,
            headline=item.headline,
            categories=data.get("categories", []),
            directional_impact=data.get("directional_impact", "neutral"),
            impact_horizon=data.get("impact_horizon", "intraday"),
            volatility_impact=data.get("volatility_impact", "low"),
            confidence=float(data.get("confidence", 0.0)),
            notes=data.get("notes", ""),
            sentiment_score=float(data.get("sentiment_score", 0.0))
        )

    def enrich_batch(self, items: List[Dict]) -> List[EnrichedNewsFeature]:
        """Enrich a batch of news items."""
        enriched = []
        for item_dict in items:
            # Handle both dict and object input
            if isinstance(item_dict, dict):
                # Ensure required fields exist
                if 'symbol' not in item_dict or 'headline' not in item_dict:
                    continue
                item = RawNewsItem(
                    symbol=item_dict.get('symbol'),
                    timestamp=item_dict.get('timestamp', ''),
                    headline=item_dict.get('headline'),
                    body=item_dict.get('body', ''),
                    provider=item_dict.get('provider', 'unknown')
                )
            else:
                item = item_dict
                
            try:
                enriched.append(self.enrich_item(item))
            except Exception as e:
                logger.error(f"Failed to enrich item: {e}")
        return enriched

    def aggregate_features(self, enriched_items: List[EnrichedNewsFeature]) -> pd.DataFrame:
        """
        Aggregate enriched news items into daily features.
        Returns DataFrame indexed by date.
        """
        if not enriched_items:
            return pd.DataFrame()
            
        data = []
        import pandas as pd
        
        for item in enriched_items:
            # Parse timestamp to date
            try:
                dt = pd.to_datetime(item.timestamp)
                date = dt.date()
            except:
                continue
                
            data.append({
                'date': date,
                'symbol': item.symbol,
                'news_sentiment': item.sentiment_score,
                'news_impact_score': 1.0 if item.directional_impact != 'neutral' else 0.0,
                'news_volatility_score': 1.0 if item.volatility_impact == 'high' else 0.5 if item.volatility_impact == 'medium' else 0.0
            })
            
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()
            
        # Calculate absolute sentiment for shock detection
        df['abs_sentiment'] = df['news_sentiment'].abs()
            
        # Group by date and symbol
        daily = df.groupby(['date', 'symbol']).agg({
            'news_sentiment': 'mean',
            'abs_sentiment': 'max',
            'news_impact_score': 'sum',
            'news_volatility_score': 'max',
            'symbol': 'count' 
        }).rename(columns={'symbol': 'news_count', 'abs_sentiment': 'max_abs_sentiment'})
        
        return daily
