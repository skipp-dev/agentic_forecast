import pandas as pd
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from ..clients.newsapi_client import NewsAPIClient

logger = logging.getLogger(__name__)

class NewsDataAgent:
    """
    Agent for collecting and preprocessing news data.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.client = NewsAPIClient(config)
        
    def fetch_news(self, symbols: List[str], start_date: str, end_date: str, limit: int = 5) -> Dict[str, List[Dict]]:
        """
        Fetch news for multiple symbols.
        """
        results = {}
        for symbol in symbols:
            logger.info(f"Fetching news for {symbol}...")
            articles = self.client.get_news_for_symbol(symbol, start_date, end_date)
            if articles:
                deduped = self._deduplicate(articles)
                # Apply limit
                if limit > 0 and len(deduped) > limit:
                    logger.info(f"Limiting news for {symbol} to {limit} articles (found {len(deduped)})")
                    deduped = deduped[:limit]
                results[symbol] = deduped
                logger.info(f"Found {len(results[symbol])} articles for {symbol}")
            else:
                logger.info(f"No news found for {symbol}")
        return results
    
    def _deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Deduplicate articles based on headline similarity or URL."""
        seen_urls = set()
        unique_articles = []
        
        for art in articles:
            url = art.get('url')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(art)
                
        return unique_articles
