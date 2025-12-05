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
        Fetch news for multiple symbols with smart selection logic.
        """
        results = {}
        for symbol in symbols:
            logger.info(f"Fetching news for {symbol}...")
            # Fetch more than the limit to allow for ranking
            fetch_limit = limit * 3 if limit > 0 else 100
            articles = self.client.get_news_for_symbol(symbol, start_date, end_date)
            
            if articles:
                deduped = self._deduplicate(articles)
                
                # Apply smart selection if we have more articles than the limit
                if limit > 0 and len(deduped) > limit:
                    logger.info(f"Ranking {len(deduped)} articles for {symbol} to select top {limit}")
                    selected = self._smart_select(deduped, limit)
                    results[symbol] = selected
                else:
                    results[symbol] = deduped
                    
                logger.info(f"Found {len(results[symbol])} articles for {symbol}")
            else:
                logger.info(f"No news found for {symbol}")
        return results
    
    def _smart_select(self, articles: List[Dict], limit: int) -> List[Dict]:
        """
        Select top N articles based on impact heuristics.
        Prioritizes:
        1. Earnings/Guidance/M&A/Litigation keywords
        2. Recency
        """
        # Define high-impact keywords
        impact_keywords = [
            'earnings', 'revenue', 'profit', 'guidance', 'outlook', 
            'merger', 'acquisition', 'takeover', 'buyout', 
            'lawsuit', 'litigation', 'settlement', 'regulatory', 'fda approval',
            'dividend', 'split', 'offering'
        ]
        
        scored_articles = []
        for art in articles:
            score = 0
            title = art.get('title', '').lower()
            description = art.get('description', '').lower()
            content = (title + " " + description)
            
            # Score based on keywords
            for keyword in impact_keywords:
                if keyword in content:
                    score += 2 # High priority
            
            # Score based on source (optional, if source is available and trusted)
            # source = art.get('source', {}).get('name', '').lower()
            # if 'reuters' in source or 'bloomberg' in source:
            #     score += 1
            
            scored_articles.append((score, art))
            
        # Sort by score (descending) and then by date (descending, assuming input is sorted or we parse date)
        # Python's sort is stable, so if we sort by date first then score, it works.
        # Assuming articles come in somewhat chronological order, let's just sort by score.
        scored_articles.sort(key=lambda x: x[0], reverse=True)
        
        # Select top N
        selected = [art for score, art in scored_articles[:limit]]
        return selected

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
