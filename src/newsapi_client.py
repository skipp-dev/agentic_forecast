import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NewsAPIClient:
    """
    Client for fetching news from NewsAPI.ai (Event Registry) or NewsAPI.org.
    Defaults to NewsAPI.ai as per Sprint 2 requirements.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.newsapi_ai_config = config.get('newsapi_ai', {})
        self.news_api_config = config.get('news_api', {})
        
        self.use_ai = self.newsapi_ai_config.get('enabled', False)
        self.ai_key = self.newsapi_ai_config.get('api_key')
        self.org_key = self.news_api_config.get('api_key')
        
        if self.use_ai and not self.ai_key:
            logger.warning("NewsAPI.ai enabled but no key found. Falling back to NewsAPI.org if available.")
            self.use_ai = False
            
    def get_news_for_symbol(self, symbol: str, start_date: str, end_date: str, limit: int = 10) -> List[Dict]:
        if self.use_ai:
            return self._fetch_newsapi_ai(symbol, start_date, end_date, limit)
        else:
            return self._fetch_newsapi_org(symbol, start_date, end_date, limit)

    def _fetch_newsapi_ai(self, symbol: str, start_date: str, end_date: str, limit: int) -> List[Dict]:
        """Fetch from NewsAPI.ai (Event Registry)"""
        url = "http://eventregistry.org/api/v1/article/getArticles"
        
        # Simple keyword search for the symbol
        # In a real implementation, we might map symbol to company name
        payload = {
            "action": "getArticles",
            "keyword": symbol,
            "articlesPage": 1,
            "articlesCount": limit,
            "articlesSortBy": "date",
            "articlesSortByAsc": False,
            "dateStart": start_date,
            "dateEnd": end_date,
            "apiKey": self.ai_key,
            "resultType": "articles",
            "dataType": ["news", "pr"],
            "lang": "eng"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = data.get('articles', {}).get('results', [])
            return self._normalize_ai_articles(articles, symbol)
        except Exception as e:
            logger.error(f"NewsAPI.ai fetch failed for {symbol}: {e}")
            return []

    def _fetch_newsapi_org(self, symbol: str, start_date: str, end_date: str, limit: int) -> List[Dict]:
        """Fetch from NewsAPI.org"""
        if not self.org_key:
            logger.warning("No NewsAPI.org key provided.")
            return []
            
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "from": start_date,
            "to": end_date,
            "sortBy": "publishedAt",
            "apiKey": self.org_key,
            "language": "en",
            "pageSize": limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = data.get('articles', [])
            return self._normalize_org_articles(articles, symbol)
        except Exception as e:
            logger.error(f"NewsAPI.org fetch failed for {symbol}: {e}")
            return []

    def _normalize_ai_articles(self, articles: List[Dict], symbol: str) -> List[Dict]:
        normalized = []
        for art in articles:
            normalized.append({
                "symbol": symbol,
                "timestamp": art.get("dateTime"), # ISO format
                "headline": art.get("title"),
                "body": art.get("body"),
                "url": art.get("url"),
                "source": art.get("source", {}).get("title"),
                "provider": "newsapi.ai"
            })
        return normalized

    def _normalize_org_articles(self, articles: List[Dict], symbol: str) -> List[Dict]:
        normalized = []
        for art in articles:
            normalized.append({
                "symbol": symbol,
                "timestamp": art.get("publishedAt"),
                "headline": art.get("title"),
                "body": art.get("description") or art.get("content"),
                "url": art.get("url"),
                "source": art.get("source", {}).get("name"),
                "provider": "newsapi.org"
            })
        return normalized
