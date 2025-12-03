import requests
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import time

logger = logging.getLogger(__name__)

class NewsAPIClient:
    """
    Client for fetching news from external providers (NewsAPI.ai / Event Registry).
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('news', {})
        self.provider = self.config.get('provider', 'newsapi_ai')
        self.api_key = self.config.get('api_key')
        self.base_url = "http://eventregistry.org/api/v1" if self.provider == 'newsapi_ai' else "https://newsapi.org/v2"
        
        if not self.api_key:
            logger.warning("News API key not found in config. News features will be disabled.")

    def get_news_for_symbol(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a specific symbol within a date range.
        """
        if not self.api_key:
            return []

        if self.provider == 'newsapi_ai':
            return self._fetch_newsapi_ai(symbol, start_date, end_date)
        else:
            return self._fetch_newsapi_org(symbol, start_date, end_date)

    def _fetch_newsapi_ai(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Fetch from NewsAPI.ai (Event Registry).
        """
        url = f"{self.base_url}/article/getArticles"
        
        # Map symbol to concept/keyword (simplified)
        # In a real system, we'd map 'AAPL' to 'Apple Inc.' URI
        keyword = symbol 
        
        payload = {
            "action": "getArticles",
            "keyword": keyword,
            "articlesPage": 1,
            "articlesCount": self.config.get('max_articles_per_symbol_per_day', 50),
            "articlesSortBy": "date",
            "articlesSortByAsc": False,
            "articlesArticleBodyLen": -1,
            "resultType": "articles",
            "dataType": [
                "news",
                "pr"
            ],
            "apiKey": self.api_key,
            "forceMaxDataTimeWindow": 31,
            "dateStart": start_date,
            "dateEnd": end_date
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = data.get('articles', {}).get('results', [])
            return [self._normalize_article(a, symbol) for a in articles]
            
        except Exception as e:
            logger.error(f"NewsAPI.ai fetch failed for {symbol}: {e}")
            return []

    def _fetch_newsapi_org(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Fetch from NewsAPI.org (Fallback).
        """
        url = f"{self.base_url}/everything"
        
        params = {
            "q": symbol,
            "from": start_date,
            "to": end_date,
            "sortBy": "relevancy",
            "apiKey": self.api_key,
            "language": "en"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = data.get('articles', [])
            return [self._normalize_article(a, symbol) for a in articles]
            
        except Exception as e:
            logger.error(f"NewsAPI.org fetch failed for {symbol}: {e}")
            return []

    def _normalize_article(self, raw_article: Dict, symbol: str) -> Dict[str, Any]:
        """
        Normalize article format across providers.
        """
        if self.provider == 'newsapi_ai':
            return {
                "symbol": symbol,
                "title": raw_article.get('title'),
                "headline": raw_article.get('title'), # Alias
                "body": raw_article.get('body'),
                "url": raw_article.get('url'),
                "source": raw_article.get('source', {}).get('title'),
                "timestamp": raw_article.get('dateTime'), # ISO format
                "provider": "newsapi_ai"
            }
        else:
            return {
                "symbol": symbol,
                "title": raw_article.get('title'),
                "headline": raw_article.get('title'), # Alias
                "body": raw_article.get('description') or raw_article.get('content'),
                "url": raw_article.get('url'),
                "source": raw_article.get('source', {}).get('name'),
                "timestamp": raw_article.get('publishedAt'),
                "provider": "newsapi_org"
            }
