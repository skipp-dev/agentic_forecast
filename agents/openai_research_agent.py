"""
OpenAI Research Agent

Advanced research agent that uses OpenAI to fetch and analyze external news,
sentiment, and market intelligence outside the system's internal data sources.
Provides additional information ingestion capabilities with sentiment analysis.
"""

import os
import sys
import requests
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import re

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.llm.client import LLMClient

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Represents a news article with metadata."""
    title: str
    content: str
    source: str
    published_at: str
    url: str
    symbols: List[str]
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    key_entities: List[str] = None
    impact_assessment: str = ""

    def __post_init__(self):
        if self.key_entities is None:
            self.key_entities = []

@dataclass
class ResearchInsights:
    """Container for research findings."""
    market_sentiment: str
    key_news: List[NewsArticle]
    risk_assessment: str
    trading_signals: List[Dict[str, Any]]
    confidence_score: float
    timestamp: str

class OpenAIResearchAgent:
    """
    Research agent that leverages OpenAI to gather external market intelligence.

    Capabilities:
    - Fetches latest news from financial APIs
    - Analyzes sentiment using advanced NLP
    - Extracts market-relevant insights
    - Provides trading signal recommendations
    - Integrates with existing agent framework
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the research agent.

        Args:
            openai_api_key: OpenAI API key (optional, will try to load from config)
        """
        # Use the new LLM factory for role-based LLM selection
        from src.llm.llm_factory import create_news_features_llm
        self.llm_client = create_news_features_llm()

        # News API configuration (can be extended to multiple sources)
        self.news_api_key = os.getenv("NEWS_API_KEY")  # For NewsAPI.org
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        # Research parameters
        self.max_articles = 50
        self.sentiment_threshold = 0.1
        self.confidence_threshold = 0.7

        logger.info("OpenAI Research Agent initialized")

    def conduct_market_research(self, symbols: Optional[List[str]] = None, days_back: int = 7) -> ResearchInsights:
        """
        Conduct comprehensive autonomous market research.

        This method autonomously expands research scope beyond input symbols to include:
        - All symbols in the watchlist
        - Broader market intelligence (commodities, crypto, macroeconomic indicators)
        - Major indices and economic news
        - Social sentiment across markets

        Args:
            symbols: Optional list of specific symbols to research (if None, uses autonomous mode)
            days_back: Number of days to look back for news

        Returns:
            ResearchInsights object with comprehensive findings
        """
        logger.info("Conducting autonomous market research across all market segments")

        try:
            # If no symbols provided, operate in fully autonomous mode
            if symbols is None or len(symbols) == 0:
                symbols = self._get_autonomous_research_symbols()

            # Expand research scope to include broader market intelligence
            research_queries = self._generate_autonomous_research_queries(symbols)

            # Gather news from all research queries
            all_news = []
            for query_info in research_queries:
                news = self._gather_news_for_query(query_info, days_back)
                all_news.extend(news)

            # Analyze sentiment and extract insights
            analyzed_news = self._analyze_news_sentiment(all_news)

            # Generate comprehensive market insights
            insights = self._generate_comprehensive_market_insights(analyzed_news, symbols)

            # Create research insights object
            research_insights = ResearchInsights(
                market_sentiment=insights['market_sentiment'],
                key_news=analyzed_news[:15],  # Top 15 most relevant
                risk_assessment=insights['risk_assessment'],
                trading_signals=insights['trading_signals'],
                confidence_score=insights['confidence_score'],
                timestamp=datetime.now().isoformat()
            )

            logger.info(f"Autonomous research completed: {len(analyzed_news)} articles analyzed, {len(research_queries)} queries executed")
            return research_insights

        except Exception as e:
            logger.error(f"Autonomous research failed: {e}")
            # Return minimal insights on failure
            return ResearchInsights(
                market_sentiment="neutral",
                key_news=[],
                risk_assessment="Unable to assess - autonomous research failed",
                trading_signals=[],
                confidence_score=0.0,
                timestamp=datetime.now().isoformat()
            )

    def _gather_news(self, symbols: List[str], days_back: int) -> List[NewsArticle]:
        """
        Gather news articles from various sources.

        Currently supports:
        - NewsAPI.org (if API key available)
        - Alpha Vantage news (if API key available)
        - Fallback to simulated news for development
        """
        all_articles = []

        # Try NewsAPI.org
        if self.news_api_key:
            articles = self._fetch_newsapi_news(symbols, days_back)
            all_articles.extend(articles)

        # Try Alpha Vantage
        if self.alpha_vantage_key:
            articles = self._fetch_alpha_vantage_news(symbols, days_back)
            all_articles.extend(articles)

        # If no external APIs available, generate sample news for development
        if not all_articles:
            logger.warning("No external news APIs available, using sample data")
            all_articles = self._generate_sample_news(symbols, days_back)

        return all_articles

    def _fetch_newsapi_news(self, symbols: List[str], days_back: int) -> List[NewsArticle]:
        """Fetch news from NewsAPI.org."""
        articles = []
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        for symbol in symbols[:5]:  # Limit to avoid rate limits
            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f'"{symbol}" stock OR "{symbol}" market',
                    'from': from_date,
                    'sortBy': 'relevancy',
                    'apiKey': self.news_api_key,
                    'pageSize': 10
                }

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                for item in data.get('articles', []):
                    article = NewsArticle(
                        title=item.get('title', ''),
                        content=item.get('description', ''),
                        source=item.get('source', {}).get('name', 'NewsAPI'),
                        published_at=item.get('publishedAt', ''),
                        url=item.get('url', ''),
                        symbols=[symbol]
                    )
                    articles.append(article)

            except Exception as e:
                logger.warning(f"Failed to fetch NewsAPI news for {symbol}: {e}")

        return articles

    def _fetch_alpha_vantage_news(self, symbols: List[str], days_back: int) -> List[NewsArticle]:
        """Fetch news from Alpha Vantage."""
        articles = []

        for symbol in symbols[:3]:  # Limit requests
            try:
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': symbol,
                    'apikey': self.alpha_vantage_key,
                    'limit': 10
                }

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                for item in data.get('feed', []):
                    article = NewsArticle(
                        title=item.get('title', ''),
                        content=item.get('summary', ''),
                        source='Alpha Vantage',
                        published_at=item.get('time_published', ''),
                        url=item.get('url', ''),
                        symbols=[symbol]
                    )
                    articles.append(article)

            except Exception as e:
                logger.warning(f"Failed to fetch Alpha Vantage news for {symbol}: {e}")

        return articles

    def _generate_sample_news(self, symbols: List[str], days_back: int) -> List[NewsArticle]:
        """Generate sample news for development/testing."""
        sample_articles = []

        for symbol in symbols:
            # Create a few sample articles per symbol
            for i in range(2):
                article = NewsArticle(
                    title=f"{symbol} Reports Strong Q4 Earnings",
                    content=f"{symbol} has reported better than expected quarterly earnings, beating analyst estimates by 15%. The company's revenue growth shows strong momentum in key markets.",
                    source="Sample News",
                    published_at=(datetime.now() - timedelta(days=i)).isoformat(),
                    url=f"https://example.com/news/{symbol.lower()}-{i}",
                    symbols=[symbol]
                )
                sample_articles.append(article)

        return sample_articles

    def _analyze_news_sentiment(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """
        Analyze sentiment of news articles using OpenAI.

        Uses advanced NLP to:
        - Determine sentiment polarity
        - Extract key entities
        - Assess market impact
        """
        analyzed_articles = []

        for article in articles:
            try:
                # Use OpenAI for sentiment analysis
                sentiment_data = self._analyze_article_sentiment(article)

                # Update article with analysis
                article.sentiment_score = sentiment_data['sentiment_score']
                article.sentiment_label = sentiment_data['sentiment_label']
                article.key_entities = sentiment_data['key_entities']
                article.impact_assessment = sentiment_data['impact_assessment']

                analyzed_articles.append(article)

            except Exception as e:
                logger.warning(f"Failed to analyze sentiment for article: {e}")
                # Add with neutral sentiment
                article.sentiment_score = 0.0
                article.sentiment_label = "neutral"
                analyzed_articles.append(article)

        return analyzed_articles

    def _analyze_article_sentiment(self, article: NewsArticle) -> Dict[str, Any]:
        """
        Use OpenAI to analyze sentiment and extract insights from an article.
        """
        prompt = f"""
        Analyze the following financial news article for sentiment and market impact:

        Title: {article.title}
        Content: {article.content}
        Symbol: {', '.join(article.symbols)}

        You must respond with ONLY a valid JSON object in this exact format:
        {{
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "key_entities": ["entity1", "entity2"],
            "impact_assessment": "Brief impact description"
        }}

        Rules:
        - sentiment_score: float between -1 (very negative) and 1 (very positive)
        - sentiment_label: must be one of "very_negative", "negative", "neutral", "positive", "very_positive"
        - key_entities: array of important entities mentioned (companies, people, sectors)
        - impact_assessment: brief description of potential market impact
        - Return ONLY the JSON object, no additional text, explanations, or formatting
        """

        try:
            response = self.llm_client.generate(prompt, temperature=0.1)

            # Clean the response to extract JSON
            cleaned_response = self._extract_json_from_response(response.strip())

            # Parse JSON response
            result = json.loads(cleaned_response)
            return result

        except Exception as e:
            logger.error(f"OpenAI sentiment analysis failed: {e}")
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "key_entities": [],
                "impact_assessment": "Analysis failed"
            }

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from LLM response that might contain extra text or formatting.
        """
        import re

        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)

        # Try to find JSON object in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # If no JSON found, try to find anything that looks like JSON
        # Look for opening brace to closing brace
        start_idx = response.find('{')
        end_idx = response.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx:end_idx + 1]

        # If all else fails, return the original response
        return response

    def _generate_market_insights(self, analyzed_articles: List[NewsArticle],
                                symbols: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive market insights from analyzed articles.
        """
        prompt = f"""
        Based on the following analyzed news articles, provide market insights:

        Articles Summary:
        {self._summarize_articles(analyzed_articles)}

        Target Symbols: {', '.join(symbols)}

        You must respond with ONLY a valid JSON object in this exact format:
        {{
            "market_sentiment": "neutral",
            "risk_assessment": "Description of risks",
            "trading_signals": [
                {{"symbol": "AAPL", "action": "buy", "reason": "Positive earnings", "confidence": 0.8}}
            ],
            "confidence_score": 0.7
        }}

        Rules:
        - market_sentiment: must be one of "bullish", "bearish", "neutral", or "mixed"
        - risk_assessment: brief description of current market risks
        - trading_signals: array of specific trading recommendations (can be empty array)
        - confidence_score: float between 0-1 indicating confidence in analysis
        - Return ONLY the JSON object, no additional text, explanations, or formatting
        """

        try:
            response = self.llm_client.generate(prompt, temperature=0.2)
            cleaned_response = self._extract_json_from_response(response.strip())
            insights = json.loads(cleaned_response)
            return insights

        except Exception as e:
            logger.error(f"Market insights generation failed: {e}")
            return {
                "market_sentiment": "neutral",
                "risk_assessment": "Unable to assess market risks",
                "trading_signals": [],
                "confidence_score": 0.0
            }

    def _summarize_articles(self, articles: List[NewsArticle]) -> str:
        """Create a summary of articles for the LLM prompt."""
        summaries = []
        for article in articles[:20]:  # Limit to avoid token limits
            summary = f"- {article.title[:100]}... (Sentiment: {article.sentiment_label}, Score: {article.sentiment_score:.2f})"
            summaries.append(summary)

        return "\n".join(summaries)

    def _get_autonomous_research_symbols(self) -> List[str]:
        """
        Autonomously determine which symbols and market segments to research.

        Returns symbols from watchlist plus broader market intelligence targets.
        """
        # Base symbols from watchlist
        watchlist_symbols = self._load_watchlist_symbols()

        # Expand to include broader market intelligence
        autonomous_symbols = watchlist_symbols.copy()

        # Add major indices
        indices = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']

        # Add commodities and currencies
        commodities = ['GLD', 'USO', 'UNG', 'SLV', 'DBA']

        # Add crypto-related (if not already included)
        crypto_related = ['MSTR', 'COIN', 'SQ', 'PYPL']

        # Add economic indicators and financials
        financials = ['JPM', 'BAC', 'WFC', 'GS', 'MS']

        # Add tech leaders (if not already included)
        tech_leaders = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA']

        # Combine all research targets
        all_targets = (watchlist_symbols + indices + commodities +
                      crypto_related + financials + tech_leaders)

        # Remove duplicates while preserving order
        seen = set()
        autonomous_symbols = []
        for symbol in all_targets:
            if symbol not in seen:
                seen.add(symbol)
                autonomous_symbols.append(symbol)

        logger.info(f"Autonomous research scope: {len(autonomous_symbols)} symbols from watchlist + market intelligence")
        return autonomous_symbols

    def _load_watchlist_symbols(self) -> List[str]:
        """Load symbols from watchlist files."""
        symbols = []

        # Try to load from watchlist CSV files
        watchlist_files = ['watchlist_ibkr.csv', 'watchlist_test.csv']

        for filename in watchlist_files:
            if os.path.exists(filename):
                try:
                    import pandas as pd
                    df = pd.read_csv(filename)
                    if 'symbol' in df.columns:
                        file_symbols = df['symbol'].dropna().unique().tolist()
                        symbols.extend(file_symbols)
                    elif len(df.columns) > 0:
                        # Assume first column contains symbols
                        file_symbols = df.iloc[:, 0].dropna().unique().tolist()
                        symbols.extend(file_symbols)
                except Exception as e:
                    logger.warning(f"Failed to load symbols from {filename}: {e}")

        # Remove duplicates
        symbols = list(set(symbols))

        # If no symbols found, use defaults
        if not symbols:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']

        return symbols

    def _generate_autonomous_research_queries(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Generate autonomous research queries covering various market segments.

        Returns a list of query dictionaries with search terms and categories.
        """
        queries = []

        # 1. Individual symbol research
        for symbol in symbols[:20]:  # Limit to avoid API limits
            queries.append({
                'query': f'"{symbol}" stock OR "{symbol}" market OR "{symbol}" earnings OR "{symbol}" news',
                'category': 'individual_symbol',
                'symbol': symbol,
                'priority': 'high'
            })

        # 2. Major market indices and broad market sentiment
        market_queries = [
            {
                'query': 'S&P 500 OR SPY OR stock market OR market sentiment OR market outlook',
                'category': 'major_indices',
                'symbol': 'SP500',
                'priority': 'high'
            },
            {
                'query': 'NASDAQ OR QQQ OR tech stocks OR technology sector',
                'category': 'major_indices',
                'symbol': 'NASDAQ',
                'priority': 'high'
            },
            {
                'query': 'DOW OR DIA OR industrial stocks OR blue chip stocks',
                'category': 'major_indices',
                'symbol': 'DOW',
                'priority': 'medium'
            }
        ]
        queries.extend(market_queries)

        # 3. Commodities and resources
        commodity_queries = [
            {
                'query': 'gold OR GLD OR precious metals OR gold price OR gold mining',
                'category': 'commodities',
                'symbol': 'GOLD',
                'priority': 'medium'
            },
            {
                'query': 'oil OR USO OR crude oil OR energy prices OR oil market',
                'category': 'commodities',
                'symbol': 'OIL',
                'priority': 'medium'
            },
            {
                'query': 'natural gas OR UNG OR gas prices OR energy sector',
                'category': 'commodities',
                'symbol': 'GAS',
                'priority': 'low'
            },
            {
                'query': 'agriculture OR DBA OR farming OR crop prices OR food prices',
                'category': 'commodities',
                'symbol': 'AGRICULTURE',
                'priority': 'low'
            }
        ]
        queries.extend(commodity_queries)

        # 4. Cryptocurrency and blockchain
        crypto_queries = [
            {
                'query': 'bitcoin OR BTC OR cryptocurrency OR crypto market OR blockchain',
                'category': 'cryptocurrency',
                'symbol': 'BTC',
                'priority': 'high'
            },
            {
                'query': 'ethereum OR ETH OR altcoins OR DeFi OR NFT',
                'category': 'cryptocurrency',
                'symbol': 'ETH',
                'priority': 'medium'
            },
            {
                'query': 'crypto regulation OR SEC crypto OR crypto ETF OR digital assets',
                'category': 'cryptocurrency',
                'symbol': 'CRYPTO',
                'priority': 'medium'
            }
        ]
        queries.extend(crypto_queries)

        # 5. Economic indicators and Fed policy
        economic_queries = [
            {
                'query': 'Federal Reserve OR Fed OR interest rates OR rate decision OR monetary policy',
                'category': 'economic_policy',
                'symbol': 'FED',
                'priority': 'high'
            },
            {
                'query': 'inflation OR CPI OR consumer prices OR inflation data',
                'category': 'economic_indicators',
                'symbol': 'INFLATION',
                'priority': 'high'
            },
            {
                'query': 'employment OR jobs report OR unemployment OR labor market',
                'category': 'economic_indicators',
                'symbol': 'LABOR',
                'priority': 'high'
            },
            {
                'query': 'GDP OR economic growth OR recession OR economic data',
                'category': 'economic_indicators',
                'symbol': 'GDP',
                'priority': 'medium'
            },
            {
                'query': 'geopolitical OR international trade OR tariffs OR global economy',
                'category': 'geopolitical',
                'symbol': 'GEOPOLITICAL',
                'priority': 'medium'
            }
        ]
        queries.extend(economic_queries)

        # 6. Sector-specific news
        sector_queries = [
            {
                'query': 'artificial intelligence OR AI OR machine learning OR tech innovation',
                'category': 'sector_news',
                'symbol': 'AI_TECH',
                'priority': 'high'
            },
            {
                'query': 'electric vehicles OR EV OR autonomous driving OR Tesla competitors',
                'category': 'sector_news',
                'symbol': 'EV_AUTO',
                'priority': 'medium'
            },
            {
                'query': 'renewable energy OR solar OR wind OR green energy OR climate',
                'category': 'sector_news',
                'symbol': 'RENEWABLES',
                'priority': 'medium'
            }
        ]
        queries.extend(sector_queries)

        logger.info(f"Generated {len(queries)} autonomous research queries across {len(set(q['category'] for q in queries))} categories")
        return queries

    def _gather_news_for_query(self, query_info: Dict[str, Any], days_back: int) -> List[NewsArticle]:
        """
        Gather news for a specific research query.

        Args:
            query_info: Dictionary with query details
            days_back: Number of days to look back

        Returns:
            List of NewsArticle objects
        """
        query = query_info['query']
        category = query_info['category']
        symbol = query_info['symbol']

        articles = []

        # Try NewsAPI.org
        if self.news_api_key:
            try:
                articles.extend(self._fetch_newsapi_with_query(query, symbol, days_back))
            except Exception as e:
                logger.warning(f"NewsAPI failed for {symbol}: {e}")

        # Try Alpha Vantage (for financial symbols)
        if self.alpha_vantage_key and symbol in ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'USO', 'UNG', 'SLV']:
            try:
                articles.extend(self._fetch_alpha_vantage_for_symbol(symbol, days_back))
            except Exception as e:
                logger.warning(f"Alpha Vantage failed for {symbol}: {e}")

        # If no external APIs available, generate sample news for development
        if not articles:
            articles = self._generate_sample_news_for_category(category, symbol, days_back)

        return articles

    def _fetch_newsapi_with_query(self, query: str, symbol: str, days_back: int) -> List[NewsArticle]:
        """Fetch news from NewsAPI.org with custom query."""
        articles = []
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'apiKey': self.news_api_key,
                'pageSize': 5  # Limit per query to avoid rate limits
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            for item in data.get('articles', []):
                article = NewsArticle(
                    title=item.get('title', ''),
                    content=item.get('description', ''),
                    source=item.get('source', {}).get('name', 'NewsAPI'),
                    published_at=item.get('publishedAt', ''),
                    url=item.get('url', ''),
                    symbols=[symbol]
                )
                articles.append(article)

        except Exception as e:
            logger.warning(f"Failed to fetch NewsAPI news for query '{query}': {e}")

        return articles

    def _fetch_alpha_vantage_for_symbol(self, symbol: str, days_back: int) -> List[NewsArticle]:
        """Fetch news from Alpha Vantage for specific symbol."""
        articles = []

        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.alpha_vantage_key,
                'limit': 5
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            for item in data.get('feed', []):
                article = NewsArticle(
                    title=item.get('title', ''),
                    content=item.get('summary', ''),
                    source='Alpha Vantage',
                    published_at=item.get('time_published', ''),
                    url=item.get('url', ''),
                    symbols=[symbol]
                )
                articles.append(article)

        except Exception as e:
            logger.warning(f"Failed to fetch Alpha Vantage news for {symbol}: {e}")

        return articles

    def _generate_sample_news_for_category(self, category: str, symbol: str, days_back: int) -> List[NewsArticle]:
        """Generate sample news for development when APIs unavailable."""
        sample_templates = {
            'individual_symbol': [
                f"{symbol} Reports Strong Quarterly Performance",
                f"Analysts Update {symbol} Price Targets",
                f"{symbol} Announces Strategic Partnership",
                f"Market Reaction to {symbol} Latest Developments"
            ],
            'major_indices': [
                f"Market Sentiment Shows {symbol} Momentum",
                f"Technical Analysis Points to {symbol} Direction",
                f"Institutional Investors Position for {symbol} Moves",
                f"Economic Data Influences {symbol} Performance"
            ],
            'commodities': [
                f"{symbol} Prices Show Volatility Amid Global Events",
                f"Supply Chain Issues Impact {symbol} Markets",
                f"Geopolitical Tensions Affect {symbol} Trading",
                f"Weather Patterns Influence {symbol} Outlook"
            ],
            'cryptocurrency': [
                f"Regulatory Developments Shape {symbol} Market",
                f"Institutional Adoption Drives {symbol} Interest",
                f"Technological Advances Boost {symbol} Ecosystem",
                f"Market Sentiment Shifts for {symbol} Assets"
            ],
            'economic_policy': [
                f"Fed Policy Decisions Create {symbol} Uncertainty",
                f"Central Bank Communications Move Markets",
                f"Interest Rate Expectations Impact Economy",
                f"Monetary Policy Affects Global Markets"
            ],
            'economic_indicators': [
                f"Economic Data Reveals {symbol} Trends",
                f"Employment Figures Influence Market Sentiment",
                f"Inflation Readings Shape Economic Outlook",
                f"GDP Growth Data Affects Investment Decisions"
            ]
        }

        templates = sample_templates.get(category, sample_templates['individual_symbol'])
        articles = []

        for i, template in enumerate(templates):
            article = NewsArticle(
                title=template,
                content=f"Market analysis indicates significant developments in {symbol} sector with potential impact on broader market trends. Experts suggest monitoring key indicators for further insights.",
                source="Sample Intelligence",
                published_at=(datetime.now() - timedelta(days=i)).isoformat(),
                url=f"https://example.com/analysis/{symbol.lower()}-{i}",
                symbols=[symbol]
            )
            articles.append(article)

        return articles

    def _generate_comprehensive_market_insights(self, analyzed_articles: List[NewsArticle],
                                              symbols: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive market insights from analyzed articles across all market segments.
        """
        prompt = f"""
        Analyze the following comprehensive market intelligence data and provide insights:

        Analyzed Articles Summary ({len(analyzed_articles)} total):
        {self._summarize_articles(analyzed_articles)}

        Research Scope: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}

        Market Segments Covered:
        - Individual company news and sentiment
        - Major market indices (S&P 500, NASDAQ, DOW)
        - Commodities (Gold, Oil, Natural Gas, Agriculture)
        - Cryptocurrency and blockchain developments
        - Economic indicators and Fed policy
        - Geopolitical and international factors
        - Sector-specific trends and innovations

        You must respond with ONLY a valid JSON object in this exact format:
        {{
            "market_sentiment": "bullish",
            "risk_assessment": "Detailed risk analysis considering all factors",
            "trading_signals": [
                {{
                    "symbol": "AAPL",
                    "action": "buy",
                    "reason": "Specific reasoning based on news and sentiment",
                    "confidence": 0.8,
                    "timeframe": "short_term"
                }}
            ],
            "confidence_score": 0.85,
            "key_drivers": ["List of main market drivers identified"],
            "market_outlook": "Brief market outlook summary"
        }}

        Rules:
        - market_sentiment: must be one of "bullish", "bearish", "neutral", "mixed"
        - risk_assessment: detailed risk analysis considering all market factors
        - trading_signals: array of specific trading recommendations (can be empty array)
        - confidence_score: float between 0-1 indicating confidence in analysis
        - key_drivers: array of main market drivers identified
        - market_outlook: brief market outlook summary
        - Return ONLY the JSON object, no additional text, explanations, or formatting
        """

        try:
            response = self.llm_client.generate(prompt, temperature=0.2, max_tokens=2000)
            cleaned_response = self._extract_json_from_response(response.strip())
            insights = json.loads(cleaned_response)
            return insights

        except Exception as e:
            logger.error(f"Comprehensive market insights generation failed: {e}")
            return {
                "market_sentiment": "neutral",
                "risk_assessment": "Unable to assess comprehensive market risks",
                "trading_signals": [],
                "confidence_score": 0.0,
                "key_drivers": [],
                "market_outlook": "Analysis unavailable"
            }

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of the research agent.
        """
        return {
            "agent_type": "OpenAIResearchAgent",
            "openai_available": self.llm_client.client is not None,
            "news_api_available": self.news_api_key is not None,
            "alpha_vantage_available": self.alpha_vantage_key is not None,
            "autonomous_mode": True,
            "status": "operational" if self.llm_client.client else "degraded"
        }