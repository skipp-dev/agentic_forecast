from typing import List, Dict, Any
from dataclasses import dataclass
import logging
import pandas as pd
import json
from src.configs.llm_prompts import extract_json_from_response

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
            from src.configs.llm_prompts import get_prompt
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
        data = {}
        if hasattr(self.llm, 'chat'):
            try:
                # Ensure we get a string response
                response = self.llm.chat(prompt)
                if hasattr(response, 'content'): # Handle object with content attr
                     raw_text = response.content
                elif hasattr(response, 'json'): # Handle requests-like object (unlikely for chat but possible)
                     # This was the old suspicious code, but if it returns a response object, we want text
                     raw_text = response.text if hasattr(response, 'text') else str(response)
                else:
                     raw_text = str(response)

                data = extract_json_from_response(raw_text)
            except Exception as e:
                logger.warning(f"LLM enrichment failed: {e}")
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
            
        # Convert to list of dicts for easier processing
        items_dicts = []
        for item in enriched_items:
            # Parse timestamp to date
            try:
                dt = pd.to_datetime(item.timestamp)
                date = dt.date()
            except:
                continue
            
            # Map impact string to score/level
            impact_level = "no_impact"
            if item.directional_impact != 'neutral':
                impact_level = "mild" # Default if not specified
                # Heuristic: if volatility is high, maybe it's strong?
                # But let's stick to what we have. The user prompt implies 'impact' field exists.
                # Our EnrichedNewsFeature has 'directional_impact' and 'volatility_impact'.
                # Let's map them.
            
            # We need to map our EnrichedNewsFeature fields to the expected dict format for scoring
            # EnrichedNewsFeature: symbol, timestamp, headline, categories, directional_impact, impact_horizon, volatility_impact, confidence, notes, sentiment_score
            
            # Infer 'impact' ("no_impact" | "mild" | "strong") from our fields
            impact = "no_impact"
            if item.volatility_impact == 'high' or abs(item.sentiment_score) > 0.7:
                impact = "strong"
            elif item.directional_impact != 'neutral' or item.volatility_impact == 'medium':
                impact = "mild"
                
            # Flatten categories if it's a list
            category = item.categories[0] if item.categories else "other"
            
            items_dicts.append({
                'symbol': item.symbol,
                'timestamp_utc': item.timestamp,
                'date': date,
                'sentiment_score': item.sentiment_score,
                'impact': impact,
                'category': category,
                'categories': item.categories, # Keep full list
                'volatility_impact': item.volatility_impact
            })

        # 1. Select Top 5 News Items per Symbol/Day based on Impact Score
        selected_items = self._select_top_news_items(items_dicts)
        
        if not selected_items:
            return pd.DataFrame()

        # 2. Build Daily Aggregates
        daily_data = []
        
        # Group by symbol and date
        df_items = pd.DataFrame(selected_items)
        grouped = df_items.groupby(['symbol', 'date'])
        
        for (symbol, date), group in grouped:
            # Basic aggregates
            avg_sentiment = group['sentiment_score'].mean()
            max_abs_sentiment = group['sentiment_score'].abs().max()
            news_count = len(group)
            
            # New Features
            strong_impact_count = len(group[group['impact'] == 'strong'])
            
            # Category flags
            all_cats = []
            for cats in group['categories']:
                if isinstance(cats, list):
                    all_cats.extend([c.lower() for c in cats])
                else:
                    all_cats.append(str(cats).lower())
            
            has_earnings = any('earnings' in c for c in all_cats)
            has_mna = any('mna' in c or 'merger' in c or 'acquisition' in c for c in all_cats)
            has_litigation = any('litigation' in c or 'lawsuit' in c or 'legal' in c for c in all_cats)
            has_fda = any('fda' in c or 'approval' in c or 'drug' in c for c in all_cats) or \
                      (any('product' in c for c in all_cats) and any('fda' in h.lower() for h in group['headline'] if 'headline' in group))
            
            # Legacy scores (keep for compatibility)
            news_impact_score = group.apply(lambda x: 1.0 if x['impact'] != 'no_impact' else 0.0, axis=1).sum()
            news_volatility_score = group.apply(lambda x: 1.0 if x['volatility_impact'] == 'high' else 0.5 if x['volatility_impact'] == 'medium' else 0.0, axis=1).max()

            daily_data.append({
                'date': date,
                'symbol': symbol,
                'news_sentiment': avg_sentiment,
                'abs_sentiment': max_abs_sentiment, # Renamed to match legacy return
                'max_abs_sentiment': max_abs_sentiment,
                'news_count': news_count,
                'news_impact_score': news_impact_score,
                'news_volatility_score': news_volatility_score,
                
                # New Features
                'strong_impact_count': strong_impact_count,
                'has_earnings_news': 1 if has_earnings else 0,
                'has_mna_news': 1 if has_mna else 0,
                'has_litigation_news': 1 if has_litigation else 0,
                'has_fda_news': 1 if has_fda else 0
            })
            
        df_daily = pd.DataFrame(daily_data)
        if df_daily.empty:
            return pd.DataFrame()
            
        # Set index for rolling calculations
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        df_daily = df_daily.sort_values(['symbol', 'date'])
        
        # 3. Calculate Rolling Features
        # news_sentiment_3d_ma, news_sentiment_5d_ma
        # strong_impact_7d_count
        
        df_daily['news_sentiment_3d_ma'] = df_daily.groupby('symbol')['news_sentiment'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df_daily['news_sentiment_5d_ma'] = df_daily.groupby('symbol')['news_sentiment'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df_daily['strong_impact_7d_count'] = df_daily.groupby('symbol')['strong_impact_count'].transform(
            lambda x: x.rolling(window=7, min_periods=1).sum()
        )
        
        # 4. Calculate time_since_last_strong_news
        # This is a bit more complex in pandas. 
        # We can create a mask where strong_impact_count > 0, then forward fill the date.
        
        def calc_time_since_strong(group):
            # Identify dates with strong news
            strong_dates = group.loc[group['strong_impact_count'] > 0, 'date']
            if strong_dates.empty:
                return pd.Series([999] * len(group), index=group.index) # Default large value
            
            # Reindex strong_dates to full index and ffill
            last_strong = pd.Series(pd.NaT, index=group.index)
            last_strong.loc[group['strong_impact_count'] > 0] = group['date']
            last_strong = last_strong.ffill()
            
            # Calculate days difference
            diff = (group['date'] - last_strong).dt.days
            return diff.fillna(999) # Fill initial NaNs with large value

        df_daily['time_since_last_strong_news'] = df_daily.groupby('symbol').apply(calc_time_since_strong).reset_index(level=0, drop=True)

        # Final formatting to match expected output (MultiIndex date, symbol)
        # The original code returned: daily = df.groupby(['date', 'symbol']).agg(...)
        # which results in MultiIndex.
        
        return df_daily.set_index(['date', 'symbol'])

    def _compute_news_impact_score(self, item: Dict) -> float:
        """
        Compute an impact score for a news item to prioritize top items.
        """
        CATEGORY_BASE_SCORES = {
            "earnings": 3.0,
            "guidance": 3.0,
            "mna": 3.0,
            "merger": 3.0,
            "acquisition": 3.0,
            "litigation": 2.5,
            "product": 2.5,
            "macro": 2.5,
            "management": 2.0,
            "sector": 2.0,
            "other": 1.0,
        }
        
        score = 0.0
        
        # Base category score
        cat = str(item.get('category', 'other')).lower()
        # Simple partial match
        base_score = 1.0
        for key, val in CATEGORY_BASE_SCORES.items():
            if key in cat:
                base_score = max(base_score, val)
        score += base_score
        
        # Impact bonus
        impact = item.get('impact', 'no_impact')
        if impact == 'strong':
            score += 2.0
        elif impact == 'mild':
            score += 0.5
            
        # Sentiment magnitude bonus
        score += 1.0 * abs(item.get('sentiment_score', 0.0))
        
        return score

    def _select_top_news_items(self, items: List[Dict], max_items: int = 5) -> List[Dict]:
        """
        Select top K news items per symbol/day based on impact score.
        """
        # Add scores
        for item in items:
            item['_impact_score'] = self._compute_news_impact_score(item)
            
        df = pd.DataFrame(items)
        if df.empty:
            return []
            
        # Sort by symbol, date, score (desc)
        df = df.sort_values(['symbol', 'date', '_impact_score'], ascending=[True, True, False])
        
        # Group and head
        top_items = df.groupby(['symbol', 'date']).head(max_items)
        
        return top_items.to_dict('records')
