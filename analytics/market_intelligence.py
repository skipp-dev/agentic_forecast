"""
Market Intelligence

Market data analysis and intelligence for IB Forecast system.
Provides market trend analysis, sentiment analysis, and trading signals.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
from collections import Counter

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.metrics_database import MetricsDatabase, MetricQuery

logger = logging.getLogger(__name__)

class MarketIntelligence:
    """
    Market intelligence and analysis system.

    Provides:
    - Market trend analysis
    - Sentiment analysis
    - Trading signal generation
    - Market regime detection
    - Correlation analysis
    """

    def __init__(self, metrics_db: Optional[MetricsDatabase] = None):
        """
        Initialize market intelligence.

        Args:
            metrics_db: Metrics database instance
        """
        self.metrics_db = metrics_db or MetricsDatabase()

        # Market intelligence configuration
        self.market_config = {
            'trend_window': 20,  # days for trend analysis
            'volatility_window': 30,  # days for volatility calculation
            'correlation_window': 60,  # days for correlation analysis
            'sentiment_thresholds': {
                'bullish': 0.6,
                'bearish': 0.4,
                'neutral': 0.5
            },
            'regime_thresholds': {
                'high_volatility': 0.03,  # 3% daily volatility
                'low_volatility': 0.01    # 1% daily volatility
            }
        }

        logger.info("Market Intelligence initialized")

    def analyze_market_trends(self, price_data: pd.DataFrame,
                             indicators: List[str] = None) -> Dict[str, Any]:
        """
        Analyze market trends using technical indicators.

        Args:
            price_data: OHLC price data DataFrame
            indicators: List of indicators to calculate

        Returns:
            Dictionary with trend analysis results
        """
        if indicators is None:
            indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger']

        results = {}

        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in price_data.columns for col in required_cols):
            return {'status': 'error', 'message': 'Missing required OHLC columns'}

        # Calculate basic price metrics
        results['price_metrics'] = self._calculate_price_metrics(price_data)

        # Calculate technical indicators
        results['indicators'] = {}
        for indicator in indicators:
            if hasattr(self, f'_calculate_{indicator}'):
                results['indicators'][indicator] = getattr(self, f'_calculate_{indicator}')(price_data)
            else:
                logger.warning(f"Unknown indicator: {indicator}")

        # Determine overall trend
        results['trend_analysis'] = self._analyze_overall_trend(price_data, results['indicators'])

        return results

    def analyze_market_sentiment(self, text_data: List[str] = None,
                                news_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze market sentiment from text and news data.

        Args:
            text_data: List of text strings to analyze
            news_data: News data DataFrame with text and timestamps

        Returns:
            Dictionary with sentiment analysis results
        """
        results = {'status': 'success'}

        if text_data:
            results['text_sentiment'] = self._analyze_text_sentiment(text_data)

        if news_data is not None and not news_data.empty:
            results['news_sentiment'] = self._analyze_news_sentiment(news_data)

        # Combine sentiment scores
        if 'text_sentiment' in results or 'news_sentiment' in results:
            results['overall_sentiment'] = self._combine_sentiment_scores(results)

        return results

    def detect_market_regime(self, returns: pd.Series,
                           volatility: pd.Series = None) -> Dict[str, Any]:
        """
        Detect current market regime based on volatility and returns.

        Args:
            returns: Asset returns series
            volatility: Volatility series (optional)

        Returns:
            Dictionary with regime detection results
        """
        if volatility is None:
            volatility = returns.rolling(window=self.market_config['volatility_window']).std()

        # Calculate regime indicators
        current_volatility = volatility.iloc[-1] if not volatility.empty else 0
        avg_volatility = volatility.mean()

        # Determine regime
        if current_volatility > self.market_config['regime_thresholds']['high_volatility']:
            regime = 'high_volatility'
            description = 'High volatility market regime - increased risk'
        elif current_volatility < self.market_config['regime_thresholds']['low_volatility']:
            regime = 'low_volatility'
            description = 'Low volatility market regime - stable conditions'
        else:
            regime = 'normal_volatility'
            description = 'Normal volatility market regime'

        # Trend analysis
        recent_returns = returns.tail(20)
        trend_direction = 'upward' if recent_returns.mean() > 0 else 'downward'

        # Momentum
        momentum = returns.tail(5).mean() - returns.tail(20).mean()

        return {
            'regime': regime,
            'description': description,
            'current_volatility': float(current_volatility),
            'average_volatility': float(avg_volatility),
            'trend_direction': trend_direction,
            'momentum': float(momentum),
            'regime_confidence': self._calculate_regime_confidence(volatility, current_volatility)
        }

    def generate_trading_signals(self, price_data: pd.DataFrame,
                               indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on technical analysis.

        Args:
            price_data: OHLC price data
            indicators: Calculated technical indicators

        Returns:
            Dictionary with trading signals
        """
        signals = {}

        # Moving average signals
        if 'sma' in indicators:
            signals['ma_signal'] = self._generate_ma_signals(price_data, indicators['sma'])

        # RSI signals
        if 'rsi' in indicators:
            signals['rsi_signal'] = self._generate_rsi_signals(indicators['rsi'])

        # MACD signals
        if 'macd' in indicators:
            signals['macd_signal'] = self._generate_macd_signals(indicators['macd'])

        # Bollinger Band signals
        if 'bollinger' in indicators:
            signals['bollinger_signal'] = self._generate_bollinger_signals(price_data, indicators['bollinger'])

        # Combine signals for overall recommendation
        signals['overall_signal'] = self._combine_trading_signals(signals)

        return signals

    def analyze_market_correlations(self, assets_data: pd.DataFrame,
                                  window: int = None) -> Dict[str, Any]:
        """
        Analyze correlations between different market assets.

        Args:
            assets_data: DataFrame with asset prices/returns
            window: Rolling window for correlation calculation

        Returns:
            Dictionary with correlation analysis
        """
        if window is None:
            window = self.market_config['correlation_window']

        # Calculate returns if price data provided
        if 'close' in assets_data.columns:
            returns_data = assets_data['close'].pct_change().dropna()
        else:
            returns_data = assets_data

        # Calculate correlation matrix
        correlation_matrix = returns_data.corr()

        # Calculate rolling correlations
        rolling_corr = {}
        asset_pairs = []

        # Get all unique pairs
        assets = returns_data.columns.tolist()
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                pair = f"{assets[i]}_{assets[j]}"
                asset_pairs.append(pair)

                # Rolling correlation
                rolling_corr[pair] = returns_data[assets[i]].rolling(window=window).corr(returns_data[assets[j]])

        # Find most correlated pairs
        most_correlated = []
        least_correlated = []

        for pair in asset_pairs:
            avg_corr = correlation_matrix.loc[pair.split('_')[0], pair.split('_')[1]]
            most_correlated.append((pair, abs(avg_corr)))
            least_correlated.append((pair, abs(avg_corr)))

        most_correlated.sort(key=lambda x: x[1], reverse=True)
        least_correlated.sort(key=lambda x: x[1])

        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'most_correlated_pairs': most_correlated[:5],
            'least_correlated_pairs': least_correlated[:5],
            'average_correlations': correlation_matrix.mean().to_dict(),
            'rolling_correlations': {pair: corr.dropna().tolist()[-10:] for pair, corr in rolling_corr.items()}
        }

    def get_market_intelligence_report(self, start_date: datetime,
                                     end_date: datetime) -> Dict[str, Any]:
        """
        Generate comprehensive market intelligence report.

        Args:
            start_date: Report start date
            end_date: Report end date

        Returns:
            Comprehensive market intelligence report
        """
        # Query market metrics from database
        query = MetricQuery(
            metric_names=['market.volatility', 'market.trend_strength', 'market.sentiment'],
            start_time=start_date,
            end_time=end_date,
            aggregation='mean',
            interval='1d'
        )

        metrics = self.metrics_db.query_metrics(query)

        if metrics.empty:
            return {'status': 'no_data'}

        report = {
            'status': 'success',
            'period': f"{start_date.date()} to {end_date.date()}",
            'sections': {}
        }

        # Market trend analysis
        volatility_data = metrics[metrics['metric_name'] == 'market.volatility']
        trend_data = metrics[metrics['metric_name'] == 'market.trend_strength']
        sentiment_data = metrics[metrics['metric_name'] == 'market.sentiment']

        report['sections']['volatility_analysis'] = self._analyze_volatility_trends(volatility_data)
        report['sections']['trend_analysis'] = self._analyze_trend_patterns(trend_data)
        report['sections']['sentiment_analysis'] = self._analyze_sentiment_patterns(sentiment_data)

        # Overall market assessment
        report['market_assessment'] = self._generate_market_assessment(report['sections'])

        return report

    def _calculate_price_metrics(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic price metrics."""
        close_prices = price_data['close']

        return {
            'current_price': float(close_prices.iloc[-1]),
            'price_change_1d': float((close_prices.iloc[-1] / close_prices.iloc[-2] - 1) * 100) if len(close_prices) > 1 else 0,
            'price_change_5d': float((close_prices.iloc[-1] / close_prices.iloc[-5] - 1) * 100) if len(close_prices) > 5 else 0,
            'price_change_20d': float((close_prices.iloc[-1] / close_prices.iloc[-20] - 1) * 100) if len(close_prices) > 20 else 0,
            'high_20d': float(close_prices.tail(20).max()),
            'low_20d': float(close_prices.tail(20).min()),
            'average_volume': float(price_data.get('volume', pd.Series()).tail(20).mean())
        }

    def _calculate_sma(self, price_data: pd.DataFrame, periods: List[int] = None) -> Dict[str, Any]:
        """Calculate Simple Moving Averages."""
        if periods is None:
            periods = [20, 50, 200]

        close_prices = price_data['close']
        smas = {}

        for period in periods:
            if len(close_prices) >= period:
                smas[f'sma_{period}'] = close_prices.rolling(window=period).mean().iloc[-1]
            else:
                smas[f'sma_{period}'] = None

        return smas

    def _calculate_ema(self, price_data: pd.DataFrame, periods: List[int] = None) -> Dict[str, Any]:
        """Calculate Exponential Moving Averages."""
        if periods is None:
            periods = [12, 26, 50]

        close_prices = price_data['close']
        emas = {}

        for period in periods:
            if len(close_prices) >= period:
                emas[f'ema_{period}'] = close_prices.ewm(span=period).mean().iloc[-1]
            else:
                emas[f'ema_{period}'] = None

        return emas

    def _calculate_rsi(self, price_data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """Calculate Relative Strength Index."""
        close_prices = price_data['close']
        if len(close_prices) < period + 1:
            return {'rsi': None}

        # Calculate price changes
        delta = close_prices.diff()

        # Separate gains and losses
        gains = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        losses = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Calculate RS and RSI
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))

        return {'rsi': float(rsi.iloc[-1])}

    def _calculate_macd(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        close_prices = price_data['close']
        if len(close_prices) < 26:
            return {'macd': None, 'signal': None, 'histogram': None}

        # Calculate EMAs
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()

        # Calculate MACD line
        macd_line = ema_12 - ema_26

        # Calculate signal line (9-period EMA of MACD)
        signal_line = macd_line.ewm(span=9).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        return {
            'macd': float(macd_line.iloc[-1]),
            'signal': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1])
        }

    def _calculate_bollinger(self, price_data: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
        """Calculate Bollinger Bands."""
        close_prices = price_data['close']
        if len(close_prices) < period:
            return {'upper': None, 'middle': None, 'lower': None, 'bandwidth': None}

        # Calculate middle band (SMA)
        middle = close_prices.rolling(window=period).mean()

        # Calculate standard deviation
        std = close_prices.rolling(window=period).std()

        # Calculate bands
        upper = middle + (std * 2)
        lower = middle - (std * 2)

        # Calculate bandwidth
        bandwidth = (upper - lower) / middle

        return {
            'upper': float(upper.iloc[-1]),
            'middle': float(middle.iloc[-1]),
            'lower': float(lower.iloc[-1]),
            'bandwidth': float(bandwidth.iloc[-1])
        }

    def _analyze_overall_trend(self, price_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall market trend."""
        close_prices = price_data['close']

        # Price trend
        recent_prices = close_prices.tail(20)
        price_trend = 'bullish' if recent_prices.iloc[-1] > recent_prices.iloc[0] else 'bearish'

        # Moving average trend
        trend_score = 0
        if 'sma' in indicators:
            sma_20 = indicators['sma'].get('sma_20')
            sma_50 = indicators['sma'].get('sma_50')
            if sma_20 and sma_50:
                if sma_20 > sma_50:
                    trend_score += 1  # Bullish
                else:
                    trend_score -= 1  # Bearish

        # RSI analysis
        rsi_signal = 'neutral'
        if 'rsi' in indicators:
            rsi_value = indicators['rsi'].get('rsi')
            if rsi_value:
                if rsi_value > 70:
                    rsi_signal = 'overbought'
                elif rsi_value < 30:
                    rsi_signal = 'oversold'

        return {
            'price_trend': price_trend,
            'trend_strength': abs(trend_score),
            'rsi_signal': rsi_signal,
            'overall_sentiment': 'bullish' if trend_score > 0 else 'bearish' if trend_score < 0 else 'neutral'
        }

    def _analyze_text_sentiment(self, text_data: List[str]) -> Dict[str, Any]:
        """Analyze sentiment from text data."""
        # Simple sentiment analysis based on keyword matching
        positive_words = ['bullish', 'up', 'gain', 'profit', 'strong', 'positive', 'good', 'excellent']
        negative_words = ['bearish', 'down', 'loss', 'decline', 'weak', 'negative', 'bad', 'terrible']

        total_texts = len(text_data)
        positive_count = 0
        negative_count = 0

        for text in text_data:
            text_lower = text.lower()
            pos_score = sum(1 for word in positive_words if word in text_lower)
            neg_score = sum(1 for word in negative_words if word in text_lower)

            if pos_score > neg_score:
                positive_count += 1
            elif neg_score > pos_score:
                negative_count += 1

        sentiment_score = (positive_count - negative_count) / total_texts if total_texts > 0 else 0

        return {
            'total_texts': total_texts,
            'positive_texts': positive_count,
            'negative_texts': negative_count,
            'sentiment_score': sentiment_score,
            'sentiment_label': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'
        }

    def _analyze_news_sentiment(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment from news data."""
        if 'text' not in news_data.columns:
            return {'status': 'error', 'message': 'No text column in news data'}

        texts = news_data['text'].tolist()
        return self._analyze_text_sentiment(texts)

    def _combine_sentiment_scores(self, sentiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple sentiment scores."""
        scores = []

        if 'text_sentiment' in sentiment_results:
            scores.append(sentiment_results['text_sentiment']['sentiment_score'])

        if 'news_sentiment' in sentiment_results:
            scores.append(sentiment_results['news_sentiment']['sentiment_score'])

        if scores:
            avg_score = np.mean(scores)
            return {
                'combined_score': avg_score,
                'sentiment_label': 'positive' if avg_score > 0.1 else 'negative' if avg_score < -0.1 else 'neutral'
            }
        else:
            return {'combined_score': 0, 'sentiment_label': 'neutral'}

    def _calculate_regime_confidence(self, volatility_series: pd.Series, current_volatility: float) -> float:
        """Calculate confidence in regime detection."""
        if volatility_series.empty:
            return 0.0

        # Calculate how many standard deviations current volatility is from mean
        mean_vol = volatility_series.mean()
        std_vol = volatility_series.std()

        if std_vol == 0:
            return 1.0  # Perfect confidence if no variation

        z_score = abs(current_volatility - mean_vol) / std_vol
        confidence = min(z_score / 2, 1.0)  # Scale to 0-1 range

        return float(confidence)

    def _generate_ma_signals(self, price_data: pd.DataFrame, sma_data: Dict[str, Any]) -> str:
        """Generate signals based on moving averages."""
        current_price = price_data['close'].iloc[-1]

        sma_20 = sma_data.get('sma_20')
        sma_50 = sma_data.get('sma_50')

        if sma_20 and sma_50:
            if current_price > sma_20 > sma_50:
                return 'strong_buy'
            elif current_price > sma_20 and sma_20 < sma_50:
                return 'buy'
            elif current_price < sma_20 < sma_50:
                return 'strong_sell'
            elif current_price < sma_20 and sma_20 > sma_50:
                return 'sell'

        return 'hold'

    def _generate_rsi_signals(self, rsi_data: Dict[str, Any]) -> str:
        """Generate signals based on RSI."""
        rsi = rsi_data.get('rsi')

        if rsi:
            if rsi > 70:
                return 'overbought_sell'
            elif rsi < 30:
                return 'oversold_buy'

        return 'neutral'

    def _generate_macd_signals(self, macd_data: Dict[str, Any]) -> str:
        """Generate signals based on MACD."""
        macd = macd_data.get('macd')
        signal = macd_data.get('signal')
        histogram = macd_data.get('histogram')

        if macd and signal and histogram:
            if macd > signal and histogram > 0:
                return 'buy'
            elif macd < signal and histogram < 0:
                return 'sell'

        return 'neutral'

    def _generate_bollinger_signals(self, price_data: pd.DataFrame, bollinger_data: Dict[str, Any]) -> str:
        """Generate signals based on Bollinger Bands."""
        current_price = price_data['close'].iloc[-1]
        upper = bollinger_data.get('upper')
        lower = bollinger_data.get('lower')

        if upper and lower:
            if current_price > upper:
                return 'overbought_sell'
            elif current_price < lower:
                return 'oversold_buy'

        return 'neutral'

    def _combine_trading_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple trading signals for overall recommendation."""
        signal_scores = {
            'strong_buy': 2, 'buy': 1, 'hold': 0, 'sell': -1, 'strong_sell': -2,
            'oversold_buy': 1, 'overbought_sell': -1, 'neutral': 0
        }

        total_score = 0
        signal_count = 0

        for signal_type, signal_value in signals.items():
            if signal_type != 'overall_signal' and signal_value in signal_scores:
                total_score += signal_scores[signal_value]
                signal_count += 1

        if signal_count == 0:
            return {'recommendation': 'hold', 'confidence': 0}

        avg_score = total_score / signal_count

        if avg_score >= 1.5:
            recommendation = 'strong_buy'
        elif avg_score >= 0.5:
            recommendation = 'buy'
        elif avg_score <= -1.5:
            recommendation = 'strong_sell'
        elif avg_score <= -0.5:
            recommendation = 'sell'
        else:
            recommendation = 'hold'

        confidence = min(abs(avg_score) / 2, 1.0)

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'signal_sources': signal_count
        }

    def _analyze_volatility_trends(self, volatility_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility trends."""
        if volatility_data.empty:
            return {'status': 'no_data'}

        values = volatility_data['value'].values

        return {
            'current_volatility': float(values[-1]) if len(values) > 0 else None,
            'average_volatility': float(np.mean(values)),
            'volatility_trend': 'increasing' if values[-1] > np.mean(values) else 'decreasing',
            'volatility_range': {
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        }

    def _analyze_trend_patterns(self, trend_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend patterns."""
        if trend_data.empty:
            return {'status': 'no_data'}

        values = trend_data['value'].values

        return {
            'current_trend': float(values[-1]) if len(values) > 0 else None,
            'average_trend': float(np.mean(values)),
            'trend_consistency': float(np.std(values)),
            'trend_direction': 'upward' if np.mean(values) > 0 else 'downward'
        }

    def _analyze_sentiment_patterns(self, sentiment_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment patterns."""
        if sentiment_data.empty:
            return {'status': 'no_data'}

        values = sentiment_data['value'].values

        return {
            'current_sentiment': float(values[-1]) if len(values) > 0 else None,
            'average_sentiment': float(np.mean(values)),
            'sentiment_volatility': float(np.std(values)),
            'sentiment_label': 'positive' if np.mean(values) > 0.1 else 'negative' if np.mean(values) < -0.1 else 'neutral'
        }

    def _generate_market_assessment(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall market assessment."""
        assessment = {'overall_sentiment': 'neutral', 'risk_level': 'moderate'}

        # Combine volatility and sentiment
        vol_section = sections.get('volatility_analysis', {})
        sent_section = sections.get('sentiment_analysis', {})

        current_vol = vol_section.get('current_volatility', 0.02)
        current_sent = sent_section.get('current_sentiment', 0)

        # Determine overall sentiment
        if current_sent > 0.2 and current_vol < 0.025:
            assessment['overall_sentiment'] = 'bullish'
        elif current_sent < -0.2 or current_vol > 0.04:
            assessment['overall_sentiment'] = 'bearish'

        # Determine risk level
        if current_vol > 0.04:
            assessment['risk_level'] = 'high'
        elif current_vol < 0.015:
            assessment['risk_level'] = 'low'

        assessment['confidence'] = min(abs(current_sent) + (0.02 - current_vol), 1.0)

        return assessment