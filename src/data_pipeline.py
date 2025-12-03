"""
Data Pipeline for IB Forecast System

Integrates Alpha Vantage data fetching with ML model training and prediction.
Provides unified interface for data acquisition, preprocessing, and model feeding.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from alpha_vantage_client import AlphaVantageClient, get_stock_data, get_technical_features

# Import ML models
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.lstm_forecaster import LSTMForecaster
from src.ensemble_methods import EnsembleForecaster

from src.drift_detection import DriftDetector

logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Unified data pipeline for fetching, preprocessing, and feeding data to ML models.
    """

    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """
        Initialize data pipeline.

        Args:
            alpha_vantage_key: API key for Alpha Vantage
        """
        self.av_client = AlphaVantageClient(alpha_vantage_key)
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.feature_columns = None

        logger.info("Data pipeline initialized")

    def fetch_stock_data(self, symbol: str, period: str = '2y', interval: str = 'daily',
                        include_technical: bool = True) -> pd.DataFrame:
        """
        Fetch stock data from Alpha Vantage.

        Args:
            symbol: Stock symbol
            period: Time period ('1y', '2y', '5y', 'max')
            interval: Data interval ('daily', 'weekly', 'monthly', '5min', etc.)
            include_technical: Whether to include technical indicators

        Returns:
            DataFrame with OHLCV and technical indicators
        """
        logger.info(f"Fetching {period} {interval} data for {symbol}")

        # Get price data
        price_data = get_stock_data(symbol, period, interval)

        if price_data.empty:
            logger.error(f"No data fetched for {symbol}")
            return pd.DataFrame()

        # Add technical indicators if requested
        if include_technical:
            try:
                tech_features = get_technical_features(symbol, ['SMA', 'EMA', 'RSI', 'MACD'])
                # Merge on date index
                combined_data = price_data.join(tech_features, how='left')
                # Forward fill missing technical indicator values
                combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
            except Exception as e:
                logger.warning(f"Could not fetch technical indicators: {e}")
                combined_data = price_data
        else:
            combined_data = price_data

        # Add basic derived features
        combined_data['returns'] = combined_data['close'].pct_change(fill_method=None)
        combined_data['log_returns'] = np.log(combined_data['close'] / combined_data['close'].shift(1))
        combined_data['volatility'] = combined_data['returns'].rolling(20).std()
        combined_data['volume_ma'] = combined_data['volume'].rolling(20).mean()

        # Fill NaN values
        combined_data = combined_data.fillna(method='bfill').fillna(0)

        logger.info(f"Fetched {len(combined_data)} data points with {len(combined_data.columns)} features")
        return combined_data

    def prepare_ml_data(self, data: pd.DataFrame, target_column: str = 'close',
                       sequence_length: int = 60, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for ML model training.

        Args:
            data: Raw data DataFrame
            target_column: Column to predict
            sequence_length: Length of input sequences
            test_size: Proportion of data for testing

        Returns:
            X_train, X_test, y_train, y_test
        """
        if data.empty:
            raise ValueError("No data provided")

        # Select features (exclude target from features if predicting price)
        if target_column in data.columns:
            feature_cols = [col for col in data.columns if col != target_column]
            self.feature_columns = feature_cols
        else:
            feature_cols = data.columns.tolist()
            self.feature_columns = feature_cols

        X = data[feature_cols].values
        y = data[target_column].values.reshape(-1, 1)

        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        # Create sequences for time series models
        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-sequence_length:i])
            y_seq.append(y_scaled[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # Split into train/test
        split_idx = int(len(X_seq) * (1 - test_size))
        X_train = X_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_train = y_seq[:split_idx]
        y_test = y_seq[split_idx:]

        logger.info(f"Prepared ML data: {X_train.shape[0]} train, {X_test.shape[0]} test sequences")
        return X_train, X_test, y_train, y_test

    def train_cnn_lstm(self, symbol: str, period: str = '2y', sequence_length: int = 60,
                       epochs: int = 50, **kwargs) -> Tuple[LSTMForecaster, Dict]:
        """
        Train CNN-LSTM model on Alpha Vantage data.

        Args:
            symbol: Stock symbol
            period: Time period
            sequence_length: Sequence length for model
            epochs: Training epochs
            **kwargs: Additional training parameters

        Returns:
            Trained model and training results
        """
        logger.info(f"Training CNN-LSTM on {symbol} data")

        # Fetch data
        data = self.fetch_stock_data(symbol, period, include_technical=True)
        if data.empty:
            raise ValueError(f"Could not fetch data for {symbol}")

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_ml_data(
            data, sequence_length=sequence_length, test_size=0.2
        )

        # Create and train model
        model = LSTMForecaster(sequence_length=sequence_length, n_features=X_train.shape[2])
        training_results = model.train(X_train, y_train, epochs=epochs, **kwargs)

        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test)

        results = {
            'model': model,
            'training_results': training_results,
            'test_metrics': test_metrics,
            'symbol': symbol,
            'data_points': len(data)
        }

        logger.info(f"CNN-LSTM training completed for {symbol}")
        return model, results

    def train_ensemble(self, symbol: str, period: str = '2y', **kwargs) -> Tuple[EnsembleForecaster, Dict]:
        """
        Train ensemble model on Alpha Vantage data.

        Args:
            symbol: Stock symbol
            period: Time period
            **kwargs: Additional parameters

        Returns:
            Trained ensemble and results
        """
        logger.info(f"Training ensemble on {symbol} data")

        # Fetch data
        data = self.fetch_stock_data(symbol, period, include_technical=True)
        if data.empty:
            raise ValueError(f"Could not fetch data for {symbol}")

        # Prepare data (ensemble might not need sequences)
        X = data.drop('close', axis=1).values
        y = data['close'].values

        # Scale
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, shuffle=False
        )

        # Create and train ensemble
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression

        ensemble = EnsembleForecaster()

        # Create and train base models first
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        lr = LinearRegression()

        # Train individual models
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        lr.fit(X_train, y_train)

        # Add trained models to ensemble
        ensemble.add_base_model('rf', rf)
        ensemble.add_base_model('gb', gb)
        ensemble.add_base_model('lr', lr)

        ensemble.train_ensemble(X_train, y_train)

        # Evaluate
        predictions = ensemble.predict(X_test)
        mae = np.mean(np.abs(y_test - predictions))
        rmse = np.sqrt(np.mean((y_test - predictions)**2))

        results = {
            'ensemble': ensemble,
            'test_mae': mae,
            'test_rmse': rmse,
            'symbol': symbol,
            'data_points': len(data)
        }

        logger.info(f"Ensemble training completed for {symbol}")
        return ensemble, results

    def setup_drift_detection(self, symbol: str, period: str = '2y') -> DriftDetector:
        """
        Setup drift detection on historical data.

        Args:
            symbol: Stock symbol
            period: Time period

        Returns:
            Configured drift detector
        """
        logger.info(f"Setting up drift detection for {symbol}")

        # Fetch data
        data = self.fetch_stock_data(symbol, period, include_technical=False)
        if data.empty:
            raise ValueError(f"Could not fetch data for {symbol}")

        # Use returns for drift detection
        returns = data['returns'].dropna().values

        # Setup drift detector
        detector = DriftDetector()
        detector.fit_baseline(returns)

        logger.info(f"Drift detection setup completed for {symbol}")
        return detector

    def get_news_sentiment(self, symbols: List[str], days_back: int = 7) -> pd.DataFrame:
        """
        Fetch recent news and sentiment for symbols.

        Args:
            symbols: List of stock symbols
            days_back: Number of days to look back

        Returns:
            DataFrame with news and sentiment
        """
        time_from = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%dT%H%M')
        news = self.av_client.get_news_sentiment(symbols, time_from=time_from, limit=100)

        if not news.empty:
            # Extract ticker sentiment data
            sentiment_data = []
            for idx, row in news.iterrows():
                ticker_sentiments = row.get('ticker_sentiment', [])
                if isinstance(ticker_sentiments, list):
                    for ts in ticker_sentiments:
                        if isinstance(ts, dict) and 'ticker' in ts:
                            sentiment_data.append({
                                'ticker': ts['ticker'],
                                'sentiment_score': ts.get('sentiment_score', 0),
                                'relevance_score': ts.get('relevance_score', 0),
                                'date': idx
                            })

            if sentiment_data:
                sentiment_df = pd.DataFrame(sentiment_data)
                # Calculate average sentiment per symbol
                sentiment_summary = sentiment_df.groupby('ticker')['sentiment_score'].agg(['mean', 'count']).round(3)
                logger.info(f"Fetched news sentiment for {len(sentiment_data)} ticker mentions")
                return sentiment_summary
            else:
                logger.warning("No ticker sentiment data found in news")
                return pd.DataFrame()
        else:
            logger.warning("No news data fetched")
            return pd.DataFrame()

    def forecast_with_models(self, symbol: str, forecast_steps: int = 30) -> Dict[str, Any]:
        """
        Generate forecasts using all trained models.

        Args:
            symbol: Stock symbol
            forecast_steps: Number of steps to forecast

        Returns:
            Dictionary with forecasts from all models
        """
        logger.info(f"Generating {forecast_steps}-step forecast for {symbol}")

        # Fetch recent data
        data = self.fetch_stock_data(symbol, '1y', include_technical=True)
        if data.empty:
            raise ValueError(f"Could not fetch data for {symbol}")

        # Prepare data for CNN-LSTM
        X = data[self.feature_columns].values[-60:]  # Last 60 points
        X_scaled = self.scaler_X.transform(X)
        X_seq = X_scaled.reshape(1, 60, -1)

        forecasts = {}

        # CNN-LSTM forecast
        try:
            cnn_model = LSTMForecaster(sequence_length=60, n_features=X_seq.shape[2])
            # Load trained model if exists
            if os.path.exists('best_cnn_lstm_model.h5'):
                cnn_model.model.load_weights('best_cnn_lstm_model.h5')
                cnn_forecast = cnn_model.forecast(forecast_steps, X_seq)
                forecasts['cnn_lstm'] = cnn_forecast
                logger.info("CNN-LSTM forecast generated")
            else:
                logger.warning("No trained CNN-LSTM model found")
        except Exception as e:
            logger.error(f"CNN-LSTM forecast failed: {e}")

        # Ensemble forecast (simplified)
        try:
            ensemble = EnsembleForecaster()
            # This would need proper ensemble training
            recent_features = X_scaled[-1:]  # Last point
            ensemble_forecast = []
            for _ in range(forecast_steps):
                pred = ensemble.predict(recent_features)[0]
                ensemble_forecast.append(pred)
                # Update features (simplified)
                recent_features = np.roll(recent_features, -1, axis=1)
                recent_features[0, -1] = pred
            forecasts['ensemble'] = np.array(ensemble_forecast)
            logger.info("Ensemble forecast generated")
        except Exception as e:
            logger.error(f"Ensemble forecast failed: {e}")

        # Get current price for reference
        current_price = data['close'].iloc[-1]
        forecasts['current_price'] = current_price
        forecasts['symbol'] = symbol

        return forecasts


def demonstrate_data_pipeline():
    """Demonstrate the data pipeline with Alpha Vantage integration."""
    print("🔄 IB Forecast Data Pipeline Demonstration")
    print("=" * 80)

    # Note: User needs to set ALPHA_VANTAGE_API_KEY in .env file
    try:
        pipeline = DataPipeline()

        # Example symbol
        symbol = 'AAPL'

        print(f"📊 Fetching data for {symbol}...")

        # Fetch data
        data = pipeline.fetch_stock_data(symbol, '1y', include_technical=True)
        print(f"✅ Fetched {len(data)} data points with {len(data.columns)} features")
        print(f"📈 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

        # Show data sample
        print("\n📋 Data Sample:")
        print(data.head())

        # Get news sentiment
        print(f"\n📰 Fetching news sentiment for {symbol}...")
        news_sentiment = pipeline.get_news_sentiment([symbol], days_back=7)
        if not news_sentiment.empty:
            print("News sentiment summary:")
            print(news_sentiment)

        # Prepare ML data
        print("\n🤖 Preparing data for ML models...")
        X_train, X_test, y_train, y_test = pipeline.prepare_ml_data(data, sequence_length=30)
        print(f"✅ Prepared {X_train.shape[0]} training sequences, {X_test.shape[0]} test sequences")

        print("\n✅ Data Pipeline Demonstration Complete!")
        print("📁 Next steps:")
        print("  1. Set ALPHA_VANTAGE_API_KEY in .env file")
        print("  2. Run training: pipeline.train_cnn_lstm('AAPL')")
        print("  3. Generate forecasts: pipeline.forecast_with_models('AAPL')")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("💡 Make sure ALPHA_VANTAGE_API_KEY is set in .env file")


if __name__ == "__main__":
    demonstrate_data_pipeline()
