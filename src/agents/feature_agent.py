import pandas as pd
from typing import Dict

class FeatureAgent:
    """
    A simple agent for generating features from raw time-series data.
    """
    def generate_features(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generates a set of basic features for each symbol.

        Args:
            raw_data (Dict[str, pd.DataFrame]): A dictionary of raw dataframes for each symbol.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of dataframes with features.
        """
        features = {}
        for symbol, df in raw_data.items():
            df_feat = df.copy()
            
            # Moving Averages
            df_feat['sma_20'] = df_feat['close'].rolling(window=20).mean()
            df_feat['sma_50'] = df_feat['close'].rolling(window=50).mean()
            
            # RSI
            delta = df_feat['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_feat['rsi'] = 100 - (100 / (1 + rs))
            
            # Sentiment features (if available)
            if 'sentiment_compound' in df_feat.columns:
                df_feat['sentiment_ma_7'] = df_feat['sentiment_compound'].rolling(window=7).mean()
                df_feat['sentiment_ma_14'] = df_feat['sentiment_compound'].rolling(window=14).mean()
                df_feat['sentiment_volatility'] = df_feat['sentiment_compound'].rolling(window=14).std()
            
            # Add target variable 'y' as close price
            df_feat['y'] = df_feat['close']
            
            # Drop NaNs created by rolling windows
            df_feat.dropna(inplace=True)
            
            features[symbol] = df_feat
            print(f"[OK] Generated features for {symbol}.")
            
        return features
