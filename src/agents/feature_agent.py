import pandas as pd
from typing import Dict, Any, Optional
from .seasonality_agent import SeasonalityAgent

class FeatureAgent:
    """
    A simple agent for generating features from raw time-series data.
    """
    def __init__(self):
        self.seasonality_agent = SeasonalityAgent()

    def generate_features(self, 
                          raw_data: Dict[str, pd.DataFrame], 
                          macro_data: Optional[Dict[str, Any]] = None,
                          regimes: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generates a set of basic features for each symbol.

        Args:
            raw_data (Dict[str, pd.DataFrame]): A dictionary of raw dataframes for each symbol.
            macro_data (Optional[Dict[str, Any]]): Macro economic data.
            regimes (Optional[Dict[str, str]]): Detected market regimes.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of dataframes with features.
        """
        features = {}
        
        # Process macro features once if available
        macro_df = None
        if macro_data and 'processed_features' in macro_data:
            macro_df = macro_data['processed_features']
            # Ensure index is datetime
            if not isinstance(macro_df.index, pd.DatetimeIndex):
                macro_df.index = pd.to_datetime(macro_df.index)

        for symbol, df in raw_data.items():
            df_feat = df.copy()
            
            # Ensure index is datetime
            if not isinstance(df_feat.index, pd.DatetimeIndex):
                df_feat.index = pd.to_datetime(df_feat.index)

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
            
            # Seasonality Features
            # We can compute them for each date
            # For efficiency, we can vectorize some of this
            df_feat['month'] = df_feat.index.month
            df_feat['day_of_week'] = df_feat.index.dayofweek
            df_feat['quarter'] = df_feat.index.quarter
            df_feat['is_month_end'] = df_feat.index.is_month_end.astype(int)
            df_feat['is_quarter_end'] = df_feat.index.is_quarter_end.astype(int)
            
            # Macro Features Integration
            if macro_df is not None:
                # Merge macro features on date index
                # Use asof merge or join
                df_feat = df_feat.join(macro_df, how='left')
                # Forward fill macro data
                df_feat.ffill(inplace=True)
            
            # Regime Features (Static for the current run, or historical if we had historical regimes)
            # Here we just add current regime as a categorical feature (if supported) or ignore
            # For now, we don't add regime as a column unless we have historical regime data
            
            # Add target variable 'y' as close price
            df_feat['y'] = df_feat['close']
            
            # Drop NaNs created by rolling windows
            df_feat.dropna(inplace=True)
            
            features[symbol] = df_feat
            print(f"[OK] Generated features for {symbol}.")
            
        return features
