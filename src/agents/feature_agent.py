import pandas as pd
from typing import Dict, Any, Optional, Union
from .seasonality_agent import SeasonalityAgent
from ..utils.time_machine import TimeMachine

class FeatureAgent:
    """
    A simple agent for generating features from raw time-series data.
    """
    def __init__(self):
        self.seasonality_agent = SeasonalityAgent()

    def generate_features(self, 
                          raw_data: Dict[str, pd.DataFrame], 
                          macro_data: Optional[Dict[str, Any]] = None,
                          regimes: Optional[Dict[str, str]] = None,
                          historical_regimes: Optional[Dict[str, Dict[str, str]]] = None,
                          cutoff_date: Optional[Union[str, pd.Timestamp]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generates a set of basic features for each symbol.

        Args:
            raw_data (Dict[str, pd.DataFrame]): A dictionary of raw dataframes for each symbol.
            macro_data (Optional[Dict[str, Any]]): Macro economic data.
            regimes (Optional[Dict[str, str]]): Detected market regimes.
            historical_regimes (Optional[Dict[str, Dict[str, str]]]): Historical regimes for feature engineering.
            cutoff_date (Optional[Union[str, pd.Timestamp]]): Strict cutoff date for data access.

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
            
            # Apply cutoff to macro data
            if cutoff_date:
                tm = TimeMachine(macro_df, date_col='index') # Assuming index is date
                macro_df = tm.get_data_as_of(cutoff_date)

        for symbol, df in raw_data.items():
            # Apply TimeMachine if cutoff_date is provided
            if cutoff_date:
                tm = TimeMachine(df)
                df_feat = tm.get_data_as_of(cutoff_date)
                print(f"⏳ Applied time cutoff {cutoff_date} for {symbol}. Rows: {len(df)} -> {len(df_feat)}")
            else:
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
            
            # Regime Features
            if historical_regimes:
                for regime_name, regime_dict in historical_regimes.items():
                    try:
                        # Convert dict to Series
                        regime_series = pd.Series(regime_dict)
                        regime_series.index = pd.to_datetime(regime_series.index)
                        regime_series.name = regime_name
                        
                        # Join with df_feat
                        df_feat = df_feat.join(regime_series, how='left')
                        
                        # Forward fill missing values (regimes persist until changed)
                        df_feat[regime_name] = df_feat[regime_name].ffill().fillna('unknown')
                        
                        # One-Hot Encode for NeuralForecast compatibility
                        dummies = pd.get_dummies(df_feat[regime_name], prefix=regime_name)
                        # Ensure dummies are numeric (0/1)
                        dummies = dummies.astype(int)
                        
                        df_feat = pd.concat([df_feat, dummies], axis=1)
                        # Drop original string column
                        df_feat.drop(columns=[regime_name], inplace=True)
                    except Exception as e:
                        print(f"Error adding regime feature {regime_name}: {e}")
            
            # Add target variable 'y' as close price
            df_feat['y'] = df_feat['close']
            
            # Drop NaNs created by rolling windows
            df_feat.dropna(inplace=True)
            
            features[symbol] = df_feat
            print(f"[OK] Generated features for {symbol}.")
            
        # Add Cross-Asset Features
        features = self._add_cross_asset_features(features)

        return features

    def _add_cross_asset_features(self, features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Adds cross-asset features like market correlation and relative strength.
        """
        if len(features) < 2:
            return features
            
        # Create a market index from available symbols (equal weighted)
        price_df = pd.DataFrame()
        
        for symbol, df in features.items():
            if 'close' in df.columns:
                price_df[symbol] = df['close']
        
        if price_df.empty:
            return features
            
        # Calculate Market Index (mean of normalized prices or just returns)
        # Using returns is safer for different price scales
        returns_df = price_df.pct_change()
        market_returns = returns_df.mean(axis=1)
        
        for symbol, df in features.items():
            if symbol not in returns_df.columns:
                continue
                
            # 1. Market Correlation (20-day rolling)
            rolling_corr = returns_df[symbol].rolling(window=20).corr(market_returns)
            df['market_correlation_20'] = rolling_corr
            
            # 2. Relative Strength (Symbol Return - Market Return)
            rel_strength = returns_df[symbol] - market_returns
            df['relative_strength_1d'] = rel_strength
            df['relative_strength_20'] = rel_strength.rolling(window=20).sum()
            
            # 3. Beta (Covariance / Variance) - simplified rolling beta
            rolling_cov = returns_df[symbol].rolling(window=60).cov(market_returns)
            rolling_var = market_returns.rolling(window=60).var()
            df['beta_60'] = rolling_cov / rolling_var
            
            # Fill NaNs generated by rolling windows
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            
            features[symbol] = df
            
        return features
