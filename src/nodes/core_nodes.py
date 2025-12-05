from src.core.state import PipelineGraphState
import pandas as pd
import numpy as np
import logging
import time
import os
import sys
import ta

# Add root to path for models import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.model_zoo import DataSpec
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

# Import Services
from src.services.training_service import GPUTrainingService
from src.services.inference_service import InferenceService
from src.services.model_registry_service import ModelRegistryService
from src.alpha_vantage_client import AlphaVantageClient
from src.agents.feature_engineer_agent import FeatureEngineerAgent
from src.services.database_service import DatabaseService

logger = logging.getLogger(__name__)

def data_ingestion_node(state: PipelineGraphState) -> PipelineGraphState:
    """
    Ingests data for the requested symbols.
    """
    logger.info("--- Node: Data Ingestion ---")
    symbols = state['symbols']
    data_map = {}
    
    try:
        client = AlphaVantageClient()
    except ValueError as e:
        logger.error(f"Failed to initialize AlphaVantageClient: {e}")
        # For testing purposes, if API key is missing, we might want to generate synthetic data
        # But for now, let's just log error and return empty data
        state['data'] = {}
        return state
    
    for symbol in symbols:
        try:
            logger.info(f"Fetching data for {symbol}...")
            # Use full outputsize for production
            df = client.get_daily_data(symbol, outputsize='full')
            
            # Apply Backtest Cutoff if present
            cutoff_date = state.get('cutoff_date')
            if cutoff_date and df is not None and not df.empty:
                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # Filter data up to cutoff (inclusive)
                # We assume cutoff_date is the "current date" of the simulation.
                # The model should only see data up to this date.
                df = df[df.index <= pd.Timestamp(cutoff_date)]
                logger.info(f"Filtered data for {symbol} up to {cutoff_date}. Rows: {len(df)}")

            if df is not None and not df.empty:
                data_map[symbol] = df
                logger.info(f"Fetched {len(df)} rows for {symbol}")
            else:
                logger.warning(f"No data found for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            
    state['data'] = data_map
    return state

def feature_engineering_node(state: PipelineGraphState) -> PipelineGraphState:
    """
    Generates features for the ingested data using FeatureEngineerAgent.
    """
    logger.info("--- Node: Feature Engineering ---")
    data_map = state.get('data', {})
    features_map = {}
    
    # Initialize Agent
    try:
        feature_agent = FeatureEngineerAgent()
    except Exception as e:
        logger.error(f"Failed to initialize FeatureEngineerAgent: {e}")
        # Fallback or re-raise? Let's try to continue but it will likely fail inside the loop if agent is None
        feature_agent = None

    for symbol, df in data_map.items():
        try:
            logger.info(f"Generating features for {symbol}...")
            
            if feature_agent:
                # Use the agent to engineer features
                # We pass 'basic' and 'spectral' to enable GPU features
                df_features = feature_agent.engineer_features(
                    symbol=symbol, 
                    data=df, 
                    feature_sets=['basic', 'spectral']
                )
                features_map[symbol] = df_features
            else:
                 # Fallback to basic manual features if agent init failed
                logger.warning(f"FeatureEngineerAgent not available, using basic fallback for {symbol}")
                df_features = df.copy()
                if 'close' in df_features.columns:
                    df_features['rsi'] = ta.momentum.rsi(df_features['close'], window=14)
                    features_map[symbol] = df_features
                else:
                    features_map[symbol] = df

        except Exception as e:
            logger.error(f"Error generating features for {symbol}: {e}")
            features_map[symbol] = df 
            
    state['features'] = features_map
    return state

def forecasting_node(state: PipelineGraphState) -> PipelineGraphState:
    """
    Generates forecasts for each symbol using optimized models from HPO.
    Computes performance on validation set.
    """
    logger.info("--- Node: Forecasting ---")
    
    symbols = state['symbols']
    features = state['features']
    best_models = state.get('best_models', {})
    config = state.get('config', {})
    run_type = state.get('run_type', 'DAILY')
    
    # Validate features
    if not features:
        logger.error("No features available for forecasting")
        # state['errors'].append("No features available for forecasting") # errors not in TypedDict yet
        return state
    
    horizon = 3
    forecasts = {}
    
    # Initialize Services
    training_service = GPUTrainingService()
    inference_service = InferenceService()
    model_registry = ModelRegistryService()
    db_service = DatabaseService()

    for symbol in symbols:
        try:
            logger.info(f"Generating forecast for {symbol}...")
            
            if symbol not in features:
                logger.error(f"No features found for {symbol}")
                continue
                
            data = features[symbol]
            
            # Convert dict back to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame.from_dict(data, orient='index')
                data.index = pd.to_datetime(data.index)
            
            # Validate data
            required_columns = ['close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns for {symbol}: {missing_columns}")
                continue
            if data.empty:
                logger.error(f"Empty data for {symbol}")
                continue
            
            # Prepare df for services
            df = data.copy()
            
            # If 'y' already exists (from FeatureAgent), drop it to avoid conflict when renaming 'close'
            if 'y' in df.columns:
                df = df.drop(columns=['y'])
                
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'ds', 'close': 'y'})
            else:
                df['ds'] = df.index
                df = df.rename(columns={'close': 'y'})
            
            df['unique_id'] = symbol
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Identify exogenous columns
            exog_cols = [col for col in df.columns if col not in ['ds', 'y', 'unique_id']]
            cols_to_keep = ['unique_id', 'ds', 'y'] + exog_cols
            df = df[cols_to_keep]
            
            train_df = df.iloc[:-horizon]
            val_df = df.iloc[-horizon:]
            full_df = df
            
            symbol_forecasts = {}
            
            # Determine which models to run
            models_to_run = []
            
            if symbol in best_models and best_models[symbol]:
                # Use HPO results
                # best_models[symbol] is expected to be a dict of {model_family: result_obj}
                for family, res in best_models[symbol].items():
                    # Handle both object and dict
                    if isinstance(res, dict):
                        model_id = res.get('best_model_id')
                        mape_val = res.get('best_val_mape', 0.0)
                    else:
                        model_id = getattr(res, 'best_model_id', None)
                        mape_val = getattr(res, 'best_val_mape', 0.0)
                    
                    if model_id:
                        models_to_run.append({
                            'family': family,
                            'model_id': model_id,
                            'mape': mape_val
                        })
            else:
                # Fallback: Train default models
                logger.info(f"No HPO results for {symbol}, training default models.")
                priority_order = ["BaselineLinear", "NLinear", "NHITS", "AutoDLinear"]
                if run_type == 'WEEKEND_HPO':
                    priority_order.extend(["TFT", "AutoNHITS", "AutoTFT", "AutoNBEATS", "graph_stgcnn"])
                
                for family in priority_order:
                    # Create DataSpec for training
                    data_spec = DataSpec(
                        job_id=f"forecast_{symbol}_{int(time.time())}",
                        symbol_scope=symbol,
                        train_df=train_df,
                        val_df=val_df,
                        feature_cols=['y'] + exog_cols,
                        target_col='y',
                        horizon=horizon,
                        exog_cols=exog_cols
                    )
                    
                    res = training_service.train_model(
                        symbol=symbol,
                        model_type=family,
                        data=data_spec,
                        hyperparams={'horizon': horizon}
                    )
                    
                    if res['status'] == 'success':
                        metrics = res.get('metrics', {})
                        models_to_run.append({
                            'family': family,
                            'model_id': res['model_id'],
                            'mape': metrics.get('mae', 0.0) # Using MAE as proxy if MAPE missing
                        })
                    else:
                        logger.error(f"Failed to train default {family} for {symbol}: {res.get('error')}")

            # Generate Forecasts
            for model_info in models_to_run:
                family = model_info['family']
                model_id = model_info['model_id']
                mape_val = model_info['mape']
                
                pred_res = inference_service.predict(
                    symbol=symbol,
                    model_id=model_id,
                    model_type=family,
                    data=full_df,
                    horizon=horizon
                )
                
                pred_future = None
                if pred_res['status'] == 'success':
                    preds = pred_res['predictions']
                    
                    # Handle list of dicts (JSON serialization from service)
                    if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
                        preds = pd.DataFrame(preds)

                    if isinstance(preds, pd.DataFrame):
                        # Find prediction column
                        pred_cols = [c for c in preds.columns if c not in ['ds', 'unique_id', 'y']]
                        if pred_cols:
                            pred_future = preds[pred_cols[0]].values
                    else:
                        pred_future = np.array(preds)

                    
                    # Ensure pred_future length matches horizon
                    if pred_future is not None:
                        if len(pred_future) > horizon:
                            pred_future = pred_future[:horizon]
                        elif len(pred_future) < horizon:
                            last_val = pred_future[-1]
                            padding = np.full(horizon - len(pred_future), last_val)
                            pred_future = np.concatenate([pred_future, padding])
                else:
                    logger.warning(f"Inference failed for {family}: {pred_res.get('message')}")
                    # Fallback to simple linear regression if inference fails
                    try:
                        logger.info(f"Using sklearn LinearRegression fallback for {symbol} due to inference failure")
                        X_train = np.arange(len(train_df)).reshape(-1, 1)
                        y_train = train_df['y'].values
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        X_future = np.arange(len(full_df), len(full_df) + horizon).reshape(-1, 1)
                        pred_future = model.predict(X_future)
                    except Exception as e:
                        logger.error(f"Fallback failed: {e}")

                if pred_future is not None:
                    ds_future = pd.date_range(start=data.index.max() + pd.Timedelta(days=1), periods=horizon, freq='D')
                    forecast_df = pd.DataFrame({'ds': ds_future, family: pred_future})
                    symbol_forecasts[family] = forecast_df
                    
                    # Save to DB
                    try:
                        first_date = ds_future[0].strftime('%Y-%m-%d')
                        first_val = float(pred_future[0])
                        db_service.save_forecast(symbol, family, first_date, first_val, horizon)
                    except Exception as e:
                        logger.error(f"Failed to save forecast to DB: {e}")
                    
            forecasts[symbol] = symbol_forecasts
            logger.info(f"Generated forecast for {symbol}.")
            
        except Exception as e:
            logger.error(f"Critical error forecasting for {symbol}: {e}")
            continue
    
    state['forecasts'] = forecasts
    logger.info("Generated all forecasts.")
    return state
