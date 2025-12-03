from ..graphs.state import GraphState

import pandas as pd
import numpy as np
import logging
import time
import os
import sys

# Add root to path for models import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ..graphs.state import GraphState
from models.model_zoo import ModelZoo, DataSpec, ModelTrainingResult
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

# Conditional imports for heavy dependencies
try:
    if os.environ.get('SKIP_NEURALFORECAST', '').lower() not in ('true', '1', 'yes'):
        import neuralforecast as nf
        from neuralforecast import NeuralForecast
        from neuralforecast.auto import AutoNHITS, AutoNBEATS, AutoDLinear, AutoTFT
        from neuralforecast.models import NLinear
        import torch
        try:
            from models.gnn_model import GNNModel
        except ImportError:
            GNNModel = None
        _HAS_HEAVY_DEPS = True
    else:
        nf = NeuralForecast = AutoNHITS = AutoNBEATS = AutoDLinear = AutoTFT = NLinear = None
        torch = None
        GNNModel = None
        _HAS_HEAVY_DEPS = False
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import heavy dependencies: {e}")
    nf = NeuralForecast = AutoNHITS = AutoNBEATS = AutoDLinear = AutoTFT = NLinear = None
    torch = None
    GNNModel = None
    _HAS_HEAVY_DEPS = False

# Ensure NLinear is defined even if imports fail or are skipped
if 'NLinear' not in locals() and 'NLinear' not in globals():
    NLinear = None
if 'NeuralForecast' not in locals() and 'NeuralForecast' not in globals():
    NeuralForecast = None

logger = logging.getLogger(__name__)

def _train_all_models(model_zoo: ModelZoo, data_spec: DataSpec, edge_index=None, node_features=None, symbol_to_idx=None, priority_order=None) -> dict[str, ModelTrainingResult]:
    """Helper to train all available models in the zoo."""
    results = {}
    
    # Use priority order if provided, otherwise use all core model families
    if priority_order:
        model_families = priority_order
    else:
        model_families = model_zoo.get_core_model_families()
    
    for family in model_families:
        # Map config names to actual model family names
        if family == "LSTM":
            train_method_name = "train_lstm"
            actual_family = "NHITS"
        elif family == "AutoDLinear":
            train_method_name = "train_autodlinear" 
            actual_family = "DLinear"
        elif family == "TFT":
            train_method_name = "train_tft"
            actual_family = "TFT"
        elif family == "BaselineLinear":
            train_method_name = "train_baseline_linear"
            actual_family = "BaselineLinear"
        else:
            # For other families, use lowercase
            train_method_name = f"train_{family.lower()}"
            actual_family = family
        
        train_method = getattr(model_zoo, train_method_name, None)
        if train_method:
            try:
                if actual_family == "GNN":
                    # For GNN, update DataSpec with graph data
                    data_spec.edge_index = edge_index
                    data_spec.node_features = node_features
                    data_spec.symbol_to_idx = symbol_to_idx
                result = train_method(data_spec)
                # Update the result to use the config name instead of actual family name
                result.model_family = family
                results[family] = result
            except NotImplementedError:
                logger.warning(f"Skipping not implemented model family: {family}")
            except Exception as e:
                logger.error(f"Error training {family}: {e}")
                # Continue with other models instead of failing completely
    return results

def forecasting_node(state: GraphState) -> GraphState:
    """
    Generates forecasts for each symbol using optimized models from HPO.
    Computes performance on validation set.
    """
    logger.info("--- Node: Forecasting ---")
    
    symbols = state['symbols']
    features = state['features']
    hpo_results = state.get('hpo_results', {})
    config = state.get('config', {})
    
    # Validate features
    if not features:
        logger.error("No features available for forecasting")
        state['errors'].append("No features available for forecasting")
        return state
    
    horizon = 3
    
    forecasts = {}
    
    model_zoo = ModelZoo()
    
    if 'edge_index' not in state:
        # Construct graph
        symbols = state['symbols']
        symbol_to_idx = {symbol: i for i, symbol in enumerate(symbols)}
        num_nodes = len(symbols)
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        
        if _HAS_HEAVY_DEPS and torch is not None:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = None  # Skip graph construction for backtest
        
        node_features_list = []
        for symbol in symbols:
            if symbol in features:
                sym_data = features[symbol]
                # Convert dict back to DataFrame if needed
                if isinstance(sym_data, dict):
                    sym_data = pd.DataFrame.from_dict(sym_data, orient='index')
                    sym_data.index = pd.to_datetime(sym_data.index)
                # Ensure 'y' column exists (target variable)
                if 'y' not in sym_data.columns:
                    # Create target variable if missing
                    sym_data = sym_data.copy()
                    sym_data['y'] = sym_data['close'].pct_change(1).shift(-1)
                    sym_data = sym_data.dropna()
                
                if not sym_data.empty and 'y' in sym_data.columns:
                    sym_features = sym_data['y'].iloc[-horizon:].values
                    # Pad or truncate to ensure consistent length
                    if len(sym_features) < horizon:
                        # Pad with mean value
                        mean_val = sym_features.mean() if len(sym_features) > 0 else 0.0
                        padding = [mean_val] * (horizon - len(sym_features))
                        sym_features = list(sym_features) + padding
                    elif len(sym_features) > horizon:
                        # Truncate to horizon
                        sym_features = sym_features[-horizon:]
                    node_features_list.append(sym_features)
        
        if node_features_list and _HAS_HEAVY_DEPS and torch is not None:
            node_features = torch.tensor(node_features_list, dtype=torch.float)
        else:
            node_features = None
        
        state['edge_index'] = edge_index
        state['node_features'] = node_features
        state['symbol_to_idx'] = symbol_to_idx
    
    performance_summary = []
    
    for symbol in symbols:
        logger.info(f"Generating forecast for {symbol}...")
        
        if symbol not in features:
            logger.error(f"No features found for {symbol}")
            state['errors'].append(f"No features found for {symbol}")
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
            state['errors'].append(f"Missing required columns for {symbol}: {missing_columns}")
            continue
        if data.empty:
            logger.error(f"Empty data for {symbol}")
            state['errors'].append(f"Empty data for {symbol}")
            continue
        
        # Prepare df for NeuralForecast - handle both date column and datetime index
        df = data.copy()
        
        # If 'y' already exists (from FeatureAgent), drop it to avoid conflict when renaming 'close'
        if 'y' in df.columns:
            df = df.drop(columns=['y'])
            
        if 'date' in df.columns:
            # Keep all columns, rename date and close
            df = df.rename(columns={'date': 'ds', 'close': 'y'})
        else:
            # Use datetime index
            df['ds'] = df.index
            df = df.rename(columns={'close': 'y'})
        
        df['unique_id'] = symbol
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Identify exogenous columns (all columns except ds, y, unique_id)
        exog_cols = [col for col in df.columns if col not in ['ds', 'y', 'unique_id']]
        
        # Ensure we keep all columns
        cols_to_keep = ['unique_id', 'ds', 'y'] + exog_cols
        df = df[cols_to_keep]
        
        train_df = df.iloc[:-horizon]
        val_df = df.iloc[-horizon:]
        full_df = df
        
        symbol_forecasts = {}
        
        if symbol in hpo_results and hpo_results[symbol]:
            model_results = hpo_results[symbol]
        else:
            # Fallback: train models according to priority order
            edge_index = state.get('edge_index')
            symbol_to_idx = state.get('symbol_to_idx')
            node_features = state.get('node_features')
            # Use only BaselineLinear for short time series
            priority_order = ["BaselineLinear"]

            if node_features is None and edge_index is not None and symbol_to_idx is not None:
                node_features_list = []
                for sym in symbol_to_idx.keys():
                    if sym in features:
                        sym_data = features[sym]
                        # Convert dict back to DataFrame if needed
                        if isinstance(sym_data, dict):
                            sym_data = pd.DataFrame.from_dict(sym_data, orient='index')
                            sym_data.index = pd.to_datetime(sym_data.index)
                        # Ensure 'y' column exists (target variable)
                        if 'y' not in sym_data.columns:
                            # Create target variable if missing
                            sym_data = sym_data.copy()
                            sym_data['y'] = sym_data['close'].pct_change(1).shift(-1)
                            sym_data = sym_data.dropna()
                        
                        if not sym_data.empty and 'y' in sym_data.columns:
                            sym_features = sym_data['y'].iloc[-horizon:].values
                            # Pad or truncate to ensure consistent length
                            if len(sym_features) < horizon:
                                # Pad with mean value
                                mean_val = sym_features.mean() if len(sym_features) > 0 else 0.0
                                padding = [mean_val] * (horizon - len(sym_features))
                                sym_features = list(sym_features) + padding
                            elif len(sym_features) > horizon:
                                # Truncate to horizon
                                sym_features = sym_features[-horizon:]
                            node_features_list.append(sym_features)
                if node_features_list:
                    if _HAS_HEAVY_DEPS and torch is not None:
                        node_features = torch.tensor(node_features_list, dtype=torch.float)
                    else:
                        node_features = None  # Skip for backtest

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
            model_results = _train_all_models(model_zoo, data_spec, edge_index, node_features, symbol_to_idx, priority_order)
        
        for model_name, model_result in model_results.items():
            if model_result:
                model_path = model_result.artifact_info.path if model_result.artifact_info else None
                # Allow BaselineLinear to proceed without a path, as it's retrained on the fly
                if not model_path and model_result.model_family != "BaselineLinear":
                    logger.warning(f"No artifact path found for {model_name} - {symbol}")
                    continue
                    
                if model_result.model_family == "GNN":
                    # Skip GNN for backtest mode
                    if not _HAS_HEAVY_DEPS or GNNModel is None:
                        logger.warning(f"Skipping GNN model for {symbol} - heavy dependencies not available")
                        continue
                    
                    # Load GNN model
                    if state.get('node_features') is not None:
                        num_node_features = state['node_features'].shape[1]
                    else:
                        # Handle case where node_features is None, e.g., by setting a default or raising an error
                        logger.warning("node_features is None. GNN model cannot be loaded without node features.")
                        continue  # Skip to the next model

                    model = GNNModel(
                        num_node_features=num_node_features,
                        hidden_channels=64,
                        num_predictions=horizon
                    )
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    with torch.no_grad():
                        predictions = model(state['node_features'], state['edge_index'])
                    # Create preds_val
                    preds_list = []
                    idx = state['symbol_to_idx'][symbol]
                    symbol_preds = predictions[idx].numpy()
                    last_ds = train_df['ds'].iloc[-1]
                    for i in range(horizon):
                        preds_list.append({
                            'unique_id': symbol,
                            'ds': last_ds + pd.Timedelta(days=i+1),
                            'GNN': symbol_preds[i]
                        })
                    preds_val = pd.DataFrame(preds_list)
                    column_name = "GNN"
                    y_pred = preds_val[column_name].values
                    mape_val = model_result.best_val_mape
                    mae_val = model_result.best_val_mae
                    model_family = model_result.model_family
                    # For future forecast (same as validation for GNN)
                    pred_future = symbol_preds
                else:
                    # Load the trained model - handle different model types
                    statsforecast_families = ["AutoARIMA", "AutoETS", "AutoTheta"]
                    
                    if model_result.model_family in statsforecast_families:
                        # Load StatsForecast model
                        from statsforecast import StatsForecast
                        sf_inst = StatsForecast.load(path=model_path)
                        column_name = model_result.model_family
                        
                        # Make predictions
                        preds_val = sf_inst.predict(h=horizon)
                        y_pred = preds_val[column_name].values
                        mape_val = model_result.best_val_mape
                        mae_val = model_result.best_val_mae
                        model_family = model_result.model_family
                        
                        # For future forecast (same as validation for statsforecast)
                        pred_future = y_pred
                    else:
                        # Load NeuralForecast model
                        if model_result.model_family == "BaselineLinear":
                            # BaselineLinear doesn't save models, retrain it
                            if not _HAS_HEAVY_DEPS or NLinear is None or NeuralForecast is None:
                                # Use sklearn fallback
                                logger.info(f"Using sklearn LinearRegression fallback for {symbol} (retraining)")
                                
                                # Prepare data
                                X_train = np.arange(len(train_df)).reshape(-1, 1)
                                y_train = train_df['y'].values
                                
                                model = LinearRegression()
                                model.fit(X_train, y_train)
                                
                                column_name = 'LinearRegression'
                                # No validation prediction needed here as we are just setting up for future prediction logic below?
                                # Wait, the code below expects preds_val
                                
                                # Predict validation
                                X_val = np.arange(len(train_df), len(train_df) + len(val_df)).reshape(-1, 1)
                                y_pred = model.predict(X_val)
                                
                                mape_val = model_result.best_val_mape
                                mae_val = model_result.best_val_mae
                                model_family = model_result.model_family
                                
                                # For future forecast
                                X_future = np.arange(len(full_df), len(full_df) + horizon).reshape(-1, 1)
                                pred_future = model.predict(X_future)
                                
                                # Skip the rest of the NeuralForecast logic block
                                # We need to structure this carefully to avoid executing the NF block
                            else:
                                # NLinear is already imported globally
                                input_size = 2 * horizon
                                nf_inst = NeuralForecast(models=[NLinear(h=horizon, input_size=input_size, max_steps=50)], freq="D")
                                # Prepare data for retraining
                                unique_id = symbol.replace(":", "_")
                                train_nf_local = pd.DataFrame({"unique_id": unique_id, "ds": train_df['ds'], "y": train_df['y']})
                                nf_inst.fit(df=train_nf_local)
                                column_name = 'NLinear'
                        else:
                            if NeuralForecast is None:
                                logger.error(f"NeuralForecast not available but model {model_result.model_family} requires it. Skipping.")
                                continue
                            nf_inst = NeuralForecast.load(path=model_path)
                            
                            # Handle different model family naming
                            if model_result.model_family == "LSTM":
                                column_name = "NHITS"
                            elif model_result.model_family == "AutoDLinear":
                                column_name = "DLinear"
                            elif model_result.model_family == "TFT":
                                column_name = "TFT"
                            elif model_result.model_family == "Ensemble":
                                column_name = "ensemble"
                            else:
                                column_name = model_result.model_family
                        
                        # Only execute NF prediction logic if we haven't already done sklearn fallback
                        if not (model_result.model_family == "BaselineLinear" and (not _HAS_HEAVY_DEPS or NLinear is None)):
                            # Prepare future dataframe for prediction - ensure it matches training format
                            # Use the same unique_id format as training: symbol_scope.replace(":", "_")
                            unique_id_formatted = symbol.replace(":", "_")

                            # For NeuralForecast models, we need to create future dataframe that matches training structure
                            # Get the last date from training and create future dates with same frequency
                            last_train_date = train_df['ds'].max()
                            future_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1),
                                                       periods=horizon, freq='D')

                            # Create future dataframe with same unique_id as training
                            futr_df = pd.DataFrame({
                                'unique_id': [unique_id_formatted] * horizon,
                                'ds': future_dates
                            })

                            # Add any exogenous variables if they exist in training data
                            # NeuralForecast expects the same columns as training
                            if len(train_df.columns) > 2:  # More than unique_id, ds, y
                                exogenous_cols = [col for col in train_df.columns if col not in ['unique_id', 'ds', 'y']]
                                for col in exogenous_cols:
                                    if col in train_df.columns:
                                        # Use last known value for exogenous variables
                                        last_val = train_df[col].iloc[-1]
                                        futr_df[col] = last_val

                            try:
                                preds_val = nf_inst.predict(futr_df=futr_df)
                                
                                if model_result.model_family == "Ensemble":
                                    y_pred = preds_val.select_dtypes(include=[float, int]).mean(axis=1).values
                                else:
                                    y_pred = preds_val[column_name].values
                                mape_val = model_result.best_val_mape
                                mae_val = model_result.best_val_mae
                                model_family = model_result.model_family
                                
                                # Fit on full data for future forecast
                                nf_inst.fit(df=full_df)
                                # Create future dataframe with same format as training
                                unique_id_formatted = symbol.replace(":", "_")

                                # Get the last date from full data and create future dates
                                last_full_date = full_df['ds'].max()
                                future_dates = pd.date_range(start=last_full_date + pd.Timedelta(days=1),
                                                           periods=horizon, freq='D')

                                futr_future_df = pd.DataFrame({
                                    'unique_id': [unique_id_formatted] * horizon,
                                    'ds': future_dates
                                })

                                # Add any exogenous variables if they exist in training data
                                if len(full_df.columns) > 2:  # More than unique_id, ds, y
                                    exogenous_cols = [col for col in full_df.columns if col not in ['unique_id', 'ds', 'y']]
                                    for col in exogenous_cols:
                                        if col in full_df.columns:
                                            # Use last known value for exogenous variables
                                            last_val = full_df[col].iloc[-1]
                                            futr_future_df[col] = last_val

                                preds_future = nf_inst.predict(futr_df=futr_future_df)
                                if model_family == "Ensemble":
                                    pred_future = preds_future.select_dtypes(include=[float, int]).mean(axis=1).values
                                else:
                                    # Use the same column name mapping as above
                                    if model_family == "LSTM":
                                        future_column = "NHITS"
                                    elif model_family == "AutoDLinear":
                                        future_column = "DLinear"
                                    elif model_family == "TFT":
                                        future_column = "TFT"
                                    elif model_family == "BaselineLinear":
                                        future_column = "NLinear"
                                    else:
                                        future_column = model_family
                                    pred_future = preds_future[future_column].values
                            except Exception as e:
                                logger.error(f"Error predicting with {model_result.model_family} for {symbol}: {e}")
                                # Skip this model and continue with others
                                continue
            else:
                # Fallback: train default BaselineLinear (skip for backtest if heavy deps not available)
                if not _HAS_HEAVY_DEPS or NLinear is None or NeuralForecast is None:
                    # Use sklearn fallback
                    logger.info(f"Using sklearn LinearRegression fallback for {symbol}")
                    
                    # Prepare data
                    X_train = np.arange(len(train_df)).reshape(-1, 1)
                    y_train = train_df['y'].values
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Predict validation
                    X_val = np.arange(len(train_df), len(train_df) + len(val_df)).reshape(-1, 1)
                    y_pred = model.predict(X_val)
                    
                    mape_val = mean_absolute_percentage_error(val_df['y'], y_pred)
                    mae_val = np.mean(np.abs(val_df['y'] - y_pred))
                    column_name = 'LinearRegression'
                    model_family = 'BaselineLinear'
                    
                    # Predict future
                    X_future = np.arange(len(full_df), len(full_df) + horizon).reshape(-1, 1)
                    pred_future = model.predict(X_future)
                else:
                    # NLinear is already imported globally
                    input_size = 2 * horizon
                    model = NLinear(h=horizon, input_size=input_size, max_steps=50)
                    nf_inst = NeuralForecast(models=[model], freq="D")
                    nf_inst.fit(df=train_df)
                    futr_df = pd.DataFrame({
                        'unique_id': [symbol] * horizon,
                        'ds': pd.date_range(start=val_df['ds'].iloc[-1], periods=horizon+1, freq='D')[1:]
                    })
                    preds_val = nf_inst.predict(futr_df=futr_df)
                    y_pred = preds_val['NLinear'].values
                    mape_val = 0.5  # Fallback, no validation mape available
                    mae_val = 0.5
                    column_name = 'NLinear'
                    model_family = 'BaselineLinear'
                    
                    # Fit on full data for future forecast
                    nf_inst.fit(df=full_df)
                    preds_future = nf_inst.predict()
                    pred_future = preds_future['NLinear'].values
            
            performance_summary.append({
                'symbol': symbol,
                'model_family': model_family,
                'mape': mape_val,
                'model_id': model_result.best_model_id if model_result else None
            })
            
            # Create forecast df
            ds_future = pd.date_range(start=data.index.max() + pd.Timedelta(days=1), periods=horizon, freq='D')
            forecast_df = pd.DataFrame({'ds': ds_future, model_family: pred_future})
            symbol_forecasts[model_family] = forecast_df
        
        forecasts[symbol] = symbol_forecasts
        logger.info(f"Generated forecast for {symbol}.")
    
    state['forecasts'] = forecasts
    state['performance_summary'] = pd.DataFrame(performance_summary)
    logger.info("Generated all forecasts.")
    return state

def action_executor_node(state: GraphState) -> GraphState:
    """
    Executes the recommended actions, such as promoting a model.
    """
    logger.info("--- Node: Action Executor ---")
    
    config = state.get('config', {})
    model_zoo = ModelZoo()
    executed_actions = []
    
    for action in state['recommended_actions']:
        if action.startswith("Promote"):
            parts = action.split(" ")
            model_family = parts[1]
            symbol = parts[3]
            model_zoo.promote_model(model_family, symbol)
            executed_actions.append(action)
            
    state['executed_actions'] = executed_actions
    logger.info(f"Executed {len(state['executed_actions'])} actions.")
    return state
