from ..graphs.state import GraphState

import pandas as pd
import logging
import time
from ..graphs.state import GraphState
from models.model_zoo import ModelZoo, DataSpec, ModelTrainingResult
from sklearn.metrics import mean_absolute_percentage_error
import neuralforecast as nf
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS, AutoNBEATS, AutoDLinear, AutoTFT
from neuralforecast.models import NLinear
import torch
from models.gnn_model import GNNModel

logger = logging.getLogger(__name__)

def _train_all_models(model_zoo: ModelZoo, data_spec: DataSpec, edge_index=None, node_features=None, symbol_to_idx=None) -> dict[str, ModelTrainingResult]:
    """Helper to train all available models in the zoo."""
    results = {}
    # Train all core model families
    all_families = model_zoo.get_core_model_families()
    
    for family in all_families:
        train_method_name = f"train_{family.lower()}"
        train_method = getattr(model_zoo, train_method_name, None)
        if train_method:
            try:
                if family == "GNN":
                    # For GNN, update DataSpec with graph data
                    data_spec.edge_index = edge_index
                    data_spec.node_features = node_features
                    data_spec.symbol_to_idx = symbol_to_idx
                result = train_method(data_spec)
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
    
    horizon = 24
    
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
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
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
        if node_features_list:
            node_features = torch.tensor(node_features_list, dtype=torch.float)
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
        if 'date' in data.columns:
            df = data[['date', 'close']].copy()
            df = df.rename(columns={'date': 'ds', 'close': 'y'})
        else:
            # Use datetime index
            df = data[['close']].copy()
            df['ds'] = df.index
            df = df.rename(columns={'close': 'y'})
        
        df['unique_id'] = symbol
        df['ds'] = pd.to_datetime(df['ds'])
        df = df[['unique_id', 'ds', 'y']]
        
        train_df = df.iloc[:-horizon]
        val_df = df.iloc[-horizon:]
        full_df = df
        
        symbol_forecasts = {}
        
        if symbol in hpo_results and hpo_results[symbol]:
            model_results = hpo_results[symbol]
        else:
            # Fallback: train all models
            edge_index = state.get('edge_index')
            symbol_to_idx = state.get('symbol_to_idx')
            node_features = state.get('node_features')

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
                    node_features = torch.tensor(node_features_list, dtype=torch.float)

            data_spec = DataSpec(
                job_id=f"forecast_{symbol}_{int(time.time())}",
                symbol_scope=symbol,
                train_df=train_df,
                val_df=val_df,
                feature_cols=['y'],
                target_col='y',
                horizon=horizon
            )
            model_results = _train_all_models(model_zoo, data_spec, edge_index, node_features, symbol_to_idx)
        
        for model_name, model_result in model_results.items():
            if model_result:
                model_path = model_result.artifact_info.local_path
                if model_result.model_family == "GNN":
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
                        nf_inst = NeuralForecast.load(path=model_path)
                        
                        # Handle different model family naming
                        if model_result.model_family == "CNNLSTM":
                            column_name = "BiTCN"
                        elif model_result.model_family == "Ensemble":
                            column_name = "ensemble"
                        else:
                            column_name = model_result.model_family
                        
                        # Prepare future dataframe for prediction
                        futr_df = nf_inst.make_future_dataframe(df=train_df, h=horizon)
                        
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
                        preds_future = nf_inst.predict()
                        if model_family == "Ensemble":
                            pred_future = preds_future.select_dtypes(include=[float, int]).mean(axis=1).values
                        else:
                            pred_future = preds_future[column_name].values
            else:
                # Fallback: train default BaselineLinear
                from neuralforecast.models import NLinear
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
