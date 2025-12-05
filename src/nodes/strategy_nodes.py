import logging
from typing import Dict, Any
import pandas as pd
from src.core.state import PipelineGraphState
from src.agents.strategy_selection_agent import StrategySelectionAgent

logger = logging.getLogger(__name__)

def strategy_node(state: PipelineGraphState) -> PipelineGraphState:
    """
    Generates trading signals based on forecasts and selected strategies.
    """
    logger.info("--- Node: Strategy Generation ---")
    
    forecasts = state.get('forecasts', {})
    analytics = state.get('analytics_results', {})
    
    # Initialize Strategy Agent
    strategy_agent = StrategySelectionAgent(config=state.get('config', {}))
    
    # Get current regimes
    regimes_data = state.get('regimes', {})
    current_regimes = {}
    if regimes_data:
        try:
            # Extract the latest regime for each type
            for regime_type, regime_series in regimes_data.items():
                if isinstance(regime_series, pd.Series) and not regime_series.empty:
                    current_regimes[regime_type] = str(regime_series.iloc[-1])
                elif isinstance(regime_series, str):
                    current_regimes[regime_type] = regime_series
        except Exception as e:
            logger.warning(f"Error extracting current regimes: {e}")

    # Select strategies based on regimes
    selected_strategies = strategy_agent.select_strategies(current_regimes=current_regimes)
    best_strategy = selected_strategies[0] if selected_strategies else None
    strategy_name = best_strategy['strategy_name'] if best_strategy else "momentum_growth"
    logger.info(f"Selected Strategy: {strategy_name} (Regimes: {current_regimes})")
    
    signals = {}
    
    for symbol, model_forecasts in forecasts.items():
        try:
            forecast_df = None
            
            # Logic to extract the best forecast DataFrame from the model_forecasts structure
            # model_forecasts is expected to be {model_name: forecast_data}
            
            if isinstance(model_forecasts, pd.DataFrame):
                forecast_df = model_forecasts
            elif isinstance(model_forecasts, dict):
                # Check if it's a serialized DataFrame (has 'ds' key) or a dict of models
                # If it has 'ds' and 'yhat'/'y', it's likely a single forecast
                if 'ds' in model_forecasts:
                     try:
                         forecast_df = pd.DataFrame(model_forecasts)
                     except ValueError:
                         forecast_df = pd.DataFrame([model_forecasts])
                else:
                    # It's likely a dict of models: {'NLinear': ..., 'Baseline': ...}
                    priorities = ['NLinear', 'NHITS', 'BaselineLinear', 'TFT']
                    
                    # Try priority models
                    for model in priorities:
                        if model in model_forecasts:
                            data = model_forecasts[model]
                            # Convert to DataFrame
                            if isinstance(data, dict):
                                try:
                                    data = pd.DataFrame(data)
                                except ValueError:
                                    try:
                                        data = pd.DataFrame([data])
                                    except:
                                        pass
                            
                            if isinstance(data, pd.DataFrame) and not data.empty:
                                forecast_df = data
                                break
                    
                    # Fallback to any model
                    if forecast_df is None:
                        for model, data in model_forecasts.items():
                            if isinstance(data, dict):
                                try:
                                    data = pd.DataFrame(data)
                                except ValueError:
                                    try:
                                        data = pd.DataFrame([data])
                                    except:
                                        pass
                            if isinstance(data, pd.DataFrame) and not data.empty:
                                forecast_df = data
                                break

            if forecast_df is None or forecast_df.empty:
                logger.warning(f"No valid forecast DataFrame found for {symbol}, skipping")
                continue

            # Ensure 'ds' is datetime
            if 'ds' in forecast_df.columns:
                forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
            else:
                logger.warning(f"Forecast DataFrame for {symbol} missing 'ds' column")
                continue

            # Get latest forecast
            # Assuming forecast_df has 'ds' and 'yhat' (or similar)
            # We need to find the forecast for "tomorrow" or the next step
            
            # Simple logic: Compare last known price with next forecast
            # In a real scenario, we'd have historical data to compare against
            
            # For now, let's assume the forecast_df contains future predictions
            # and we need the last actual close price.
            # We can get that from state['data'][symbol]
            
            raw_data = state.get('data', {}).get(symbol)
            
            # Handle potential serialization to dict
            if isinstance(raw_data, dict) and raw_data:
                try:
                    # Attempt to reconstruct DataFrame
                    # Check if it looks like a JSON-serialized DataFrame (e.g. dict of columns)
                    raw_data = pd.DataFrame(raw_data)
                    # Ensure index is datetime if it looks like it should be
                    if 'date' in raw_data.columns:
                        raw_data['date'] = pd.to_datetime(raw_data['date'])
                        raw_data.set_index('date', inplace=True)
                    elif isinstance(raw_data.index, pd.Index) and not isinstance(raw_data.index, pd.DatetimeIndex):
                         # Try to convert index to datetime if it looks like dates
                         try:
                             raw_data.index = pd.to_datetime(raw_data.index)
                         except:
                             pass
                except Exception as e:
                    logger.warning(f"Could not convert raw_data dict to DataFrame for {symbol}: {e}")

            if raw_data is None or (isinstance(raw_data, pd.DataFrame) and raw_data.empty) or (isinstance(raw_data, dict) and not raw_data):
                logger.warning(f"No raw data for {symbol}, skipping strategy generation")
                continue
                
            if not isinstance(raw_data, pd.DataFrame):
                logger.warning(f"raw_data for {symbol} is not a DataFrame (type: {type(raw_data)}), skipping")
                continue

            last_close = raw_data['close'].iloc[-1]
            last_date = raw_data.index[-1] if isinstance(raw_data.index, pd.DatetimeIndex) else raw_data['date'].iloc[-1]
            
            # Find the first forecast after the last known date
            # forecast_df usually has 'ds' column
            future_forecasts = forecast_df[forecast_df['ds'] > last_date]
            
            if future_forecasts.empty:
                logger.warning(f"No future forecasts for {symbol}")
                continue
                
            next_forecast = future_forecasts.iloc[0]
            predicted_price = next_forecast['NLinear'] if 'NLinear' in next_forecast else next_forecast.get('yhat')
            
            if predicted_price is None:
                 # Try finding any column that looks like a prediction
                 pred_cols = [c for c in next_forecast.index if c not in ['ds', 'unique_id', 'y']]
                 if pred_cols:
                     predicted_price = next_forecast[pred_cols[0]]
            
            if predicted_price is None:
                logger.warning(f"Could not identify prediction column for {symbol}")
                continue
                
            # Calculate expected return
            expected_return = (predicted_price - last_close) / last_close
            
            # Select Strategy
            # strategy_name is determined at the node level based on macro regimes
            
            # Generate Signal
            signal = "HOLD"
            confidence = 0.0
            
            # Simple Threshold Strategy
            threshold = 0.01 # 1%
            
            if expected_return > threshold:
                signal = "BUY"
                confidence = min(abs(expected_return) * 10, 1.0) # Scale confidence
            elif expected_return < -threshold:
                signal = "SELL"
                confidence = min(abs(expected_return) * 10, 1.0)
            
            signals[symbol] = {
                "signal": signal,
                "current_price": float(last_close),
                "predicted_price": float(predicted_price),
                "expected_return": float(expected_return),
                "confidence": float(confidence),
                "strategy": strategy_name,
                "timestamp": str(pd.Timestamp.now())
            }
            
            logger.info(f"Signal for {symbol}: {signal} (Return: {expected_return:.2%})")
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            
    state['signals'] = signals
    return state
