import logging
import pandas as pd
from typing import List, Dict, Any
from src.interfaces.broker_interface import Order, Fill
from src.brokers.paper_broker import PaperBroker
from src.brokers.transaction_costs import VolatilityAdjustedSlippageModel

logger = logging.getLogger(__name__)

class ExecutionAgent:
    """
    Manages the execution of portfolio orders.
    Handles sizing, risk checks, and routing to the broker.
    """
    
    def __init__(self, config: Dict[str, Any] = None, broker: Any = None):
        self.config = config or {}
        # Use injected broker or default to PaperBroker with Volatility Adjusted Costs
        if broker:
            self.broker = broker
        else:
            cost_model = VolatilityAdjustedSlippageModel()
            self.broker = PaperBroker(cost_model=cost_model) 
        
    def execute_orders(self, target_orders: List[Dict[str, Any]], data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Executes a list of target orders (target weights).
        
        Args:
            target_orders: List of dicts with {'symbol', 'target_weight', 'action'}
            data: Dictionary of DataFrames containing market data for symbols
            
        Returns:
            List of execution results (fills)
        """
        logger.info("--- Execution Agent: Processing Orders ---")
        
        # 1. Calculate Total Portfolio Value
        # For paper trading, we need to estimate it.
        # Value = Cash + Sum(Position * Price)
        cash = self.broker.get_cash()
        positions = self.broker.get_positions()
        
        equity = 0.0
        current_prices = {}
        
        # Get latest prices
        for symbol, df in data.items():
            if not df.empty:
                # Assume 'close' or 'Close' column exists
                col = 'close' if 'close' in df.columns else 'Close'
                if col in df.columns:
                    price = df.iloc[-1][col]
                    current_prices[symbol] = price
                else:
                    logger.warning(f"No price column found for {symbol}")
        
        # Calculate Equity
        for symbol, qty in positions.items():
            price = current_prices.get(symbol, 0.0)
            if price == 0.0:
                logger.warning(f"No price found for held position {symbol}. Using 0 value.")
            equity += qty * price
            
        total_portfolio_value = cash + equity
        logger.info(f"Total Portfolio Value: ${total_portfolio_value:,.2f} (Cash: ${cash:,.2f}, Equity: ${equity:,.2f})")
        
        execution_results = []
        
        # 2. Generate Trade Orders
        # We iterate through the target orders (which define the DESIRED state)
        # Note: The Portfolio Node sent us a list of ALL symbols with their target weights.
        # If a symbol is not in the list but we hold it, we should probably sell it (or the Portfolio Node should have sent a 0 weight).
        # For now, we assume the Portfolio Node sends targets for things we want to trade.
        
        for target in target_orders:
            symbol = target['symbol']
            target_weight = target['target_weight']
            
            if symbol not in current_prices:
                logger.warning(f"Skipping {symbol}: No price data available for execution.")
                continue
                
            price = current_prices[symbol]
            
            # Calculate Target Value
            target_value = total_portfolio_value * target_weight
            
            # Calculate Current Value
            current_qty = positions.get(symbol, 0)
            current_value = current_qty * price
            
            # Calculate Difference
            diff_value = target_value - current_value
            
            # Determine Action and Quantity
            # We use a threshold to avoid tiny trades
            min_trade_amount = 100.0 # $100 minimum trade
            
            if abs(diff_value) < min_trade_amount:
                logger.info(f"Skipping {symbol}: Trade value ${diff_value:.2f} below minimum.")
                continue
                
            quantity = int(abs(diff_value) / price)
            
            if quantity == 0:
                continue
                
            action = "BUY" if diff_value > 0 else "SELL"
            
            # Create Order Object
            order = Order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=price,
                order_type="MARKET"
            )
            
            logger.info(f"Placing Order: {action} {quantity} {symbol} @ ${price:.2f}")
            
            # Calculate Volatility and Volume for Cost Model
            volatility = 0.01 # Default
            avg_volume = 1000000.0 # Default
            
            if symbol in data and not data[symbol].empty:
                df = data[symbol]
                # Calculate Volatility (20-day std dev of returns)
                # Handle case sensitivity for columns
                close_col = 'close' if 'close' in df.columns else 'Close'
                vol_col = 'volume' if 'volume' in df.columns else 'Volume'
                
                if close_col in df.columns:
                    returns = df[close_col].pct_change()
                    vol = returns.tail(20).std()
                    if not pd.isna(vol):
                        volatility = float(vol)
                
                # Calculate Avg Volume (20-day)
                if vol_col in df.columns:
                    avg_vol = df[vol_col].tail(20).mean()
                    if not pd.isna(avg_vol) and avg_vol > 0:
                        avg_volume = float(avg_vol)

            # 3. Place Order
            fill = self.broker.place_order(order, volatility=volatility, avg_volume=avg_volume)
            
            if fill.status == "FILLED":
                logger.info(f"Order Filled: {fill}")
                execution_results.append({
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "price": fill.price,
                    "status": "FILLED",
                    "timestamp": fill.timestamp
                })
            else:
                logger.warning(f"Order Rejected: {fill.status}")
                execution_results.append({
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "status": "REJECTED",
                    "reason": fill.status
                })
                
        return execution_results
