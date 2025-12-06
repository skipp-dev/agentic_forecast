import json
import os
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from filelock import FileLock
from src.interfaces.broker_interface import BrokerInterface, Order, Fill
from src.services.database_service import DatabaseService
from src.brokers.transaction_costs import TransactionCostModel, LinearSlippageModel
from src.risk.circuit_breaker import CircuitBreaker
from src.monitoring.metrics import PORTFOLIO_VALUE, CASH_BALANCE, TRADE_COUNT

logger = logging.getLogger(__name__)

class PaperBroker(BrokerInterface):
    """
    Simulates a broker for paper trading.
    Persists state to a JSON file and SQLite Database.
    """
    
    def __init__(self, state_file: str = "data/paper_portfolio.json", initial_cash: float = 100000.0, cost_model: Optional[TransactionCostModel] = None, db_path: Optional[str] = None):
        self.state_file = state_file
        self.lock_file = state_file + ".lock"
        self.initial_cash = initial_cash
        self.is_backtest = "backtest" in state_file
        self.db_service = DatabaseService(db_path=db_path) if db_path else DatabaseService()
        self.cost_model = cost_model or LinearSlippageModel()
        self.portfolio = self._load_state()
        
        # Initialize Circuit Breaker
        self.circuit_breaker = CircuitBreaker(max_drawdown_pct=0.10, max_daily_loss_pct=0.05)
        # Initialize equity tracking
        current_equity = self.get_portfolio_value()
        self.circuit_breaker.update_equity(current_equity, is_new_day=True) # Assume start of run is new day for now
        
        # Initialize Metrics
        if not self.is_backtest:
            PORTFOLIO_VALUE.set(current_equity)
            CASH_BALANCE.set(self.portfolio["cash"])
        
    def _load_state(self) -> Dict[str, Any]:
        # Try to load from DB first
        state = self.db_service.get_latest_portfolio(is_backtest=self.is_backtest)
        
        if state:
            state['history'] = [] # Legacy compatibility
            return state

        # Fallback to file (DEPRECATED: Only for migration)
        if os.path.exists(self.state_file):
            try:
                lock = FileLock(self.lock_file)
                with lock:
                    with open(self.state_file, 'r') as f:
                        logger.warning("Loading state from deprecated JSON file. Migrating to DB...")
                        state = json.load(f)
                        # Migrate to DB immediately
                        self.db_service.save_portfolio(state, is_backtest=self.is_backtest)
                        return state
            except Exception as e:
                logger.error(f"Failed to load paper portfolio state: {e}")
        
        # Initialize new state
        state = {
            "cash": self.initial_cash,
            "positions": {}, # symbol: quantity
            "history": []
        }
        self.portfolio = state
        self._save_state()
        return state

    def _save_state(self):
        # Save to DB
        try:
            # Calculate total value (approximate as we might not have live prices here)
            # For now, just use cash + 0 (since we don't have prices easily accessible here without passing them)
            # Ideally, the caller updates total_value, but for persistence we save what we have.
            total_val = self.portfolio["cash"] 
            # If we wanted real value, we'd need to inject a price source.
            
            self.db_service.save_portfolio_state(
                cash=self.portfolio['cash'],
                positions=self.portfolio['positions'],
                total_value=total_val,
                is_backtest=self.is_backtest
            )
            
            # Update Metrics
            if not self.is_backtest:
                PORTFOLIO_VALUE.set(total_val)
                CASH_BALANCE.set(self.portfolio['cash'])
                
        except Exception as e:
            logger.error(f"Failed to save portfolio to DB: {e}")

        # JSON State persistence is DEPRECATED and REMOVED.
        # State is now exclusively managed by DatabaseService (SQLite/Postgres).
        # Legacy files (paper_portfolio.json) are read-only for migration in _load_state.

    def get_cash(self) -> float:
        return self.portfolio["cash"]

    def get_positions(self) -> Dict[str, float]:
        return self.portfolio["positions"]

    def get_portfolio_value(self) -> float:
        # This is an approximation. In a real system, we need live prices.
        # For paper trading, we might not have live prices here unless passed.
        # We will return Cash + (Positions * Last Known Price).
        # Since we don't store prices in state, this is tricky.
        # We will assume the caller handles total value calculation if they have prices,
        # or we return cash if no positions.
        # For now, let's just return Cash. The Execution Agent should calculate Equity.
        return self.portfolio["cash"] 

    def place_order(self, order: Order, **kwargs) -> Fill:
        # 1. Check Circuit Breaker
        # We need current equity and exposure. 
        # Since get_portfolio_value is approximate, we rely on it for now.
        current_equity = self.get_portfolio_value()
        # Exposure is roughly Total Value - Cash (assuming long only)
        current_exposure = current_equity - self.portfolio["cash"]
        
        # Update breaker state
        self.circuit_breaker.update_equity(current_equity)
        
        if not self.circuit_breaker.check_risk(current_equity, current_exposure):
            logger.critical(f"Order REJECTED by Circuit Breaker: {self.circuit_breaker.trip_reason}")
            return Fill(
                order_id=str(uuid.uuid4()),
                symbol=order.symbol,
                action=order.action,
                quantity=0,
                price=0,
                timestamp=datetime.now(),
                status="REJECTED_RISK"
            )

        if order.price is None or order.price <= 0:
            logger.warning(f"Order for {order.symbol} rejected: Invalid price {order.price}")
            return Fill(
                order_id=str(uuid.uuid4()),
                symbol=order.symbol,
                action=order.action,
                quantity=0,
                price=0,
                timestamp=datetime.now(),
                status="REJECTED"
            )

        # Calculate Transaction Costs
        order_value = order.quantity * order.price
        slippage = self.cost_model.get_slippage_cost(order_value, order.quantity, order.price, **kwargs)
        commission = self.cost_model.get_commission_cost(order_value, order.quantity, order.price, **kwargs)
        
        # Adjust Fill Price based on Slippage
        # Buy: Pay more (Price + SlippagePerShare)
        # Sell: Receive less (Price - SlippagePerShare)
        slippage_per_share = slippage / order.quantity if order.quantity > 0 else 0
        
        if order.action == "BUY":
            fill_price = order.price + slippage_per_share
            total_cost = (fill_price * order.quantity) + commission
            
            if self.portfolio["cash"] >= total_cost:
                self.portfolio["cash"] -= total_cost
                current_qty = self.portfolio["positions"].get(order.symbol, 0)
                self.portfolio["positions"][order.symbol] = current_qty + order.quantity
                status = "FILLED"
            else:
                logger.warning(f"Order for {order.symbol} rejected: Insufficient funds (Req: {total_cost:.2f}, Avail: {self.portfolio['cash']:.2f})")
                status = "REJECTED"
                fill_price = 0
                commission = 0
                
        elif order.action == "SELL":
            fill_price = order.price - slippage_per_share
            total_proceeds = (fill_price * order.quantity) - commission
            
            current_qty = self.portfolio["positions"].get(order.symbol, 0)
            if current_qty >= order.quantity:
                self.portfolio["cash"] += total_proceeds
                self.portfolio["positions"][order.symbol] = current_qty - order.quantity
                # Clean up zero positions
                if self.portfolio["positions"][order.symbol] <= 0:
                    del self.portfolio["positions"][order.symbol]
                status = "FILLED"
            else:
                logger.warning(f"Order for {order.symbol} rejected: Insufficient position")
                status = "REJECTED"
                fill_price = 0
                commission = 0
        
        else:
            status = "REJECTED"
            fill_price = 0
            commission = 0

        # Update Metrics
        if not self.is_backtest:
            TRADE_COUNT.labels(symbol=order.symbol, action=order.action, status=status).inc()

        fill = Fill(
            order_id=str(uuid.uuid4()),
            symbol=order.symbol,
            action=order.action,
            quantity=order.quantity,
            price=fill_price,
            timestamp=datetime.now(),
            commission=commission,
            status=status
        )
        
        if status == "FILLED":
            self.portfolio["history"].append({
                "timestamp": datetime.now().isoformat(),
                "symbol": order.symbol,
                "action": order.action,
                "quantity": order.quantity,
                "price": fill_price,
                "commission": commission,
                "slippage": slippage,
                "cost": total_cost if order.action == "BUY" else -total_proceeds
            })
            self._save_state()
            
        # Log to DB
        try:
            trade_dict = fill.__dict__.copy()
            if isinstance(trade_dict['timestamp'], datetime):
                trade_dict['timestamp'] = trade_dict['timestamp'].isoformat()
            self.db_service.log_trade(trade_dict, is_backtest=self.is_backtest)
        except Exception as e:
            logger.error(f"Failed to log trade to DB: {e}")
            
        return fill
