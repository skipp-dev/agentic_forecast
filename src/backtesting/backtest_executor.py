import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.brokers.paper_broker import PaperBroker
from src.brokers.transaction_costs import LinearSlippageModel

logger = logging.getLogger(__name__)

class BacktestExecutor:
    """
    Executes a backtest by running the agentic graph over a historical timeline.
    Implements an Event-Driven architecture where the system steps through time,
    preventing look-ahead bias and simulating realistic execution.
    """
    def __init__(self, graph, start_date: str, end_date: str, step_days: int = 1, initial_cash: float = 100000.0):
        """
        Initialize the backtest executor.
        
        Args:
            graph: The compiled LangGraph executable.
            start_date: Start date for the backtest (ISO string).
            end_date: End date for the backtest (ISO string).
            step_days: Number of days to advance in each step.
            initial_cash: Initial cash for the backtest portfolio.
        """
        self.graph = graph
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.step_days = step_days
        self.initial_cash = initial_cash
        self.results = []
        self.performance_metrics = []
        self.portfolio_history = []
        
        # Initialize Backtest Broker
        # We use a separate file to avoid corrupting the production portfolio
        self.backtest_broker_file = "data/backtest_portfolio.json"
        if os.path.exists(self.backtest_broker_file):
            os.remove(self.backtest_broker_file)
            
        # Initialize Cost Model (e.g., 5bps slippage, $0.005/share commission)
        self.cost_model = LinearSlippageModel(slippage_bps=5.0, commission_per_share=0.005)
            
        self.broker = PaperBroker(state_file=self.backtest_broker_file, initial_cash=initial_cash, cost_model=self.cost_model)
        logger.info(f"Initialized Backtest Broker with ${initial_cash:,.2f} and LinearSlippageModel")

    def run(self, initial_state: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute the backtest simulation.
        
        Args:
            initial_state: The initial state dictionary for the graph.
            
        Returns:
            DataFrame containing the backtest results.
        """
        current_date = self.start_date
        
        total_days = (self.end_date - self.start_date).days
        steps = total_days // self.step_days
        
        logger.info(f"Starting Event-Driven Backtest from {self.start_date.date()} to {self.end_date.date()} ({steps} steps)")
        
        with tqdm(total=steps) as pbar:
            while current_date <= self.end_date:
                cutoff_str = current_date.strftime('%Y-%m-%d')
                pbar.set_description(f"Processing {cutoff_str}")
                
                # Prepare State
                state_input = initial_state.copy()
                state_input['cutoff_date'] = cutoff_str
                state_input['broker'] = self.broker # Inject the backtest broker
                state_input['run_type'] = 'BACKTEST' # Signal to agents that this is a backtest
                
                try:
                    # Run the graph
                    # The graph will:
                    # 1. Fetch data (filtered by cutoff_date)
                    # 2. Generate forecasts
                    # 3. Plan strategy
                    # 4. Execute trades (using self.broker)
                    result_state = self.graph.invoke(state_input, {"recursion_limit": 100})
                    
                    # Capture Results
                    self._capture_results(result_state, cutoff_str)
                    
                    # Capture Portfolio State
                    self._capture_portfolio_state(cutoff_str)
                    
                except Exception as e:
                    logger.error(f"Backtest failed for {cutoff_str}: {e}")
                    # Continue to next date even if one fails
                
                current_date += timedelta(days=self.step_days)
                pbar.update(1)
        
        return pd.DataFrame(self.results)

    def _capture_portfolio_state(self, date: str):
        """Capture the current value of the portfolio."""
        cash = self.broker.get_cash()
        positions = self.broker.get_positions()
        # Note: get_portfolio_value() in PaperBroker currently returns cash only
        # We should calculate equity if possible, but we need prices.
        # For now, we log what we have.
        
        record = {
            'date': date,
            'cash': cash,
            'positions_count': len(positions),
            'total_value': self.broker.get_portfolio_value() # Approximation
        }
        self.portfolio_history.append(record)

    def _capture_results(self, state: Dict[str, Any], date: str):
        """Capture metrics and forecasts from the state."""
        forecasts = state.get('forecasts', {})
        performance = state.get('performance_summary', pd.DataFrame())
        
        # Capture Forecasts
        for symbol, symbol_forecasts in forecasts.items():
            for model_name, forecast_df in symbol_forecasts.items():
                if not forecast_df.empty:
                    # The forecast_df contains 'ds' and the model prediction column
                    # We are interested in the first forecast (t+1)
                    first_pred = forecast_df.iloc[0]
                    pred_value = first_pred.get(model_name)
                    
                    record = {
                        'date': date,
                        'symbol': symbol,
                        'model': model_name,
                        'forecast': pred_value,
                        'forecast_date': first_pred.get('ds')
                    }
                    self.results.append(record)

        # Capture Performance Metrics (if available from backtesting within the step)
        if not performance.empty:
            performance['date'] = date
            self.performance_metrics.append(performance)

    def get_performance_metrics(self) -> pd.DataFrame:
        """Return the collected performance metrics."""
        if self.performance_metrics:
            return pd.concat(self.performance_metrics, ignore_index=True)
        return pd.DataFrame()
        
    def get_portfolio_history(self) -> pd.DataFrame:
        """Return the portfolio value history."""
        return pd.DataFrame(self.portfolio_history)

    def save_results(self, output_dir: str = "backtest_results"):
        """Save results to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        
        results_df = pd.DataFrame(self.results)
        if not results_df.empty:
            results_df.to_csv(os.path.join(output_dir, "forecasts.csv"), index=False)
            
        perf_df = self.get_performance_metrics()
        if not perf_df.empty:
            perf_df.to_csv(os.path.join(output_dir, "performance.csv"), index=False)
            
        port_df = self.get_portfolio_history()
        if not port_df.empty:
            port_df.to_csv(os.path.join(output_dir, "portfolio_history.csv"), index=False)
            
        logger.info(f"Results saved to {output_dir}")

