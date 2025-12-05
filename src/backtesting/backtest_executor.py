import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)

class BacktestExecutor:
    """
    Executes a backtest by running the agentic graph over a historical timeline.
    """
    def __init__(self, graph, start_date: str, end_date: str, step_days: int = 1):
        """
        Initialize the backtest executor.
        
        Args:
            graph: The compiled LangGraph executable.
            start_date: Start date for the backtest (ISO string).
            end_date: End date for the backtest (ISO string).
            step_days: Number of days to advance in each step.
        """
        self.graph = graph
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.step_days = step_days
        self.results = []
        self.performance_metrics = []

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
        
        logger.info(f"Starting backtest from {self.start_date.date()} to {self.end_date.date()} ({steps} steps)")
        
        with tqdm(total=steps) as pbar:
            while current_date <= self.end_date:
                cutoff_str = current_date.strftime('%Y-%m-%d')
                pbar.set_description(f"Processing {cutoff_str}")
                
                # Update state with current cutoff
                # We must copy the initial state to avoid polluting it, 
                # but we might want to carry over some state (like trained models) if the graph supports it.
                # For a strict walk-forward, we usually want to carry over the model state but refresh data.
                # However, LangGraph state is usually ephemeral per run unless we use a checkpointer.
                # Here we pass the state as input.
                
                state_input = initial_state.copy()
                state_input['cutoff_date'] = cutoff_str
                
                try:
                    # Run the graph
                    # Increase recursion limit for complex graph with potential HPO loops
                    result_state = self.graph.invoke(state_input, {"recursion_limit": 100})
                    
                    # Extract relevant results
                    self._capture_results(result_state, cutoff_str)
                    
                except Exception as e:
                    logger.error(f"Backtest failed for {cutoff_str}: {e}")
                    # Continue to next date even if one fails
                
                current_date += timedelta(days=self.step_days)
                pbar.update(1)
        
        return pd.DataFrame(self.results)

    def _capture_results(self, state: Dict[str, Any], date: str):
        """Capture metrics and forecasts from the state."""
        forecasts = state.get('forecasts', {})
        actions = state.get('recommended_actions', [])
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

        # Capture Actions
        # Actions are usually strings or dicts. 
        # Assuming list of dicts or strings.
        # Based on previous read, actions are strings like "Promote NLinear symbol"
        # But state.py says List[Dict[str, Any]]. 
        # Let's handle both or check execution_nodes.py again.
        # execution_nodes.py: for action in state['recommended_actions']: if action.startswith("Promote")
        # So it seems they are strings.
        
        # Capture Performance Metrics (if available from backtesting within the step)
        if not performance.empty:
            performance['date'] = date
            self.performance_metrics.append(performance)

    def get_performance_metrics(self) -> pd.DataFrame:
        """Return the collected performance metrics."""
        if self.performance_metrics:
            return pd.concat(self.performance_metrics, ignore_index=True)
        return pd.DataFrame()

    def save_results(self, output_dir: str = "backtest_results"):
        """Save results to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        
        results_df = pd.DataFrame(self.results)
        if not results_df.empty:
            results_df.to_csv(os.path.join(output_dir, "forecasts.csv"), index=False)
            
        perf_df = self.get_performance_metrics()
        if not perf_df.empty:
            perf_df.to_csv(os.path.join(output_dir, "performance.csv"), index=False)
            
        logger.info(f"Results saved to {output_dir}")
