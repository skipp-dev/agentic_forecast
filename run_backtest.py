import os
import sys
import logging
import yaml
import pandas as pd
from langgraph.graph import StateGraph, END
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.state import PipelineGraphState
from src.nodes.execution_nodes import (
    data_ingestion_node,
    feature_engineering_node,
    forecasting_node
)
from src.nodes.macro_nodes import macro_data_node, regime_detection_node
from src.nodes.hpo_nodes import hpo_node
from src.nodes.agent_nodes import analytics_node
from src.nodes.monitoring_nodes import monitoring_node
from src.nodes.retraining_nodes import retraining_node
from src.nodes.reporting_nodes import generate_report_node
from src.nodes.strategy_nodes import strategy_node
from src.nodes.portfolio_nodes import portfolio_construction_node
from src.nodes.trade_execution_nodes import trade_execution_node
from src.agents.orchestrator_agent import OrchestratorAgent
from src.backtesting.backtest_executor import BacktestExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def build_graph(config):
    orchestrator = OrchestratorAgent(config=config)

    def route_after_features(state: PipelineGraphState):
        # For backtest, we might want to skip HPO to save time, or force it.
        # Let's assume we skip HPO for this test run unless configured otherwise.
        return "forecasting"

    def route_after_analytics(state: PipelineGraphState):
        # Always proceed to strategy in backtest
        return "strategy"

    workflow = StateGraph(PipelineGraphState)

    # Add nodes
    workflow.add_node("data_ingestion", data_ingestion_node)
    workflow.add_node("macro_data", macro_data_node)
    workflow.add_node("regime_detection", regime_detection_node)
    workflow.add_node("feature_engineering", feature_engineering_node)
    workflow.add_node("hpo", hpo_node)
    workflow.add_node("forecasting", forecasting_node)
    workflow.add_node("analytics", analytics_node)
    workflow.add_node("strategy", strategy_node)
    workflow.add_node("portfolio_construction", portfolio_construction_node)
    workflow.add_node("trade_execution", trade_execution_node)
    workflow.add_node("monitoring", monitoring_node)
    workflow.add_node("retraining", retraining_node)
    workflow.add_node("reporting", generate_report_node)

    # Define edges
    workflow.set_entry_point("data_ingestion")
    workflow.add_edge("data_ingestion", "macro_data")
    workflow.add_edge("macro_data", "regime_detection")
    workflow.add_edge("regime_detection", "feature_engineering")
    
    workflow.add_conditional_edges(
        "feature_engineering",
        route_after_features,
        {
            "hpo": "hpo",
            "forecasting": "forecasting"
        }
    )
    
    workflow.add_edge("hpo", "forecasting")
    workflow.add_edge("forecasting", "analytics")
    
    workflow.add_conditional_edges(
        "analytics",
        route_after_analytics,
        {
            "strategy": "strategy",
            "retrain": "retraining",
            END: END
        }
    )
    
    workflow.add_edge("strategy", "portfolio_construction")
    workflow.add_edge("portfolio_construction", "trade_execution")
    workflow.add_edge("trade_execution", "monitoring")
    workflow.add_edge("monitoring", "reporting")
    workflow.add_edge("reporting", END)
    workflow.add_edge("retraining", "forecasting")

    return workflow.compile()

def main():
    config = load_config()
    
    # Test Parameters
    symbols = ["AAPL", "MSFT"] # Top 2 for speed
    start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Running Backtest for {symbols} from {start_date} to {end_date}")
    
    graph = build_graph(config)
    
    executor = BacktestExecutor(
        graph=graph,
        start_date=start_date,
        end_date=end_date,
        step_days=1,
        initial_cash=100000.0
    )
    
    initial_state = {
        "symbols": symbols,
        "config": config,
        "run_type": "BACKTEST"
    }
    
    results_df = executor.run(initial_state)
    
    executor.save_results("backtest_results_test")
    
    print("\nBacktest Completed!")
    print(f"Results saved to backtest_results_test/")
    
    # Print final portfolio value
    history = executor.get_portfolio_history()
    if not history.empty:
        final_val = history.iloc[-1]['total_value']
        print(f"Final Portfolio Value: ${final_val:,.2f}")
        print(history.tail())

if __name__ == "__main__":
    main()
