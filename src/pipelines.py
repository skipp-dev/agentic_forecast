import logging
import os
import pandas as pd
from .core.run_context import RunContext, RunType
from .graphs.main_graph import create_main_graph
from .graphs.state import GraphState

logger = logging.getLogger(__name__)

def build_initial_state(symbols, config, ctx: RunContext):
    """Build the initial GraphState with run_type from context"""
    return GraphState(
        symbols=symbols,
        config=config,
        run_type=ctx.run_type.value,
        raw_data={},
        features={},
        forecasts={},
        performance_summary=pd.DataFrame(),
        drift_metrics=pd.DataFrame(),
        risk_kpis=pd.DataFrame(),
        anomalies={},
        recommended_actions=[],
        executed_actions=[],
        retrained_models={},
        best_models={},
        errors=[],
        hpo_results={},
        shap_results={},
        analytics_summary=pd.DataFrame(),
        hpo_decision={},
        retraining_history=[],
        guardrail_log=[],
        hpo_triggered=False,
        drift_detected=False,
        edge_index=None,
        node_features=None,
        symbol_to_idx={}
    )

def run_pipeline(ctx: RunContext, symbols: list, config: dict):
    """Common pipeline runner"""
    logger.info(f"Initializing pipeline for {ctx.run_type.value} run (ID: {ctx.run_id})")
    
    # Setup environment based on run type
    if ctx.run_type == RunType.BACKTEST:
        os.environ['SKIP_NEURALFORECAST'] = 'true'
        os.environ['RUN_TYPE'] = 'BACKTEST'
        logger.info("BACKTEST mode: Skipping NeuralForecast imports")
    elif ctx.run_type == RunType.WEEKEND_HPO:
        os.environ['RUN_TYPE'] = 'WEEKEND_HPO'
    else:
        os.environ['RUN_TYPE'] = 'DAILY'

    app = create_main_graph(config)
    initial_state = build_initial_state(symbols, config, ctx)
    
    for output in app.stream(initial_state):
        for key, value in output.items():
            logger.info(f"Output from node '{key}':")
            logger.info("---")
            # logger.info(str(value)) # Reduce verbosity if needed
        logger.info("\n---\n")

    logger.info(f"{ctx.run_type.value} workflow finished.")

def run_daily_pipeline(ctx: RunContext, symbols: list, config: dict):
    run_pipeline(ctx, symbols, config)

def run_weekend_hpo_pipeline(ctx: RunContext, symbols: list, config: dict):
    run_pipeline(ctx, symbols, config)

def run_backtest_pipeline(ctx: RunContext, symbols: list, config: dict):
    run_pipeline(ctx, symbols, config)
