import logging
import os
import pandas as pd
from .core.run_context import RunContext, RunType
from .graphs.main_graph import create_main_graph
from .graphs.state import GraphState
from src.monitoring.tracing import setup_tracing

logger = logging.getLogger(__name__)
tracer = setup_tracing("pipeline_orchestrator")

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
        drift_metrics={},
        risk_kpis={},
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
        symbol_to_idx={},
        macro_data={},
        regimes={}
    )

def run_pipeline(ctx: RunContext, symbols: list, config: dict):
    """Common pipeline runner"""
    with tracer.start_as_current_span("run_pipeline") as span:
        span.set_attribute("run_id", ctx.run_id)
        span.set_attribute("run_type", ctx.run_type.value)
        span.set_attribute("symbol_count", len(symbols))
        
        logger.info(f"Initializing pipeline for {ctx.run_type.value} run (ID: {ctx.run_id})")
        
        # Setup environment based on run type
        if ctx.run_type == RunType.BACKTEST:
        os.environ['SKIP_NEURALFORECAST'] = 'true'
        os.environ['RUN_TYPE'] = 'BACKTEST'
        logger.info("BACKTEST mode: Skipping NeuralForecast imports")
        
        # Import BacktestExecutor
        from src.backtesting.backtest_executor import BacktestExecutor
        
        # Determine dates from config or defaults
        start_date = config.get('backtest', {}).get('start_date', '2023-01-01')
        end_date = config.get('backtest', {}).get('end_date', '2023-01-10')
        step_days = config.get('backtest', {}).get('step_days', 1)
        
        app = create_main_graph(config)
        initial_state = build_initial_state(symbols, config, ctx)
        
        executor = BacktestExecutor(app, start_date, end_date, step_days)
        results = executor.run(initial_state)
        executor.save_results()
        
        logger.info("Backtest completed.")
        return

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
