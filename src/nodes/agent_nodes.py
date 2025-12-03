import pandas as pd
import logging

from ..graphs.state import GraphState
from ..agents.feature_agent import FeatureAgent
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

logger = logging.getLogger(__name__)

# Skip LLM agent imports for BACKTEST mode to avoid import hangs
_SKIP_LLM_AGENTS = os.environ.get('RUN_TYPE', '').upper() == 'BACKTEST'

def feature_agent_node(state: GraphState) -> GraphState:
    """
    Runs the feature engineering agent to generate features from the raw data.
    """
    logger.info("--- Node: Feature Agent ---")

    # Import FeatureEngineer locally to avoid module-level import issues
    try:
        from pipelines.run_features import FeatureEngineer
        agent = FeatureEngineer()

        # Process all symbols in the raw_data dictionary
        features = {}
        for symbol, data in state['raw_data'].items():
            try:
                features_df = agent.engineer_features_for_symbol(symbol, data, experiment="default")
                # Convert DataFrame to serializable format for LangSmith tracing
                features_df.index = features_df.index.astype(str)
                features[symbol] = features_df.to_dict('index')
                logger.info(f"Generated {features_df.shape[1]} features for {symbol}")
            except Exception as e:
                logger.error(f"Failed to generate features for {symbol}: {e}")
                # Fall back to original data if feature engineering fails
                data.index = data.index.astype(str)
                features[symbol] = data.to_dict('index')

        state['features'] = features
        return state

    except ImportError as e:
        logger.warning(f"FeatureEngineer import failed ({e}), falling back to basic FeatureAgent")
        # Fallback to original simple feature agent
        features = agent.generate_features(state['raw_data'])
        # Convert DataFrames to serializable format
        serializable_features = {}
        for symbol, df in features.items():
            df.index = df.index.astype(str)
            serializable_features[symbol] = df.to_dict('index')
        state['features'] = serializable_features
        return state

from ..agents.analytics_explainer import AnalyticsAgent

def analytics_agent_node(state: GraphState) -> GraphState:
    """
    Runs the analytics agent to calculate performance metrics and other KPIs.
    """
    logger.info("--- Node: Analytics Agent ---")
    agent = AnalyticsAgent()

    analytics_summary = agent.calculate_performance_summary(
        state.get('forecasts', {}),
        state.get('raw_data', {})
    )

    # Convert DataFrame to serializable format for LangSmith tracing
    if analytics_summary is not None and not analytics_summary.empty:
        state['analytics_summary'] = analytics_summary.to_dict('records')
    else:
        state['analytics_summary'] = []

    if analytics_summary is not None and not analytics_summary.empty:
        best_rows = analytics_summary.sort_values('mape').groupby('symbol').first()
        logger.info("Ran analytics. Best observed MAPE per symbol:")
        for symbol, row in best_rows.iterrows():
            logger.info(f"   • {symbol}: {row['model_family']} @ {row['mape']:.3f} MAPE")
    else:
        logger.warning("Analytics summary is empty (no overlapping forecasts/actuals).")

    return state

from ..agents.decision_agent import DecisionAgent
from src.analytics.trust_score import TrustScoreCalculator

def decision_agent_node(state: GraphState, config: dict) -> GraphState:
    """
    Runs the decision agent to select the best model and recommend actions.
    Also calculates Trust Scores.
    """
    logger.info("--- Node: Decision Agent ---")
    agent = DecisionAgent(config.get('hpo', {}))
    trust_calculator = TrustScoreCalculator()
    
    # Convert analytics_summary back to DataFrame for the agent
    analytics_summary = pd.DataFrame(state.get('analytics_summary', []))
    
    # Calculate Trust Scores
    trust_scores = trust_calculator.calculate_trust_scores(
        performance_summary=analytics_summary,
        risk_kpis=state.get('risk_kpis', {}),
        guardrail_flags=state.get('guardrail_log', []),
        drift_metrics=state.get('drift_metrics', {})
    )
    state['trust_scores'] = trust_scores
    logger.info(f"Calculated trust scores for {len(trust_scores)} symbols")

    # Save trust scores to JSON for Prometheus exporter
    try:
        output_path = "data/metrics/trust_scores.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        import json
        with open(output_path, "w") as f:
            json.dump(trust_scores, f, indent=2)
        logger.info(f"Saved trust scores to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save trust scores: {e}")
    
    best_models = agent.select_best_model(analytics_summary, state.get('anomalies', {}))
    
    recommended_actions = []
    for symbol, model in best_models.items():
        model_family = model.get('model_family', 'UnknownModel')
        
        # Use Trust Score to determine action
        trust_score = trust_scores.get(symbol, 0.5)
        trading_decision = agent.get_trading_decision(symbol, trust_score)
        
        if trading_decision['auto_trade_allowed']:
            action = f"Promote {model_family} for {symbol} (Trust: {trust_score:.2f}, Size: {trading_decision['position_size_multiplier']}x)"
        else:
            action = f"Hold {model_family} for {symbol} (Trust: {trust_score:.2f} too low)"
            
        mape = model.get('mape')
        if mape is not None and not pd.isna(mape):
            action += f" [MAPE {float(mape):.3f}]"
        recommended_actions.append(action)

    # Check if HPO is needed
    hpo_decision = agent.should_run_hpo(
        analytics_summary,
        state.get('anomalies'),
        state.get('market_conditions'),
        state.get('raw_data')
    )
    state['hpo_decision'] = hpo_decision

    if hpo_decision.get('should_run'):
        state['hpo_triggered'] = True
        reason = hpo_decision.get('reason', 'performance degradation detected')
        recommended_actions.append(f"Trigger HPO ({reason})")
    else:
        state['hpo_triggered'] = False
        
    state['recommended_actions'] = recommended_actions
    state['best_models'] = best_models
    logger.info("Made decisions.")
    return state

from ..agents.guardrail_agent import GuardrailAgent

def guardrail_agent_node(state: GraphState, config: dict) -> GraphState:
    """
    Runs the guardrail agent to vet the recommended actions.
    """
    logger.info("--- Node: Guardrail Agent ---")
    agent = GuardrailAgent(config.get('anomaly_detection', {}))
    result = agent.run(state)
    guardrail_entries = result.pop('guardrail_log', [])
    state.update(result)
    log = state.get('guardrail_log', [])
    log.extend(guardrail_entries)
    state['guardrail_log'] = log
    logger.info("Applied guardrails.")
    logger.debug(f"Guardrail returning state with keys: {list(state.keys())}")
    return state

# from ..agents.explainability_agent import ExplainabilityAgent

# Conditional import for ModelZoo to avoid neuralforecast imports in BACKTEST mode
if not _SKIP_LLM_AGENTS:
    from models.model_zoo import ModelZoo

# def explainability_agent_node(state: GraphState) -> GraphState:
#     """
#     Runs the explainability agent to generate SHAP values for the best models.
#     """
#     logger.info("--- Node: Explainability Agent ---")
#     logger.debug(f"Explainability node called with state keys: {list(state.keys())}")
#     logger.debug(f"Best models available: {list(state.get('best_models', {}).keys())}")
    
#     try:
#         # Import here to ensure availability
#         from neuralforecast import NeuralForecast
#         from neuralforecast.models import NLinear
#         import numpy as np
        
#         best_models = state.get('best_models', {})
#         if not best_models:
#             logger.info("No best models selected, skipping explainability.")
#             return state

#         model_zoo = ModelZoo()
#         shap_results = {}
        
#         for symbol, model_info in best_models.items():
#             model_id = model_info.get('model_id')
#             model_family = model_info.get('model_family', 'BaselineLinear')
            
#             try:
#                 # Get features for this symbol first
#                 symbol_features = state['features'][symbol]
                
#                 if model_id:
#                     # Load the trained model
#                     nf_model = model_zoo.load_model(model_id)
#                 else:
#                     # Create a fallback model for explainability
#                     logger.info(f"No trained model for {symbol}, using fallback for SHAP analysis.")
#                     # For fallback, we'll create a simple sklearn model that can be explained
#                     from sklearn.linear_model import LinearRegression
#                     model = LinearRegression()
                    
#                     # Fit the model on a small sample of the features data
#                     sample_data = symbol_features.head(50).copy()  # Use more data for fitting
#                     sample_data = sample_data.reset_index(drop=True)
#                     sample_data['y'] = np.random.normal(100, 10, len(sample_data))  # Dummy target for fitting
#                     # Use only numeric columns
#                     numeric_cols = symbol_features.select_dtypes(include=[np.number]).columns
#                     model.fit(sample_data[numeric_cols], sample_data['y'])
                    
#                     model_family = 'LinearRegression'
#                     nf_model = model
                
#                 # Create explainability agent
#                 numeric_cols = symbol_features.select_dtypes(include=[np.number]).columns
#                 agent = ExplainabilityAgent(nf_model, model_family, numeric_cols.tolist())
                
#                 # Generate SHAP explanations
#                 shap_results[symbol] = agent.explain(symbol_features)
                
#             except Exception as e:
#                 logger.error(f"Error generating SHAP for {symbol}: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue

#         state['shap_results'] = shap_results
#         logger.info("Generated SHAP explanations.")
#         return state
#     except Exception as e:
#         logger.error(f"Critical error in explainability node: {e}")
#         import traceback
#         traceback.print_exc()
#         return state

from ..agents.graph_construction_agent import GraphConstructionAgent

def graph_construction_node(state: GraphState) -> GraphState:
    """
    Runs the graph construction agent to build the stock relationship graph.
    """
    logger.info("--- Node: Graph Construction Agent ---")
    
    # Define stock metadata (this could be loaded from a config or database)
    stock_metadata = {
        'AAPL': {'sector': 'Technology'},
        'NVDA': {'sector': 'Technology'},
        'TSLA': {'sector': 'Automotive'},
        # Add more stocks as needed - for now, we'll use a default sector for unknown symbols
    }
    
    # Add default metadata for any symbols not in the predefined metadata
    symbols = list(state['raw_data'].keys())
    for symbol in symbols:
        if symbol not in stock_metadata:
            stock_metadata[symbol] = {'sector': 'Unknown'}
    
    agent = GraphConstructionAgent(stock_metadata)
    
    edge_index, symbol_to_idx = agent.create_graph(symbols)
    
    state['edge_index'] = edge_index
    state['symbol_to_idx'] = symbol_to_idx
    
    logger.info("Constructed stock relationship graph.")
    return state

import os

# Skip LLM agent imports for BACKTEST mode to avoid import hangs
_SKIP_LLM_AGENTS = os.environ.get('RUN_TYPE', '').upper() == 'BACKTEST'

if not _SKIP_LLM_AGENTS:
    from ..agents.research_agent import OpenAIResearchAgent

def news_data_node(state: GraphState) -> GraphState:
    """
    Runs the news data agent to gather external market intelligence.
    """
    logger.info("--- Node: News Data Agent ---")
    
    if _SKIP_LLM_AGENTS:
        logger.info("Skipping news data collection for BACKTEST mode")
        state['news_insights'] = None
        return state
    
    try:
        agent = OpenAIResearchAgent()
        symbols = state.get('symbols', [])
        
        # Conduct research for a subset of symbols to avoid API limits
        research_symbols = symbols[:10]  # Limit to first 10 symbols
        
        insights = agent.conduct_market_research(symbols=research_symbols, days_back=7)
        
        # Store news insights in state
        state['news_insights'] = insights
        state['market_sentiment'] = insights.market_sentiment
        state['key_news'] = [article.__dict__ for article in insights.key_news]
        
        logger.info(f"Gathered news insights for {len(research_symbols)} symbols.")
        
    except Exception as e:
        logger.error(f"News data collection failed: {e}")
        state['news_insights'] = None
    
    return state

if not _SKIP_LLM_AGENTS:
    from ..agents.llm_news_agent import LLMNewsFeatureAgent, RawNewsItem, EnrichedNewsFeature

def llm_news_enrichment_node(state: GraphState) -> GraphState:
    """
    Runs the LLM news feature enrichment agent to add structured features to news articles.
    """
    logger.info("--- Node: LLM News Enrichment Agent ---")

    if _SKIP_LLM_AGENTS:
        logger.info("Skipping LLM news enrichment for BACKTEST mode")
        state['enriched_news'] = []
        return state

    try:
        from src.llm.llm_factory import create_news_features_llm
        llm_client = create_news_features_llm()

        agent = LLMNewsFeatureAgent(llm_client)

        # Get news articles from previous research
        news_insights = state.get('news_insights')
        if not news_insights or not hasattr(news_insights, 'key_news'):
            logger.warning("No news insights available for enrichment")
            state['enriched_news'] = []
            return state

        # Convert ResearchInsights articles to RawNewsItem format
        enriched_features = []

        for article in news_insights.key_news[:10]:  # Limit to avoid rate limits
            try:
                # Convert article to RawNewsItem format
                raw_item = RawNewsItem(
                    symbol=article.symbols[0] if article.symbols else "UNKNOWN",
                    timestamp=article.published_at,
                    headline=article.title,
                    body=article.content,
                    provider=article.source
                )

                # Enrich the news item
                enriched_feature = agent.enrich_item(raw_item)
                enriched_features.append(enriched_feature.__dict__)

                logger.info(f"Enriched news for {raw_item.symbol}")

            except Exception as e:
                logger.warning(f"Failed to enrich news item: {e}")
                continue

        # Store enriched news features
        state['enriched_news'] = enriched_features

        logger.info(f"Enriched {len(enriched_features)} news items with structured features")

    except Exception as e:
        logger.error(f"LLM news enrichment failed: {e}")
        state['enriched_news'] = []

    return state

if not _SKIP_LLM_AGENTS:
    from ..agents.analytics_explainer import LLMAnalyticsExplainerAgent, AnalyticsInput

def llm_analytics_node(state: GraphState) -> GraphState:
    """
    Runs the LLM analytics explainer agent to provide natural language insights.
    Now uses the new orchestrator for JSON + Markdown reports with LangSmith tracing.
    """
    logger.info("--- Node: LLM Analytics Agent ---")

    if _SKIP_LLM_AGENTS:
        logger.info("Skipping LLM analytics for BACKTEST mode")
        state['llm_analytics_summary'] = "LLM analysis skipped for backtest"
        state['llm_actions'] = []
        state['llm_notes'] = {}
        return state

    try:
        from src.analytics.llm_analytics_orchestrator import run_llm_analytics_explainer

        # Run the full orchestrator: health → LLM → JSON + Markdown reports
        explanation = run_llm_analytics_explainer()

        # Store results in state for downstream nodes
        state['llm_analytics_summary'] = explanation.get('global_summary', 'LLM analysis completed')
        state['llm_actions'] = explanation.get('recommendations', [])
        state['llm_notes'] = explanation  # Store full explanation dict

        logger.info("Generated LLM analytics insights with JSON/Markdown reports and LangSmith tracing.")

    except Exception as e:
        logger.error(f"LLM analytics failed: {e}")
        state['llm_analytics_summary'] = "LLM analysis unavailable"
        state['errors'].append(f"LLM analytics error: {e}")

    return state

if not _SKIP_LLM_AGENTS:
    from ..agents.llm_hpo_planner_agent import LLMHPOPlannerAgent, HPOPlanInput, HPORun

def llm_hpo_planning_node(state: GraphState) -> GraphState:
    """
    Runs the LLM HPO planner agent to optimize hyperparameter search.
    """
    logger.info("--- Node: LLM HPO Planning Agent ---")
    
    if _SKIP_LLM_AGENTS:
        logger.info("Skipping LLM HPO planning for BACKTEST mode")
        state['llm_hpo_plan'] = None
        return state
    
    try:
        from src.llm.llm_factory import create_hpo_planner_llm
        llm_client = create_hpo_planner_llm()
        
        agent = LLMHPOPlannerAgent(llm_client)
        
        # Prepare input data
        past_runs = []  # Could be populated from previous HPO results
        performance_summary = state.get('analytics_summary', pd.DataFrame()).to_dict('records')
        
        plan_input = HPOPlanInput(
            past_runs=past_runs,
            performance_summary=performance_summary,
            total_hpo_budget=100,
            per_family_min_trials=5,
            per_family_max_trials=50
        )
        
        # Get HPO plan
        hpo_plan = agent.plan(plan_input)
        
        # Store results
        state['llm_hpo_plan'] = hpo_plan
        
        logger.info("Generated LLM HPO plan.")
        
    except Exception as e:
        logger.error(f"LLM HPO planning failed: {e}")
    
    return state

from ..agents.forecast_agent import ForecastAgent

def forecast_agent_node(state: GraphState) -> GraphState:
    """
    Runs the forecast agent to interpret raw forecasts and produce risk-aware JSON summaries.
    """
    logger.info("--- Node: Forecast Agent ---")

    try:
        agent = ForecastAgent()

        # Get forecasts and analytics data
        forecasts = state.get('forecasts', {})
        analytics_summary_data = state.get('analytics_summary', [])
        analytics_summary = pd.DataFrame(analytics_summary_data)

        # Update policy based on performance
        agent.update_policy_from_performance(analytics_summary_data)

        # Get guardrail information from state
        guardrail_log = state.get('guardrail_log', [])
        guardrail_flags = []

        # Extract active guardrail flags from the log
        for entry in guardrail_log:
            if isinstance(entry, dict) and 'flag' in entry:
                guardrail_flags.append(entry['flag'])
            elif isinstance(entry, str):
                # Try to extract flags from string entries
                if 'shock_regime' in entry.lower():
                    guardrail_flags.append('shock_regime_active')
                elif 'high_error' in entry.lower():
                    guardrail_flags.append('high_error_recently')
                elif 'news_shock' in entry.lower():
                    guardrail_flags.append('news_shock_active')
                elif 'drift' in entry.lower():
                    guardrail_flags.append('data_drift_suspected')

        # Process each symbol's forecasts
        interpreted_forecasts = {}

        for symbol in state.get('symbols', []):
            symbol_forecasts = forecasts.get(symbol, [])
            if not symbol_forecasts:
                logger.warning(f"No forecasts available for {symbol}")
                continue

            # Get error metrics for this symbol from analytics
            symbol_metrics = {}
            if not analytics_summary.empty:
                symbol_data = analytics_summary[analytics_summary['symbol'] == symbol]
                if not symbol_data.empty:
                    # Use the best performing model's metrics
                    best_row = symbol_data.loc[symbol_data['mape'].idxmin()]
                    symbol_metrics = {
                        'directional_accuracy': best_row.get('directional_accuracy', 0),
                        'smape': best_row.get('smape', 0),
                        'mae': best_row.get('mae', 0),
                        'mae_vs_baseline': best_row.get('mae_vs_baseline', 1.0)
                    }

            # Create regime info
            regime_info = {
                'guardrail_flags': guardrail_flags
            }

            # Interpret forecasts for this symbol
            try:
                interpretation = agent.interpret_forecasts(
                    symbol=symbol,
                    forecasts=symbol_forecasts,
                    error_metrics=symbol_metrics,
                    regime_and_guardrail_info=regime_info
                )
                interpreted_forecasts[symbol] = interpretation
                logger.info(f"Interpreted forecasts for {symbol}")
            except Exception as e:
                logger.error(f"Failed to interpret forecasts for {symbol}: {e}")
                continue

        # Store interpreted forecasts in state
        state['interpreted_forecasts'] = interpreted_forecasts

        logger.info(f"Forecast agent completed interpretation for {len(interpreted_forecasts)} symbols")

    except Exception as e:
        logger.error(f"Forecast agent failed: {e}")
        state['interpreted_forecasts'] = {}
        state['errors'].append(f"Forecast agent error: {e}")

    return state
