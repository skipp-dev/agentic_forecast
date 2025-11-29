import pandas as pd
import logging

from ..graphs.state import GraphState
from agents.feature_agent import FeatureAgent
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

logger = logging.getLogger(__name__)

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
                features[symbol] = features_df
                logger.info(f"Generated {features_df.shape[1]} features for {symbol}")
            except Exception as e:
                logger.error(f"Failed to generate features for {symbol}: {e}")
                # Fall back to original data if feature engineering fails
                features[symbol] = data

        state['features'] = features
        return state

    except ImportError as e:
        logger.warning(f"FeatureEngineer import failed ({e}), falling back to basic FeatureAgent")
        # Fallback to original simple feature agent
        agent = FeatureAgent()
        state['features'] = agent.generate_features(state['raw_data'])
        return state

from ..agents.analytics_agent import AnalyticsAgent

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

    state['analytics_summary'] = analytics_summary

    if analytics_summary is not None and not analytics_summary.empty:
        best_rows = analytics_summary.sort_values('mape').groupby('symbol').first()
        logger.info("Ran analytics. Best observed MAPE per symbol:")
        for symbol, row in best_rows.iterrows():
            logger.info(f"   • {symbol}: {row['model_family']} @ {row['mape']:.3f} MAPE")
    else:
        logger.warning("Analytics summary is empty (no overlapping forecasts/actuals).")

    return state

from ..agents.decision_agent import DecisionAgent

def decision_agent_node(state: GraphState, config: dict) -> GraphState:
    """
    Runs the decision agent to select the best model and recommend actions.
    """
    logger.info("--- Node: Decision Agent ---")
    agent = DecisionAgent(config.get('hpo', {}))
    best_models = agent.select_best_model(state['performance_summary'], state.get('anomalies', {}))
    
    recommended_actions = []
    for symbol, model in best_models.items():
        model_family = model.get('model_family', 'UnknownModel')
        action = f"Promote {model_family} for {symbol}"
        mape = model.get('mape')
        if mape is not None and not pd.isna(mape):
            action += f" (MAPE {float(mape):.3f})"
        recommended_actions.append(action)

    # Check if HPO is needed
    hpo_decision = agent.should_run_hpo(
        state['performance_summary'],
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

from agents.openai_research_agent import OpenAIResearchAgent

def news_data_node(state: GraphState) -> GraphState:
    """
    Runs the news data agent to gather external market intelligence.
    """
    logger.info("--- Node: News Data Agent ---")
    
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

from agents.llm_analytics_agent import LLMAnalyticsExplainerAgent, AnalyticsInput

def llm_analytics_node(state: GraphState) -> GraphState:
    """
    Runs the LLM analytics explainer agent to provide natural language insights.
    """
    logger.info("--- Node: LLM Analytics Agent ---")
    
    try:
        from src.llm.llm_factory import create_analytics_explainer_llm
        llm_client = create_analytics_explainer_llm()
        
        agent = LLMAnalyticsExplainerAgent(llm_client)
        
        # Prepare input data
        performance_summary = state.get('analytics_summary', pd.DataFrame()).to_dict('records')
        drift_events = []  # Could be populated from drift detection results
        
        analytics_input = AnalyticsInput(
            performance_summary=performance_summary,
            drift_events=drift_events
        )
        
        # Get LLM analysis
        recommendation = agent.analyze(analytics_input)
        
        # Store results
        state['llm_analytics_summary'] = recommendation.summary_text
        state['llm_actions'] = recommendation.actions
        state['llm_notes'] = recommendation.notes_for_humans
        
        logger.info("Generated LLM analytics insights.")
        
    except Exception as e:
        logger.error(f"LLM analytics failed: {e}")
        state['llm_analytics_summary'] = "LLM analysis unavailable"
    
    return state

from agents.llm_hpo_planner_agent import LLMHPOPlannerAgent, HPOPlanInput, HPORun

def llm_hpo_planning_node(state: GraphState) -> GraphState:
    """
    Runs the LLM HPO planner agent to optimize hyperparameter search.
    """
    logger.info("--- Node: LLM HPO Planning Agent ---")
    
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
