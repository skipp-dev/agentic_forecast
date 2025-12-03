from langgraph.graph import StateGraph, END
from functools import partial
from .state import GraphState
from ..nodes import (
    data_nodes_optimized as data_nodes, 
    agent_nodes, 
    execution_nodes, 
    utility_nodes, 
    monitoring_nodes, 
    retraining_nodes,
    hpo_nodes,
    ensemble_nodes,
    anomaly_detection_nodes,
    reporting_nodes
)

def should_continue(state: GraphState) -> str:
    """
    Determines whether to continue with the main workflow or trigger specialized loops.
    Includes loop prevention and continuous learning triggers.
    """
    # Check for loop prevention - limit retraining/HPO attempts
    retraining_history = state.get('retraining_history', [])
    hpo_results = state.get('hpo_results', {})

    max_retraining_attempts = 2  # Allow max 2 retraining cycles per run
    max_hpo_attempts = 1        # Allow max 1 HPO cycle per run

    retraining_attempts = len(retraining_history)
    hpo_attempts = len(hpo_results) if hpo_results else 0

    # Check for continuous learning triggers from reports
    continuous_learning_applied = state.get('continuous_learning_applied', False)
    learning_feedback = state.get('learning_feedback', {})

    if continuous_learning_applied and learning_feedback:
        # Check if high-priority actions were triggered
        decisions_updated = learning_feedback.get('decisions_updated', [])
        for decision in decisions_updated:
            if decision['type'] == 'retraining' and retraining_attempts < max_retraining_attempts:
                print(f"🔄 Continuous learning triggering retraining for {decision.get('symbol', 'all')}")
                return "retrain"
            elif decision['type'] == 'hpo' and hpo_attempts < max_hpo_attempts:
                print(f"🔄 Continuous learning triggering HPO for {decision.get('symbol', 'all')}")
                return "hpo"

    # Check for WEEKEND_HPO run type
    run_type = state.get('run_type', 'DAILY')
    if run_type == 'WEEKEND_HPO' and hpo_attempts < max_hpo_attempts:
        print("🔄 WEEKEND_HPO triggering HPO run")
        return "hpo"

    # Original logic for drift and HPO triggers
    if state.get('hpo_triggered') and hpo_attempts < max_hpo_attempts:
        return "hpo"
    elif state.get('drift_detected') and retraining_attempts < max_retraining_attempts:
        return "retrain"
    else:
        # Reset flags to prevent future triggers in this run
        state['hpo_triggered'] = False
        state['drift_detected'] = False
        state['continuous_learning_applied'] = False
        return "continue"

def create_main_graph(config: dict):
    """
    Creates the main forecasting graph.
    """
    graph = StateGraph(GraphState)

    # Partially apply the config to the nodes that need it
    decision_agent_node_with_config = partial(agent_nodes.decision_agent_node, config=config)
    guardrail_agent_node_with_config = partial(agent_nodes.guardrail_agent_node, config=config)

    # Add nodes
    graph.add_node("load_data", data_nodes.load_data_node)
    graph.add_node("news_data", agent_nodes.news_data_node)
    graph.add_node("enrich_news", agent_nodes.llm_news_enrichment_node)
    graph.add_node("construct_graph", agent_nodes.graph_construction_node)
    graph.add_node("detect_drift", monitoring_nodes.drift_detection_node)
    graph.add_node("detect_anomalies", anomaly_detection_nodes.anomaly_detection_node)
    graph.add_node("assess_risk", monitoring_nodes.risk_assessment_node)
    graph.add_node("llm_hpo_planning", agent_nodes.llm_hpo_planning_node)
    graph.add_node("run_hpo", hpo_nodes.hpo_node)
    graph.add_node("retrain_model", retraining_nodes.retraining_node)
    graph.add_node("generate_features", agent_nodes.feature_agent_node)
    graph.add_node("generate_forecasts", execution_nodes.forecasting_node)
    graph.add_node("create_ensemble", ensemble_nodes.ensemble_node)
    graph.add_node("run_analytics", agent_nodes.analytics_agent_node)
    graph.add_node("llm_analytics", agent_nodes.llm_analytics_node)
    graph.add_node("interpret_forecasts", agent_nodes.forecast_agent_node)
    graph.add_node("make_decisions", decision_agent_node_with_config)
    graph.add_node("apply_guardrails", guardrail_agent_node_with_config)
    graph.add_node("execute_actions", execution_nodes.action_executor_node)
    # graph.add_node("run_explainability", agent_nodes.explainability_agent_node)
    graph.add_node("generate_report", reporting_nodes.generate_report_node)
    graph.add_node("auto_documentation", agent_nodes.auto_documentation_node)

    # Define edges
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "news_data")
    graph.add_edge("news_data", "enrich_news")
    graph.add_edge("enrich_news", "construct_graph")
    graph.add_edge("construct_graph", "detect_drift")
    graph.add_edge("detect_drift", "detect_anomalies")
    graph.add_edge("detect_anomalies", "assess_risk")
    
    # Conditional edge after risk assessment
    graph.add_conditional_edges(
        "assess_risk",
        should_continue,
        {
            "continue": "generate_features",
            "retrain": "retrain_model",
            "hpo": "llm_hpo_planning",
        },
    )
    
    graph.add_edge("llm_hpo_planning", "run_hpo")
    graph.add_edge("retrain_model", "generate_features") # Continue pipeline after retraining
    graph.add_edge("run_hpo", "generate_features") # Continue pipeline after HPO

    graph.add_edge("generate_features", "generate_forecasts")
    graph.add_edge("generate_forecasts", "create_ensemble")
    graph.add_edge("create_ensemble", "run_analytics")
    graph.add_edge("run_analytics", "llm_analytics")
    graph.add_edge("llm_analytics", "interpret_forecasts")
    graph.add_edge("interpret_forecasts", "make_decisions")
    graph.add_edge("make_decisions", "apply_guardrails")
    # graph.add_edge("apply_guardrails", "run_explainability")
    # graph.add_edge("run_explainability", "execute_actions")
    graph.add_edge("apply_guardrails", "execute_actions")
    graph.add_edge("execute_actions", "generate_report")
    graph.add_edge("generate_report", "auto_documentation")
    graph.add_edge("auto_documentation", END)

    return graph.compile()
