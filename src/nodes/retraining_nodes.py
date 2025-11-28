from typing import Dict, Any
from src.agents.retraining_agent import RetrainingAgent
from src.graphs.state import GraphState

def retraining_node(state: GraphState) -> GraphState:
    """
    Retrains the model based on the detected drift and updates the state.
    """
    print("--- Node: Retraining ---")
    retraining_agent = RetrainingAgent(model=None)
    retraining_results = retraining_agent.run(state)
    state.update(retraining_results)
    return state
