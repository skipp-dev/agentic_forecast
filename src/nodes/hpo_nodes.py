from typing import Dict, Any
from ..agents.hpo_agent import HPOAgent
from ..graphs.state import GraphState

def hpo_node(state: GraphState) -> GraphState:
    """
    Runs the hyperparameter optimization process.
    """
    print("--- Node: Hyperparameter Optimization ---")

    symbols = state.get("symbols", [])
    if not symbols:
        print("No symbols found in state. Skipping HPO.")
        return state

    try:
        hpo_agent = HPOAgent(symbols=symbols)
        hpo_agent.run_hpo_session()
        hpo_results = hpo_agent.results

        state['hpo_results'] = hpo_results
        # Reset the trigger flag to prevent loops
        state['hpo_triggered'] = False
        print("HPO session finished.")
        return state

    except Exception as e:
        print(f"Error during HPO: {e}")
        state['errors'].append(f"HPO Error: {e}")
        # Reset flag even on error
        state['hpo_triggered'] = False
        return state
