from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import json
from src.configs.llm_prompts import (
    get_prompt, 
    build_hpo_planner_user_prompt, 
    build_llm_messages,
    extract_json_from_response
)

logger = logging.getLogger(__name__)

@dataclass
class HPORun:
    model_family: str
    trial_id: str
    params: Dict[str, Any]
    metric: float
    directional_accuracy: float
    status: str

@dataclass
class HPOPlanInput:
    past_runs: List[HPORun]
    performance_summary: List[Dict[str, Any]]
    total_hpo_budget: int
    per_family_min_trials: int
    per_family_max_trials: int

@dataclass
class HPOPlan:
    symbols_to_focus: List[str] = field(default_factory=list)
    horizons_to_focus: List[int] = field(default_factory=list)
    families_to_prioritize: List[str] = field(default_factory=list)
    per_family_search_spaces: Dict[str, Any] = field(default_factory=dict)
    budget_allocation: Dict[str, int] = field(default_factory=dict)
    symbol_family_overrides: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

class LLMHPOPlannerAgent:
    """
    Agent that plans HPO runs using an LLM.
    """
    def __init__(self, llm_client: Any):
        """
        Initialize the HPO Planner Agent.
        
        Args:
            llm_client: Client for LLM interactions (must support .generate(messages, ...))
        """
        self.llm = llm_client

    def plan(self, plan_input: HPOPlanInput) -> HPOPlan:
        """
        Generate an HPO plan based on past performance and budget constraints.
        """
        # Prepare data for the prompt
        # Convert dataclasses to dicts for JSON serialization
        past_runs_dicts = [r.__dict__ for r in plan_input.past_runs]
        
        # Build the prompts
        system_prompt = get_prompt("hpo_planner")
        user_prompt = build_hpo_planner_user_prompt(
            past_hpo_runs=past_runs_dicts,
            family_performance=plan_input.performance_summary,
            total_trials=plan_input.total_hpo_budget,
            min_trials=plan_input.per_family_min_trials,
            max_trials=plan_input.per_family_max_trials
        )
        
        messages = build_llm_messages(system_prompt, user_prompt)
        
        logger.info("Generating HPO plan with LLM")
        try:
            response = self.llm.generate(messages, temperature=0.2)
            
            data = extract_json_from_response(response)
            
            # Filter and validate keys against the dataclass
            valid_keys = HPOPlan.__annotations__.keys()
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            
            return HPOPlan(**filtered)
            
        except Exception as e:
            logger.error(f"Failed to generate or parse LLM HPO plan: {e}")
            # Return safe fallback plan
            return HPOPlan(
                symbols_to_focus=[],
                horizons_to_focus=[],
                families_to_prioritize=[],
                per_family_search_spaces={},
                budget_allocation={},
                symbol_family_overrides=[],
                notes=f"Failed to parse plan. Error: {str(e)}"
            )
