from typing import List, Dict, Any
from dataclasses import dataclass
import logging

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
class HPOJob:
    model_family: str
    priority: str
    n_trials: int
    search_space: Dict[str, Any]
    notes: str

@dataclass
class HPOPlan:
    jobs: List[HPOJob]
    global_notes: str

class LLMHPOPlannerAgent:
    """
    Agent that plans HPO runs using an LLM.
    """
    def __init__(self, llm_client):
        self.llm = llm_client
        from src.prompts.llm_prompts import get_prompt
        self.get_prompt = get_prompt

    def plan(self, plan_input: HPOPlanInput) -> HPOPlan:
        """
        Generate an HPO plan.
        """
        # Prepare data
        past_runs_json = str([r.__dict__ for r in plan_input.past_runs])
        perf_rows_json = str(plan_input.performance_summary[:50])
        
        prompt = self.get_prompt(
            "hpo_budget_planning",
            total_trials=plan_input.total_hpo_budget,
            min_trials=plan_input.per_family_min_trials,
            max_trials=plan_input.per_family_max_trials,
            family_performance_json=perf_rows_json
        )
        
        logger.info("Generating HPO plan")
        response = self.llm.chat(prompt)
        data = response.json() if hasattr(response, 'json') else {}
        
        jobs = [
            HPOJob(
                model_family=j.get("model_family", "unknown"),
                priority=j.get("priority", "low"),
                n_trials=j.get("n_trials", 0),
                search_space=j.get("search_space", {}),
                notes=j.get("notes", "")
            )
            for j in data.get("jobs", [])
        ]
        
        return HPOPlan(
            jobs=jobs,
            global_notes=data.get("global_notes", "")
        )
