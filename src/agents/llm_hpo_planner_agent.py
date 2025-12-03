from typing import List, Dict, Any
from dataclasses import dataclass
import logging
import json
import re

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
        response = self.llm.generate(prompt, temperature=0.2)
        
        try:
            # Extract JSON from code block if present
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find the first { and last }
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1:
                    json_str = response[start:end+1]
                else:
                    json_str = response
                
            data = json.loads(json_str)
            
            jobs = []
            for job_data in data.get('jobs', []):
                jobs.append(HPOJob(
                    model_family=job_data.get('model_family', 'Unknown'),
                    priority=job_data.get('priority', 'medium'),
                    n_trials=job_data.get('n_trials', 0),
                    search_space=job_data.get('search_space', {}),
                    notes=job_data.get('notes', '')
                ))
                
            return HPOPlan(
                jobs=jobs,
                global_notes=data.get('global_notes', '')
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM HPO plan: {e}")
            return HPOPlan(
                jobs=[],
                global_notes=f"Failed to parse plan. Raw response: {response}"
            )
