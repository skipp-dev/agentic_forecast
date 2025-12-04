#!/usr/bin/env python
"""
summarize_capabilities.py

Pretty-print an overview of the Agentic Forecast system:

- Available agents (grouped by type)
- LLM backends + role assignment
- Model families:
    - all available model families
    - active per run_type (DAILY / WEEKEND_HPO / BACKTEST)
    - warnings if a run_type is "BaselineLinear only"
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any

# ──────────────────────────────────────────────────────────────────────────────
# 1. Types / dataclasses
# ──────────────────────────────────────────────────────────────────────────────


class RunType(str, Enum):
    DAILY = "DAILY"
    WEEKEND_HPO = "WEEKEND_HPO"
    BACKTEST = "BACKTEST"


@dataclass
class LLMBackend:
    name: str           # internal id (e.g. "openai_gpt4o_mini")
    model: str          # external model name (e.g. "gpt-4o-mini")
    provider: str       # e.g. "openai"


@dataclass
class ModelPolicySnapshot:
    """Summary of which families are active for a run_type."""
    run_type: RunType
    primary_models: List[str]
    fallback_models: List[str]
    priority_order: List[str]

    @property
    def enabled_families(self) -> List[str]:
        # Union while preserving order from priority_order if present
        if self.priority_order:
            return self.priority_order
        seen = set()
        ordered: List[str] = []
        for name in self.primary_models + self.fallback_models:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered


# ──────────────────────────────────────────────────────────────────────────────
# 2. Helpers to fetch info from your project
# ──────────────────────────────────────────────────────────────────────────────

def load_yaml(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def load_all_agents() -> List[str]:
    """
    Return a flat list of agent display names based on files in src/agents.
    """
    agents_dir = os.path.join("src", "agents")
    agents = []
    if os.path.exists(agents_dir):
        for f in os.listdir(agents_dir):
            if f.endswith("_agent.py") and f != "__init__.py":
                # Convert filename to display name
                name = f.replace("_agent.py", "").replace("_", " ").title() + " Agent"
                agents.append(name)
            elif f == "analytics_explainer.py":
                agents.append("Analytics Explainer Agent")
            elif f == "hpo_planner.py":
                agents.append("HPO Planner Agent")
            elif f == "news_enricher.py":
                agents.append("News Enricher Agent")
            elif f == "policy_optimizer.py":
                pass # Not an agent
            elif f == "strategy_selector.py":
                pass # Not an agent
    return sorted(agents)


def load_llm_backends() -> List[LLMBackend]:
    """
    Return configured LLM backends from config.yaml.
    """
    config = load_yaml("config.yaml")
    backends = []
    if "llm" in config and "backends" in config["llm"]:
        for name, details in config["llm"]["backends"].items():
            backends.append(LLMBackend(name, details.get("model", "unknown"), details.get("provider", "unknown")))
    return backends


def load_llm_role_assignments() -> Dict[str, str]:
    """
    Return mapping role_name -> backend_name from config.yaml.
    """
    config = load_yaml("config.yaml")
    roles = {}
    if "llm" in config and "default_llm_roles" in config["llm"]:
        for role, details in config["llm"]["default_llm_roles"].items():
            roles[role] = details.get("backend", "unknown")
    return roles


def load_available_model_families() -> List[str]:
    """
    All model families that exist in your model zoo.
    """
    return [
        "AutoTFT",
        "AutoNBEATS",
        "AutoNHITS",
        "AutoDLinear",
        "BaselineLinear",
        "graph_stgcnn",
        "Ensemble"
    ]


def load_policy_for_run_type(run_type: RunType) -> ModelPolicySnapshot:
    """
    Build a ModelPolicySnapshot for a given run_type.
    """
    config = load_yaml("config.yaml")
    model_families_config = load_yaml(os.path.join("src", "configs", "model_families.yaml"))
    
    if run_type == RunType.DAILY:
        # DAILY typically uses the main config.yaml settings
        models_cfg = config.get("models", {})
        return ModelPolicySnapshot(
            run_type=run_type,
            primary_models=models_cfg.get("primary", ["BaselineLinear"]),
            fallback_models=models_cfg.get("fallback", ["BaselineLinear"]),
            priority_order=models_cfg.get("priority_order", ["BaselineLinear"]),
        )
    elif run_type == RunType.WEEKEND_HPO:
        # WEEKEND_HPO uses the full zoo defined in model_families.yaml
        # We'll aggregate all primary/secondary from default_policy
        default_policy = model_families_config.get("default_policy", {})
        
        primaries = set()
        secondaries = set()
        baselines = set()
        
        for horizon in ["short_horizon", "medium_horizon", "long_horizon"]:
            h_pol = default_policy.get(horizon, {})
            primaries.update(h_pol.get("primary", []))
            secondaries.update(h_pol.get("secondary", []))
            baselines.update(h_pol.get("baseline", []))
            
        # Construct a logical priority order
        # This is an approximation since it varies by horizon, but gives a good overview
        priority = list(primaries) + list(secondaries) + list(baselines)
        # Remove duplicates while preserving order
        seen = set()
        deduped_priority = []
        for p in priority:
            if p not in seen:
                deduped_priority.append(p)
                seen.add(p)

        return ModelPolicySnapshot(
            run_type=run_type,
            primary_models=list(primaries),
            fallback_models=list(baselines),
            priority_order=deduped_priority,
        )
    else:  # BACKTEST
        # BACKTEST is usually restricted
        return ModelPolicySnapshot(
            run_type=run_type,
            primary_models=["BaselineLinear"],
            fallback_models=["BaselineLinear"],
            priority_order=["BaselineLinear"],
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Classification / pretty printing
# ──────────────────────────────────────────────────────────────────────────────

def classify_agent(name: str) -> str:
    """Roughly group agents by name heuristics."""
    n = name.lower()

    data_kw = ["data agent", "price agent", "alpha vantage", "crypto", "fx", "macro", "fundamentals", "news data", "news enricher"]
    model_kw = ["forecast agent", "graph model", "global model", "ensemble agent",
                "retraining agent", "feature agent", "feature engineer", "hyperparameter", "hpo planner"]
    risk_kw = ["risk", "guardrail", "drift", "health", "quality", "anomaly", "monitoring", "regime"]
    llm_kw = ["llm ", "openai"]
    orch_kw = ["orchestrator", "supervisor", "strategy", "decision", "notification", "reporting", "auto documentation", "analytics explainer"]

    if any(k in n for k in data_kw):
        return "Data & Feeds"
    if any(k in n for k in model_kw):
        return "Modelling & Features"
    if any(k in n for k in risk_kw):
        return "Risk, Guardrails & Monitoring"
    if any(k in n for k in llm_kw):
        return "LLM & Meta-Agents"
    if any(k in n for k in orch_kw):
        return "Orchestration & Strategy"
    return "Other"


def group_agents(names: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for n in sorted(set(names)):
        g = classify_agent(n)
        groups.setdefault(g, []).append(n)
    return groups


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Main reporting
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("================================================================")
    print("                 AGENTIC FORECAST CAPABILITIES                  ")
    print("================================================================")

    # --- Agents ---------------------------------------------------
    agents = load_all_agents()
    grouped = group_agents(agents)

    print_section("--- AVAILABLE AGENTS ---")
    for group_name in sorted(grouped.keys()):
        print(f"\n{group_name}:")
        for name in grouped[group_name]:
            print(f"  - {name}")

    # --- LLM configuration ----------------------------------------
    backends = load_llm_backends()
    roles = load_llm_role_assignments()

    print_section("--- LLM CONFIGURATION ---")
    print("Backends:")
    for b in backends:
        print(f"  - {b.name:<18} : {b.model} ({b.provider})")

    print("\nRoles Assignment:")
    for role, backend in roles.items():
        print(f"  - {role:<20} : {backend}")

    # --- Model families & policies --------------------------------
    print_section("--- MODEL FAMILIES ---")

    all_families = load_available_model_families()
    print("Available families:")
    print("  " + ", ".join(all_families))

    for rt in RunType:
        policy = load_policy_for_run_type(rt)
        enabled = policy.enabled_families
        baseline_only = set(m.lower() for m in enabled) <= {"baselinelinear"}

        print(f"\nRun Type: {rt.value}")
        print(f"  Primary Models : {', '.join(policy.primary_models) or 'None'}")
        print(f"  Fallback Models: {', '.join(policy.fallback_models) or 'None'}")
        print(f"  Priority Order : {', '.join(policy.priority_order) or 'None'}")
        print(f"  Enabled Families: {', '.join(enabled) or 'None'}")

        if baseline_only:
            print("  ⚠ WARNING: This run type is configured as 'BaselineLinear-only'.")
            print("    Consider enabling AutoNHITS/AutoTFT/AutoNBEATS/AutoDLinear/graph_stgcnn for non-backtest runs.")

    print()
    print("================================================================")


if __name__ == "__main__":
    main()
