# src/llm/llm_factory.py
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict
import logging
import time
from collections import defaultdict

from src.services.llm_client import OpenAILLMClient

logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# LLM Usage Monitoring
_llm_usage_stats = defaultdict(lambda: {
    'calls': 0,
    'total_tokens': 0,
    'total_cost': 0.0,
    'last_used': None,
    'errors': 0
})


def get_llm_usage_stats():
    """
    Get current LLM usage statistics across all roles.
    """
    return dict(_llm_usage_stats)


def reset_llm_usage_stats():
    """
    Reset LLM usage statistics.
    """
    global _llm_usage_stats
    _llm_usage_stats = defaultdict(lambda: {
        'calls': 0,
        'total_tokens': 0,
        'total_cost': 0.0,
        'last_used': None,
        'errors': 0
    })


def _record_llm_usage(role_name: str, tokens: int = 0, cost: float = 0.0, error: bool = False):
    """
    Record LLM usage for monitoring.
    """
    stats = _llm_usage_stats[role_name]
    stats['calls'] += 1
    stats['total_tokens'] += tokens
    stats['total_cost'] += cost
    stats['last_used'] = time.time()
    if error:
        stats['errors'] += 1

    # Log usage for monitoring
    if error:
        logger.warning(f"LLM usage - Role: {role_name}, Error recorded")
    else:
        logger.debug(f"LLM usage - Role: {role_name}, Tokens: {tokens}, Cost: ${cost:.4f}")


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).parent.parent.parent / "config.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return data or {}


def _get_llm_backend_config(backend_name: str) -> Dict[str, Any]:
    cfg = _load_config()
    llm_cfg = cfg.get("llm", {})
    backends = llm_cfg.get("backends", {})
    if backend_name not in backends:
        raise ValueError(f"Unknown LLM backend: {backend_name}")
    return backends[backend_name]


def _get_backend_for_role(role_name: str) -> Dict[str, Any]:
    cfg = _load_config()
    llm_cfg = cfg.get("llm", {})
    roles = llm_cfg.get("default_llm_roles", {})
    role_cfg = roles.get(role_name)
    if role_cfg is None:
        raise ValueError(f"No LLM role configured for: {role_name}")
    backend_name = role_cfg["backend"]
    return _get_llm_backend_config(backend_name)


def create_llm_for_role(role_name: str):
    """
    Factory: given a logical role (analytics_explainer, hpo_planner, …),
    return an LLM client instance.
    """
    backend_cfg = _get_backend_for_role(role_name)
    provider = backend_cfg["provider"]

    if provider == "openai":
        model = backend_cfg["model"]
        base_url = backend_cfg.get("base_url")
        client = OpenAILLMClient(model=model, base_url=base_url)
        # Wrap the client to add usage tracking
        return _TrackedLLMClient(client, role_name)

    # You can add "smith", "local", etc. later
    raise ValueError(f"Unsupported LLM provider: {provider}")


class _TrackedLLMClient:
    """
    Wrapper around LLM clients to add usage tracking.
    """

    def __init__(self, client, role_name: str):
        self._client = client
        self._role_name = role_name

    def complete(self, *args, **kwargs):
        """
        Track usage for completion calls.
        """
        try:
            result = self._client.complete(*args, **kwargs)
            # Estimate token usage (rough approximation)
            prompt_tokens = len(str(args) + str(kwargs)) // 4  # Rough token estimation
            completion_tokens = len(result) // 4
            total_tokens = prompt_tokens + completion_tokens

            # Estimate cost (rough approximation for OpenAI)
            cost_per_token = 0.00015  # Rough average for GPT models
            estimated_cost = total_tokens * cost_per_token

            _record_llm_usage(self._role_name, tokens=total_tokens, cost=estimated_cost)
            return result
        except Exception as e:
            _record_llm_usage(self._role_name, error=True)
            raise

    def chat(self, *args, **kwargs):
        """
        Track usage for chat calls.
        """
        try:
            result = self._client.chat(*args, **kwargs)
            # Estimate token usage (rough approximation)
            prompt_tokens = len(str(args) + str(kwargs)) // 4
            completion_tokens = len(str(result)) // 4
            total_tokens = prompt_tokens + completion_tokens

            cost_per_token = 0.00015
            estimated_cost = total_tokens * cost_per_token

            _record_llm_usage(self._role_name, tokens=total_tokens, cost=estimated_cost)
            return result
        except Exception as e:
            _record_llm_usage(self._role_name, error=True)
            raise

    def generate(self, *args, **kwargs):
        """
        Track usage for generate calls.
        """
        try:
            result = self._client.generate(*args, **kwargs)
            # Estimate token usage (rough approximation)
            prompt_tokens = len(str(args) + str(kwargs)) // 4
            completion_tokens = len(str(result)) // 4
            total_tokens = prompt_tokens + completion_tokens

            cost_per_token = 0.00015
            estimated_cost = total_tokens * cost_per_token

            _record_llm_usage(self._role_name, tokens=total_tokens, cost=estimated_cost)
            return result
        except Exception as e:
            _record_llm_usage(self._role_name, error=True)
            raise

    @property
    def client(self):
        """Pass through client property."""
        return self._client.client


def create_analytics_explainer_llm():
    """
    Factory for analytics explainer LLM role.
    """
    return create_llm_for_role("analytics_explainer")


def create_hpo_planner_llm():
    """
    Factory for HPO planner LLM role.
    """
    return create_llm_for_role("hpo_planner")


def create_news_features_llm():
    """
    Factory for news features/enrichment LLM role.
    """
    return create_llm_for_role("news_enricher")


def create_research_agent_llm():
    """
    Factory for research agent LLM role.
    """
    return create_llm_for_role("research_agent")


def create_reporting_agent_llm():
    """
    Factory for reporting agent LLM role.
    """
    return create_llm_for_role("reporting_agent")


def create_explainability_agent_llm():
    """
    Factory for explainability agent LLM role.
    """
    return create_llm_for_role("explainability_agent")


def create_notification_agent_llm():
    """
    Factory for notification agent LLM role.
    """
    return create_llm_for_role("notification_agent")


def create_strategy_planner_llm():
    """
    Factory for strategy planner LLM role.
    """
    return create_llm_for_role("strategy_planner")


# Legacy/deprecated functions (keep for backward compatibility)
def create_news_analyzer_llm():
    """
    Factory for news analyzer LLM role (deprecated - use create_research_agent_llm).
    """
    return create_llm_for_role("research_agent")


def create_risk_assessment_llm():
    """
    Factory for risk assessment LLM role (deprecated - use create_analytics_explainer_llm).
    """
    return create_llm_for_role("analytics_explainer")