"""
Unified LLM Factory for Role-Based LLM Creation

This factory provides a clean interface for creating LLM clients based on
logical roles (analytics_explainer, hpo_planner, etc.) rather than specific
providers. It reads from the config to determine which backend to use for
each role.
"""

import os
import yaml
from typing import Optional, Protocol, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for an LLM backend."""
    provider: str         # "lm_studio" | "openai" | "smith"
    model: str            # model ID or alias
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None


class LLMProtocol(Protocol):
    """Protocol that all LLM clients must implement."""
    def complete(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> str: ...
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str: ...


def load_config() -> Dict[str, Any]:
    """Load the main configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def get_llm_backend(name: str) -> LLMConfig:
    """
    Read backend configuration from settings.

    Args:
        name: The backend name (e.g., "smith_analytics", "openai_direct")

    Returns:
        LLMConfig for the specified backend
    """
    config = load_config()
    llm_backends = config.get("llm_backends", {})

    if name not in llm_backends:
        raise ValueError(f"LLM backend '{name}' not found in configuration")

    backend_config = llm_backends[name]
    return LLMConfig(
        provider=backend_config["provider"],
        model=backend_config["model"],
        base_url=backend_config.get("base_url"),
        api_key_env=backend_config.get("api_key_env")
    )


def create_llm_for_role(role: str) -> LLMProtocol:
    """
    Return an LLM client for a given logical role using config-driven backend selection.

    Args:
        role: Logical role name (e.g., "analytics_explainer", "hpo_planner")

    Returns:
        LLM client implementing the LLMProtocol
    """
    config = load_config()
    default_roles = config.get("default_llm_roles", {})

    if role not in default_roles:
        raise ValueError(f"LLM role '{role}' not found in configuration")

    backend_name = default_roles[role]
    cfg = get_llm_backend(backend_name)

    if cfg.provider == "lm_studio":
        # Import here to avoid circular imports
        try:
            from .local_client import LocalLlamaClient
        except ImportError:
            # Fallback for different import paths
            import sys
            sys.path.append(str(Path(__file__).parent))
            from local_client import LocalLlamaClient

        if not cfg.base_url:
            raise ValueError(f"base_url required for LM Studio backend '{backend_name}'")

        return LocalLlamaClient(base_url=cfg.base_url, model=cfg.model)

    elif cfg.provider == "openai":
        # Use the existing LLMClient for OpenAI
        from .client import LLMClient
        return LLMClient(model=cfg.model)

    elif cfg.provider == "smith":
        # Use the new Smith LLM client
        from .smith_client import SmithLLMClient
        return SmithLLMClient(model=cfg.model)

    else:
        raise ValueError(f"Unknown LLM provider: {cfg.provider}")


# Convenience functions for specific roles
def create_analytics_explainer_llm() -> LLMProtocol:
    """Create LLM for analytics explanation tasks."""
    return create_llm_for_role("analytics_explainer")


def create_hpo_planner_llm() -> LLMProtocol:
    """Create LLM for HPO planning tasks."""
    return create_llm_for_role("hpo_planner")


def create_news_features_llm() -> LLMProtocol:
    """Create LLM for news feature extraction tasks."""
    return create_llm_for_role("news_features")


def create_reporting_llm() -> LLMProtocol:
    """Create LLM for reporting and summarization tasks."""
    return create_llm_for_role("reporting")


def test_llm_backends():
    """Test function to verify all configured LLM backends are working."""
    config = load_config()
    backends = config.get("llm_backends", {})
    roles = config.get("default_llm_roles", {})

    print("Testing LLM Backends Configuration")
    print("=" * 50)

    # Test each backend
    for backend_name, backend_config in backends.items():
        print(f"\nTesting backend: {backend_name}")
        try:
            cfg = get_llm_backend(backend_name)
            print(f"  ✓ Config loaded: provider={cfg.provider}, model={cfg.model}")

            # Try to create a client (but don't make actual API calls)
            if cfg.provider == "smith":
                from .smith_client import SmithLLMClient
                client = SmithLLMClient(model=cfg.model)
                health = client.health_check()
                print(f"  ✓ Smith client initialized: {health['status']}")
            elif cfg.provider == "openai":
                from .client import LLMClient
                client = LLMClient(model=cfg.model)
                print(f"  ✓ OpenAI client initialized: {'✓' if client.client else '✗ (no API key)'}")
            elif cfg.provider == "lm_studio":
                print(f"  ✓ LM Studio config valid: base_url={cfg.base_url}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Test role mappings
    print("\nTesting Role Mappings")
    print("-" * 30)
    for role, backend in roles.items():
        print(f"  {role} → {backend}")
        try:
            # Just test that the role can be resolved (don't create actual client)
            cfg = get_llm_backend(backend)
            print(f"    ✓ Maps to {cfg.provider}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")

    print("\nLLM Backend Test Complete")


if __name__ == "__main__":
    test_llm_backends()