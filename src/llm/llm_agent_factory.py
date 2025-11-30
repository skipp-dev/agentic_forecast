"""
LLM Agent Factory - Declarative LLM Agent Creation

Creates LLM agents based on configuration, automatically wiring:
- Correct model selection
- System/user prompt loading from PROMPTS
- Message formatting for LLM API calls

This makes LLM usage explicit, declarative, and easy to change.
"""

import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from src.llm.llm_factory import create_llm_for_role
from src.prompts.llm_prompts import PROMPTS, build_llm_messages

logger = logging.getLogger(__name__)


class LLMAgentFactory:
    """
    Factory for creating LLM agents based on declarative configuration.

    Reads agents_config.yaml and creates agents with proper model/prompt wiring.
    """

    def __init__(self, config_path: str = "agents_config.yaml"):
        """
        Initialize factory with configuration.

        Args:
            config_path: Path to the agents configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse the agent configuration."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Agent config file not found: {config_file}")

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _validate_config(self):
        """Validate that the configuration has required structure."""
        if 'llm_agents' not in self.config:
            raise ValueError("Config must contain 'llm_agents' section")

        required_keys = ['class', 'model', 'system_prompt_key', 'user_prompt_key']

        for agent_name, agent_config in self.config['llm_agents'].items():
            missing_keys = [key for key in required_keys if key not in agent_config]
            if missing_keys:
                raise ValueError(f"Agent '{agent_name}' missing required keys: {missing_keys}")

            # Validate that prompt keys exist in PROMPTS
            system_key = agent_config['system_prompt_key']
            user_key = agent_config['user_prompt_key']

            if system_key not in PROMPTS:
                raise ValueError(f"System prompt key '{system_key}' not found in PROMPTS")
            if user_key not in PROMPTS:
                raise ValueError(f"User prompt key '{user_key}' not found in PROMPTS")

    def create_agent(self, agent_name: str):
        """
        Create an LLM agent instance based on configuration.

        Args:
            agent_name: Name of the agent to create (key in llm_agents config)

        Returns:
            Instantiated agent object

        Raises:
            ValueError: If agent_name not found in config
        """
        if agent_name not in self.config['llm_agents']:
            available_agents = list(self.config['llm_agents'].keys())
            raise ValueError(f"Unknown agent '{agent_name}'. Available: {available_agents}")

        agent_config = self.config['llm_agents'][agent_name]

        # Get the agent class
        class_name = agent_config['class']
        module_name = f"agents.{self._get_module_name(class_name)}"

        try:
            import importlib
            module = importlib.import_module(module_name)
            agent_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import agent class '{class_name}' from '{module_name}': {e}")

        # Create LLM client for this agent
        model = agent_config['model']
        llm_client = create_llm_for_role(agent_name, model_override=model)

        # Create agent instance
        agent_instance = agent_class(llm_client)

        logger.info(f"Created LLM agent '{agent_name}' with class '{class_name}' using model '{model}'")
        return agent_instance

    def _get_module_name(self, class_name: str) -> str:
        """Convert class name to module name (snake_case)."""
        # Convert CamelCase to snake_case
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get the configuration for a specific agent."""
        return self.config['llm_agents'].get(agent_name, {})

    def get_available_agents(self) -> list[str]:
        """Get list of all available agent names."""
        return list(self.config['llm_agents'].keys())

    def get_agent_prompt_keys(self, agent_name: str) -> Dict[str, str]:
        """Get the prompt keys for a specific agent."""
        config = self.get_agent_config(agent_name)
        if not config:
            return {}

        return {
            'system': config['system_prompt_key'],
            'user_template': config['user_prompt_key']
        }

    def build_agent_messages(self, agent_name: str, user_content: str) -> list:
        """
        Build LLM messages for an agent using its configured prompts.

        Args:
            agent_name: Name of the agent
            user_content: Formatted user content (already templated)

        Returns:
            List of message dictionaries for LLM API
        """
        prompt_keys = self.get_agent_prompt_keys(agent_name)
        if not prompt_keys:
            raise ValueError(f"No prompt keys found for agent '{agent_name}'")

        system_prompt = PROMPTS[prompt_keys['system']]
        return build_llm_messages(system_prompt, user_content)


# Global factory instance for convenience
_factory_instance: Optional[LLMAgentFactory] = None


def get_llm_agent_factory() -> LLMAgentFactory:
    """Get the global LLM agent factory instance."""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = LLMAgentFactory()
    return _factory_instance


def create_llm_agent(agent_name: str):
    """
    Convenience function to create an LLM agent.

    Args:
        agent_name: Name of the agent to create

    Returns:
        Instantiated agent object
    """
    factory = get_llm_agent_factory()
    return factory.create_agent(agent_name)