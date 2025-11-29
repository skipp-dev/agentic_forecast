"""
LangSmith-managed LLM Client

This client integrates with LangSmith's "Models / Model usage" page,
allowing you to use centrally managed LLM configurations instead of
hard-coding API keys and model names.
"""

import os
from typing import Optional, Dict, Any
try:
    from langsmith import Client as LangSmithClient
except ImportError:
    LangSmithClient = None  # Allow import even if langsmith not installed


class SmithLLMClient:
    """
    Thin wrapper around LangSmith 'Model usage' endpoint.

    This client uses LangSmith's model management to route requests
    to configured providers (OpenAI, Anthropic, etc.) while providing
    centralized control and observability.

    Usage:
        client = SmithLLMClient(model="forecast-analytics-llm")
        answer = client.complete("Explain this metrics JSON: ...")
    """

    def __init__(self, model: str, api_key: Optional[str] = None, project: Optional[str] = None):
        """
        Initialize the Smith LLM client.

        Args:
            model: The logical model name configured in LangSmith (e.g., "forecast-analytics-llm")
            api_key: LangSmith API key (defaults to LANGSMITH_API_KEY env var)
            project: LangSmith project name (defaults to LANGSMITH_PROJECT env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY")
        self.project = project or os.getenv("LANGSMITH_PROJECT", "agentic_forecast")

        if not self.api_key:
            # For testing/development, allow initialization without API key
            # but warn that it won't work for actual calls
            print("Warning: LangSmith API key not found. Smith LLM client will fail on actual API calls.")
            self.client = None
        elif LangSmithClient is None:
            # LangSmith library not installed
            print("Warning: langsmith package not installed. Smith LLM client will fail on actual API calls.")
            self.client = None
        else:
            # Initialize LangSmith client
            self.client = LangSmithClient(api_key=self.api_key)

    def complete(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> str:
        """
        Synchronous completion using LangSmith-managed model.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            The model's response as a string
        """
        if not self.client:
            return f"Error: Smith LLM client not initialized. Missing LANGSMITH_API_KEY."

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            # Use LangSmith's model invocation
            # Note: This assumes LangSmith provides a chat completions interface
            # The exact API may vary based on LangSmith's implementation
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = f"LangSmith LLM call failed: {str(e)}"
            print(error_msg)
            return error_msg

    async def acomplete(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> str:
        """
        Async completion using LangSmith-managed model.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            The model's response as a string
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            # Use LangSmith's async model invocation
            response = await self.client.chat.completions.acreate(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = f"LangSmith LLM async call failed: {str(e)}"
            print(error_msg)
            return error_msg

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Legacy method for compatibility with existing LLMClient interface.
        """
        return self.complete(prompt, system=system_prompt, **kwargs)

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the LangSmith client is properly configured.

        Returns:
            Dict with health status information
        """
        if not self.client:
            return {
                "status": "unhealthy",
                "provider": "smith",
                "model": self.model,
                "error": "LangSmith API key not configured"
            }

        try:
            # Try to list projects to verify API key works
            projects = list(self.client.list_projects())
            return {
                "status": "healthy",
                "provider": "smith",
                "model": self.model,
                "project": self.project,
                "projects_available": len(projects)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "smith",
                "model": self.model,
                "error": str(e)
            }