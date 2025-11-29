"""
Local LLM Client for LM Studio HTTP API

This client provides a unified interface for local LLMs running via LM Studio's
HTTP API server, which is compatible with OpenAI's chat completions format.
"""

import os
import requests
from typing import Optional, Dict, Any
import openai


class LocalLlamaClient:
    """
    Client for local LLMs via LM Studio HTTP API.

    This client connects to LM Studio's local server (typically http://127.0.0.1:1234/v1)
    and provides a unified interface compatible with the LLMProtocol.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", model: str = "local-model"):
        """
        Initialize the local LLM client.

        Args:
            base_url: Base URL for the LM Studio server (e.g., "http://127.0.0.1:1234/v1")
            model: Model name/identifier
        """
        self.base_url = base_url.rstrip('/')
        self.model = model

        # Initialize OpenAI client pointing to local LM Studio server
        self.client = openai.OpenAI(
            api_key="lm-studio",  # LM Studio doesn't require a real API key
            base_url=self.base_url
        )

    def complete(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> str:
        """
        Complete method compatible with LLMProtocol.
        """
        return self.generate(prompt, system_prompt=system, **kwargs)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response using the local LM Studio model.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the model

        Returns:
            The model's response as a string
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            error_msg = f"Local LLM call failed: {str(e)}"
            print(error_msg)
            return error_msg

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the LM Studio server is running and accessible.

        Returns:
            Dict with health status information
        """
        try:
            # Try to get model info from LM Studio
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return {
                    "status": "healthy",
                    "provider": "lm_studio",
                    "base_url": self.base_url,
                    "models_available": len(models_data.get("data", []))
                }
            else:
                return {
                    "status": "unhealthy",
                    "provider": "lm_studio",
                    "base_url": self.base_url,
                    "error": f"HTTP {response.status_code}"
                }

        except requests.exceptions.RequestException as e:
            return {
                "status": "unhealthy",
                "provider": "lm_studio",
                "base_url": self.base_url,
                "error": f"Connection failed: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "lm_studio",
                "base_url": self.base_url,
                "error": str(e)
            }

    def close(self):
        """Clean up resources (no-op for HTTP client)."""
        pass