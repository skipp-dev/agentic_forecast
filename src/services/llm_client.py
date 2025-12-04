# src/services/llm_client.py
import os
from typing import Optional
from openai import OpenAI


class OpenAILLMClient:
    def __init__(self, model: str, base_url: Optional[str] = None) -> None:
        self.model = model
        self.base_url = base_url
        self._client = None  # Lazy-loaded

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.base_url,  # None = default OpenAI endpoint
            )
        return self._client

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Use appropriate parameter based on model
        completion_params = {
            "model": self.model,
            "messages": messages,
        }

        # o-series models (like o4-mini) don't support temperature or max_tokens parameters
        if self.model.startswith("o"):
            # o-series models only support max_completion_tokens
            completion_params["max_completion_tokens"] = max_tokens
            # temperature is not supported for o-series models
        else:
            completion_params["temperature"] = temperature
            completion_params["max_tokens"] = max_tokens

        resp = self.client.chat.completions.create(**completion_params)
        return resp.choices[0].message.content

    def generate(self, messages: list, **kwargs) -> str:
        """
        Generate response from a list of messages (chat format).
        """
        # Use appropriate parameter based on model
        completion_params = {
            "model": self.model,
            "messages": messages,
        }

        # o-series models (like o4-mini) don't support temperature or max_tokens parameters
        if self.model.startswith("o"):
            # o-series models only support max_completion_tokens
            if "max_tokens" in kwargs:
                 completion_params["max_completion_tokens"] = kwargs.pop("max_tokens")
            # temperature is not supported for o-series models
            if "temperature" in kwargs:
                kwargs.pop("temperature")
        else:
            # Pass through kwargs like temperature, max_tokens
            completion_params.update(kwargs)

        resp = self.client.chat.completions.create(**completion_params)
        return resp.choices[0].message.content
