"""
SmithLLMClient - thin wrapper around LangSmith-hosted chat models.

This client calls the LangSmith "Model usage" chat API endpoint and is
designed to be used from create_llm_for_role() and your LLM agents.

You MUST:
- set LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) in your environment
- set LANGSMITH_ENDPOINT (e.g. https://eu.smith.langchain.com)
- copy the correct path & headers from the LangSmith UI "Model usage" snippet
"""

from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional

import requests
try:
    import httpx  # optional, for async
except ImportError:
    httpx = None

try:
    from langsmith import traceable
except ImportError:
    # Fallback if langsmith not installed
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def _smith_process_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Process SmithLLMClient inputs for LangSmith tracing."""
    # The inputs come as method arguments, we need to extract the messages
    # This is a bit tricky since we need to reconstruct the messages from the method call
    # For now, let's assume the traceable decorator will handle the basic structure
    return inputs


def _smith_process_outputs(outputs: str) -> Dict[str, Any]:
    """Process SmithLLMClient outputs for LangSmith tracing."""
    return {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": outputs
                    }
                ]
            }
        ]
    }


class SmithLLMError(RuntimeError):
    """Raised when the Smith LLM API returns an error response."""
    pass


class SmithLLMClient:
    """
    Simple synchronous + async client for calling LangSmith-hosted models.

    Example:
        client = SmithLLMClient(model="forecast-analytics-llm")
        text = client.complete("Explain this forecast report in simple English.")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model

        # Prefer explicit arguments, fall back to env vars
        self.base_url = base_url or "https://api.openai.com"  # Default to OpenAI
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LANGSMITH_API_KEY")
        self.timeout = timeout

        if not self.base_url:
            raise ValueError("SmithLLMClient: base_url is not set. "
                             "Set LANGSMITH_ENDPOINT or LANGCHAIN_ENDPOINT env var.")
        if not self.api_key:
            raise ValueError("SmithLLMClient: API key is not set. "
                             "Set LANGSMITH_API_KEY or LANGCHAIN_API_KEY env var.")

        # IMPORTANT:
        # Replace this path with exactly what the LangSmith "Model usage" page shows.
        # For many setups this will be something like: "/v1/chat/completions"
        # If your curl example shows "/v1/ls/chat/completions", use that instead
        self._chat_path = "/v1/chat/completions"

    # ---------------------------------------------------------------------
    # Utility: build messages + headers
    # ---------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        system: Optional[str] = None,
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_headers(self) -> Dict[str, str]:
        # Adjust header names if LangSmith's snippet shows something different.
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ---------------------------------------------------------------------
    # LangSmith tracing helpers
    # ---------------------------------------------------------------------

    def _process_inputs_for_tracing(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Convert messages to LangSmith format for tracing."""
        return {
            "messages": [
                {
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": msg["content"]
                        }
                    ]
                }
                for msg in messages
            ]
        }

    def _process_outputs_for_tracing(self, response_text: str) -> Dict[str, Any]:
        """Convert response to LangSmith format for tracing."""
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": response_text
                        }
                    ]
                }
            ]
        }

    # ---------------------------------------------------------------------
    # Public sync API
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Public sync API
    # ---------------------------------------------------------------------

    @traceable(
        run_type="llm",
        metadata={"ls_provider": "smith", "ls_model_name": "o4-mini"},
    )
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Simple synchronous completion. Returns the assistant text only.
        """
        messages = self._build_messages(prompt, system=system, extra_messages=extra_messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        url = self.base_url.rstrip("/") + self._chat_path

        try:
            resp = requests.post(
                url,
                headers=self._build_headers(),
                data=json.dumps(payload),
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise SmithLLMError(f"Request to Smith LLM failed: {e}") from e

        if resp.status_code >= 400:
            raise SmithLLMError(
                f"Smith LLM API error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        try:
            response_text = data["choices"][0]["message"]["content"]
            return response_text
        except (KeyError, IndexError) as e:
            raise SmithLLMError(
                f"Unexpected Smith LLM response format: {data}"
            ) from e

    # ---------------------------------------------------------------------
    # Public async API (optional)
    # ---------------------------------------------------------------------

    async def acomplete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Async completion – requires httpx to be installed.
        """
        if httpx is None:
            raise RuntimeError(
                "httpx is not installed; install it or use complete() instead."
            )

        messages = self._build_messages(prompt, system=system, extra_messages=extra_messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        url = self.base_url.rstrip("/") + self._chat_path

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    url,
                    headers=self._build_headers(),
                    json=payload,
                )
            except httpx.HTTPError as e:
                raise SmithLLMError(f"Async request to Smith LLM failed: {e}") from e

            if resp.status_code >= 400:
                raise SmithLLMError(
                    f"Smith LLM API error {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                raise SmithLLMError(
                    f"Unexpected Smith LLM response format: {data}"
                ) from e

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Legacy method for compatibility with existing LLMClient interface.
        """
        return self.complete(prompt, system=system_prompt, **kwargs)

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the Smith LLM client is properly configured.

        Returns:
            Dict with health status information
        """
        if not self.api_key:
            return {
                "status": "unhealthy",
                "provider": "smith",
                "model": self.model,
                "error": "Smith LLM API key not configured"
            }

        if not self.base_url:
            return {
                "status": "unhealthy",
                "provider": "smith",
                "model": self.model,
                "error": "Smith LLM base_url not configured"
            }

        try:
            # Try a simple test request to verify connectivity
            # Use a minimal prompt to avoid costs
            test_prompt = "Hello"
            self.complete(test_prompt, max_tokens=10)
            return {
                "status": "healthy",
                "provider": "smith",
                "model": self.model,
                "base_url": self.base_url
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "smith",
                "model": self.model,
                "error": str(e)
            }