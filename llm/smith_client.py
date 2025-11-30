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
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model

        # Prefer explicit arguments, fall back to env vars
        self.base_url = base_url or os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
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
    # Public sync API
    # ---------------------------------------------------------------------

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
            return data["choices"][0]["message"]["content"]
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