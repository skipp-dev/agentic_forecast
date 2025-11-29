# SMITH_LLM_CLIENT – LangSmith-Backed LLM Integration

This document explains how the **Smith LLM client** integrates LangSmith-managed models into the `agentic_forecast` stack.

The goal:

- Use **LangSmith "Models / Model usage"** as a central control panel for LLMs.
- Keep **agents** (Analytics, HPO Planner, News, Reporting) decoupled from specific vendors.
- Allow switching models (OpenAI, Anthropic, etc.) by **config/UI**, not code changes.

---

## 1. High-Level Concept

We treat each LLM as a **logical role**, not a hard-coded model name:

- `analytics_explainer` – explains metrics, drift, & performance
- `hpo_planner` – plans hyperparameter searches, trial budgets
- `news_features` – converts news into structured sentiment/impact features
- `reporting` – generates executive summaries & dashboards narratives

Each role is mapped to an **LLM backend**:

- a local LM Studio model (`lm_studio`)
- a direct OpenAI model (`openai`)
- or a LangSmith-managed model (`smith`)

LangSmith sits in the middle and routes to configured providers (e.g. OpenAI), while also giving you observability and model switching.

---

## 2. Environment Variables

To use Smith as a backend, you typically need:

```text
LANGSMITH_API_KEY       # Your LangSmith API key
LANGSMITH_PROJECT       # Optional: project name for trace grouping
LANGCHAIN_TRACING_V2    # Already used in this repo, typically "true"
LANGCHAIN_API_KEY       # If you use LangChain's official client, as already configured
```

In your `.env` or system environment:

```env
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=agentic_forecast
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-or-langchain-key
```

> Note: The exact combination depends on the client library you use (LangChain / LangSmith SDK). The idea is: **Smith client uses these env vars to reach the same org/project as your web dashboard.**

---

## 3. Configuration – `config.yaml` (Example)

We add **LLM backends** and **role mappings**.

```yaml
llm_backends:
  # Local LM Studio – for cheap, offline work
  local_lm_studio:
    provider: lm_studio
    base_url: "http://127.0.0.1:1234/v1"
    model: "meta-llama-3.1-8b-instruct"

  # Direct OpenAI – if you want to call OpenAI directly
  openai_direct:
    provider: openai
    model: "gpt-4o"

  # Smith (LangSmith-managed) – using the "Model usage" page config
  smith_analytics:
    provider: smith
    model: "forecast-analytics-llm"       # logical model name in LangSmith

  smith_heavy_research:
    provider: smith
    model: "forecast-heavy-research-llm"  # logical model name in LangSmith

default_llm_roles:
  reporting:           smith_analytics
  analytics_explainer: smith_analytics
  hpo_planner:         smith_heavy_research
  news_features:       smith_analytics
```

* `llm_backends.*.provider`

  * `"lm_studio"` → local HTTP server
  * `"openai"` → direct OpenAI API
  * `"smith"` → via LangSmith "Model usage"

* `llm_backends.*.model`

  * For `smith`, this is the **model identifier/alias** configured in the **LangSmith Models/Usage** page.

* `default_llm_roles.*`

  * Which backend to use for each logical role.

Changing the analytics LLM is now a **config change**, not a code change.

---

## 4. Python: LLMConfig and Factory

Centralize all LLM creation in one place, e.g. `llm/llm_factory.py`.

```python
from dataclasses import dataclass
from typing import Optional

# You already have some settings loader (YAML/TOML)
from config.settings import settings


@dataclass
class LLMConfig:
    provider: str         # "lm_studio" | "openai" | "smith"
    model: str            # model ID or alias
    base_url: Optional[str] = None


def get_llm_backend(name: str) -> LLMConfig:
    """Read backend configuration from settings."""
    cfg = settings.llm_backends[name]
    return LLMConfig(
        provider=cfg["provider"],
        model=cfg["model"],
        base_url=cfg.get("base_url"),
    )
```

---

## 5. SmithLLMClient (Pseudo Interface)

This is a **pseudo interface** to keep your code clean. The actual implementation depends on which LangSmith client you use (HTTP API, LangChain's `Client`, etc.), but the shape is:

```python
class SmithLLMClient:
    """
    Thin wrapper around LangSmith 'Model usage' endpoint.

    Usage:
        client = SmithLLMClient(model="forecast-analytics-llm")
        answer = client.complete("Explain this metrics JSON: ...")
    """

    def __init__(self, model: str):
        self.model = model
        # here you'd initialize the actual LangSmith/LangChain client
        # e.g. self.client = langsmith.Client() or similar

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """
        Synchronous completion for simple use cases.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # PSEUDO: actual call depends on the client library you choose
        # resp = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=messages,
        # )
        # return resp.choices[0].message.content

        raise NotImplementedError("Wire this to your actual LangSmith client")

    async def acomplete(self, prompt: str, *, system: str | None = None) -> str:
        """
        Async completion for agentic flows / LangGraph.
        """
        # Same idea, but using an async client
        raise NotImplementedError("Wire this to your actual LangSmith async client")
```

You can later swap the internals for the official LangSmith/LangChain call pattern without touching any agent code.

---

## 6. Unified `create_llm_for_role` Helper

Now we build a single, unified factory that all agents use:

```python
from typing import Protocol

class LLMProtocol(Protocol):
    def complete(self, prompt: str, *, system: str | None = None) -> str: ...
    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str: ...


def create_llm_for_role(role: str) -> LLMProtocol:
    """
    Return an LLM client for a given logical role using config-driven backend selection.
    """
    backend_name = settings.default_llm_roles[role]
    cfg = get_llm_backend(backend_name)

    if cfg.provider == "lm_studio":
        from llm.lm_studio_client import LmStudioClient
        return LmStudioClient(
            base_url=cfg.base_url,
            model=cfg.model,
        )

    if cfg.provider == "openai":
        from llm.openai_client import OpenAIClient
        return OpenAIClient(
            model=cfg.model,
        )

    if cfg.provider == "smith":
        from llm.smith_client import SmithLLMClient
        return SmithLLMClient(
            model=cfg.model,
        )

    raise ValueError(f"Unknown LLM provider: {cfg.provider}")
```

This isolates:

* **Where the model comes from** (LM Studio, OpenAI, Smith)
* From **what the agent does** (analytics, planning, summarization, news)

---

## 7. Wiring Agents to Roles

Now your agents just ask for a role:

```python
from llm.llm_factory import create_llm_for_role

class LLMAnalyticsExplainerAgent:
    def __init__(self, ...):
        self.llm = create_llm_for_role("analytics_explainer")

    def explain_metrics(self, metrics_json: dict) -> str:
        prompt = build_analytics_prompt(metrics_json)
        return self.llm.complete(prompt)


class LLMHPOPlannerAgent:
    def __init__(self, ...):
        self.llm = create_llm_for_role("hpo_planner")

    def plan_hpo(self, hpo_history: dict, budget: dict) -> dict:
        prompt = build_hpo_planner_prompt(hpo_history, budget)
        answer = self.llm.complete(prompt)
        return parse_hpo_plan(answer)


class LLMNewsFeatureAgent:
    def __init__(self, ...):
        self.llm = create_llm_for_role("news_features")

    def enrich_news(self, headlines: list[dict]) -> list[dict]:
        prompt = build_news_enrichment_prompt(headlines)
        answer = self.llm.complete(prompt)
        return parse_news_features(answer)


class ReportingLLM:
    def __init__(self, ...):
        self.llm = create_llm_for_role("reporting")

    def generate_report(self, run_summary: dict) -> str:
        prompt = build_reporting_prompt(run_summary)
        return self.llm.complete(prompt)
```

If you decide that `analytics_explainer` should use a local LM Studio model for a while, you simply change:

```yaml
default_llm_roles:
  analytics_explainer: local_lm_studio
```

No agent code changes.

---

## 8. Recommended Role → Backend Mapping (Current Setup)

Based on your stack and goals:

* `analytics_explainer` → `smith_analytics`

  * Good balance of cost & quality
  * Works on metrics JSON, drift events, guardrail outputs

* `hpo_planner` → `smith_heavy_research`

  * Needs more reasoning depth for budget planning & search-space pruning

* `news_features` → `smith_analytics`

  * Takes pre-fetched, stored headlines and turns them into tags, sentiment, impact

* `reporting` → `smith_analytics` (or `smith_heavy_research` if you want very rich narratives)

  * Executive summaries, cross-asset narratives, risk explanations

Meanwhile:

* Keep LM Studio for:

  * local coding assistant
  * "Interactive Analyst Mode" in your IDE or notebooks
  * quick, non-critical exploratory LLM work

---

## 9. Testing Checklist

Before using this in production:

1. **Config sanity**

   * `config.yaml` has `llm_backends` and `default_llm_roles` as shown.
   * `smith_*` entries use model names that actually exist in the LangSmith UI.

2. **Env vars**

   * `LANGSMITH_API_KEY` set
   * `LANGSMITH_PROJECT` optional but helpful
   * `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` if you use LangChain integration

3. **Unit tests**

   * Write a tiny test that calls `create_llm_for_role("analytics_explainer")`
   * Mock the underlying LangSmith/OpenAI/LM Studio clients if needed.

4. **End-to-end smoke test**

   * Run a small pipeline that triggers:

     * `LLMAnalyticsExplainerAgent.explain_metrics()`
     * `ReportingLLM.generate_report()`
   * Verify traces appear in LangSmith
   * Confirm the configured Smith model is being used (via Smith UI)

Once this is all green, changing models becomes a **UI/config task** instead of a repo surgery, and all LangSmith-traced LLM logic in your agentic_forecast system will be consistent and observable.