# LLM Policy & Roles
_For the agentic_forecast LLM layer_

## 1. Goal

The LLM layer provides **explanations, planning, and enrichment**, not direct execution:

- Helps humans and agents understand performance, drift, and news.
- Proposes actions (HPO budgets, retraining, config changes).
- Never directly trades or manipulates external systems.

LLMs are **assistants**, not autonomous decision-makers.

---

## 2. LLM Roles

### 2.1 Active

**ReportingLLM**

- Purpose: Executive summaries of forecasting performance.
- Inputs: Metrics, drift summaries, key events.
- Outputs: Human-readable text for dashboards/reports.
- Backend: Local GGUF models (Gemma / Llama 3.1) with fallback to rule-based summaries.

### 2.2 Defined but pending full graph integration

**LLMAnalyticsExplainerAgent**

- Purpose: Explain performance & drift, recommend actions.
- Inputs: Metrics JSON (MAE, MAPE, DA, regime metrics, drift events).
- Outputs:
  - `explanation_text`
  - `recommended_actions` (e.g., "run HPO on bucket X", "consider retraining Y family").

**LLMHPOPlannerAgent**

- Purpose: Plan hyperparameter optimization.
- Inputs: Historical HPO runs (metrics, trial counts, families, budgets).
- Outputs:
  - Trial allocations per model family.
  - Narrowed search spaces.

**LLMNewsFeatureAgent**

- Purpose: Enrich news into structured features.
- Inputs: Headlines, timestamps, optional body text.
- Outputs:
  - sentiment (pos/neg/neutral),
  - impact direction (bullish/bearish/neutral),
  - horizon (intraday/short/medium),
  - volatility/impact flags.

---

## 3. LLM Tiers & Routing

We distinguish LLM backends into tiers:

- **Tier 1 – Fast local**
  - Gemma-2 2B
  - Llama 3.1 8B Q4
- **Tier 2 – Accurate local**
  - Llama 3.1 8B higher-precision
  - Mistral Q8
- **Tier 3 – Remote / premium (optional)**
  - OpenAI or other APIs (for heavy research only)

Routing policy (examples):

- ReportingLLM:
  - Tier 1 for routine daily summaries.
  - Tier 2 fallback for large, complex reports.
- LLMAnalyticsExplainer:
  - Prefer Tier 2 (or Tier 3 if enabled) for nuanced drift explanations.
- LLMHPOPlanner:
  - Tier 1 (structured JSON, small inputs).
- LLMNewsFeatureAgent:
  - Tier 1 for daily enrichment; Tier 2 for deep analysis days.

---

## 4. Integration Points in the Graph

LLM nodes in the LangGraph pipeline:

1. `llm_analytics_explainer_node`
   - After analytics & drift evaluation.
   - Produces explanations + action suggestions for HPO / retraining.

2. `llm_hpo_planner_node`
   - Before HPO execution.
   - Takes budgets & history, proposes trial allocations and search spaces.

3. `llm_news_enrichment_node`
   - After raw news ingestion.
   - Emits structured news features for the Feature Agent.

4. `reporting_llm_node`
   - Near the end of the pipeline.
   - Generates executive summaries for dashboards & logs.

Every node should have clear input/output contracts (JSON/datatypes).

---

## 5. Observability & Caching

- **LangSmith** is used to trace:
  - prompts,
  - responses,
  - errors,
  - latency.
- Explanations & enriched outputs are cached by:
  - run_id / date / hash of inputs.
- Cached JSONs live under:
  - `results/llm/...`
- If a cache hit exists and inputs haven't changed, reuse the LLM result.

---

## 6. Safety & Governance

- LLMs:
  - do not directly place orders,
  - do not interact with brokers,
  - do not write to production configs without going through ConfigAgent.
- Config changes:
  - are proposed by LLMs/agents,
  - and applied only via the Config/Governance layer,
  - optionally requiring human confirmation in production.

---

## 7. Evolution & "Learning"

The models themselves are static (no online fine-tuning in this repo).

The LLM layer improves by:

- better prompts in `llm_prompts.py`,
- better input context (regime-aware metrics, cross-asset features),
- smarter routing between tiers,
- evaluation of outputs via LangSmith and regression tests.

Over time we may introduce:
- Fine-tuning on internal forecasting/explanation data.
- Feedback loops based on which recommendations correlate with improved metrics.