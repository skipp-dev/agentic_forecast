# Plan: LLM Integration Roadmap

## Phase 1: Foundation & Interfaces
- [ ] **Create LLM Client Wrapper**: Implement a generic interface for LLM calls (`src/llm/client.py`).
- [ ] **Prompt Library**: Create `src/prompts/llm_prompts.py` containing the 20+ identified prompt templates.
- [ ] **Configuration**: Update `config.yaml` to include LLM provider settings (API keys, model names).

## Phase 2: Core LLM Agents Implementation
- [ ] **LLM Analytics Agent**:
    - Implement `agents/llm_analytics_agent.py`.
    - Define input/output schemas (Pydantic models).
    - Connect to `AnalyticsDriftAgent` outputs.
- [ ] **LLM HPO Planner Agent**:
    - Implement `agents/llm_hpo_planner_agent.py`.
    - Define `HPOPlan` schema.
    - Integrate with `HyperparameterSearchAgent`.
- [ ] **LLM News Feature Agent**:
    - Implement `agents/llm_news_agent.py`.
    - Define `EnrichedNewsFeature` schema.
    - Integrate with data pipeline.

## Phase 3: Orchestration & Wiring
- [ ] **Update GraphState**: Add keys for `analytics_recommendations`, `hpo_plan`, `enriched_news`.
- [ ] **Update Orchestrator**: Add new nodes to the LangGraph workflow.
- [ ] **Define Triggers**:
    - Run Analytics Agent daily.
    - Run HPO Planner on drift detection.
    - Run News Agent on ingestion.

## Phase 4: Testing & Validation
- [ ] **Unit Tests**: Mock LLM responses to test agent logic.
- [ ] **Integration Tests**: Run full pipeline with dummy LLM (or low-cost model).
- [ ] **Evaluation**: Verify that LLM recommendations make sense (sanity check).

## Phase 5: Documentation & UI
- [ ] **Update Dashboard**: Visualize LLM explanations and recommendations.
- [ ] **Documentation**: Update `README.md` and developer guides.
