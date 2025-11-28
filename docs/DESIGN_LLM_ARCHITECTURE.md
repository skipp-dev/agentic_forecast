# Design: LLM-Enhanced Agentic Forecasting

## Overview
This document outlines the architecture for integrating Large Language Models (LLMs) into the `agentic_forecast` system. The goal is to use LLMs as the "brain" or "meta-layer" around the existing numerical time-series models (NHITS, TFT, LSTM, etc.), rather than replacing them.

## Core Philosophy
> ⚙️ **Numerical Models**: Do the heavy lifting (forecasting, training).
> 🧠 **LLM Agents**: Orchestrate, explain, monitor, experiment, and generate features.

## New Agent Roles

### 1. LLM Analytics & Explainer Agent
*   **Role**: Interprets performance metrics and drift statistics.
*   **Input**: `performance_summary`, `drift_stats`, `risk_kpis`.
*   **Output**: Natural language reports, structured recommendations (e.g., "Schedule HPO for AAPL due to drift").
*   **Integration**: Runs asynchronously (batch mode) after monitoring cycles.

### 2. LLM HPO Planner Agent
*   **Role**: Designs adaptive search spaces for Hyperparameter Optimization.
*   **Input**: Past HPO runs, model performance history.
*   **Output**: JSON plan for the next HPO cycle (model families, search bounds, budget allocation).
*   **Integration**: Triggers before the `HyperparameterSearchAgent`.

### 3. LLM News Feature Agent
*   **Role**: Converts unstructured news into structured features.
*   **Input**: Raw news headlines/bodies (IBKR, AlphaVantage).
*   **Output**: Structured tags (`earnings`, `guidance_cut`), sentiment scores, impact horizon.
*   **Integration**: Part of the data ingestion pipeline, feeding into `FeatureEngineerAgent`.

### 4. Guardrail Agent (LLM-Enhanced)
*   **Role**: Policy checker and sanity reviewer.
*   **Input**: Proposed model switches, configuration changes.
*   **Output**: Approval/Rejection with reasoning.
*   **Integration**: Gatekeeper before deployment or major state changes.

### 5. Reporting Agent (LLM-Enhanced)
*   **Role**: Generates executive summaries and stakeholder reports.
*   **Input**: System-wide state, performance logs.
*   **Output**: High-level summaries for dashboards/email.

## Architecture Changes

### LangGraph Integration
New nodes will be added to the `OrchestratorAgent`'s graph:
*   `llm_analytics_node`
*   `llm_hpo_planner_node`
*   `llm_news_enricher_node`

### Data Flow
1.  **Monitoring** -> `AnalyticsDriftAgent` -> **LLM Analytics Agent** -> *Recommendations*
2.  *Recommendations* -> **LLM HPO Planner** -> `HyperparameterSearchAgent`
3.  **News Ingestion** -> **LLM News Agent** -> `FeatureEngineerAgent` -> *Forecasting Models*

## Technology Stack
*   **LLM Interface**: Generic client wrapper (supporting OpenAI, Gemini, Anthropic, Local LLMs).
*   **Prompt Engineering**: Centralized prompt library (`src/prompts/llm_prompts.py`).
*   **Structured Output**: JSON enforcement for all agent outputs.

## Next Steps
See `PLAN_LLM_INTEGRATION.md` for the implementation roadmap.
