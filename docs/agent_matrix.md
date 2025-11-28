# Agent Matrix

This document outlines the responsibilities, inputs, and outputs of each agent in the AGENTIC_FORECAST framework.

| Agent / Node | Responsibility | Inputs (from GraphState) | Outputs (to GraphState) | Implementation Notes |
| :--- | :--- | :--- | :--- | :--- |
| **LoadDataNode** | Fetches raw time-series data. | `symbols` | `raw_data` | Connects to IBKR gateway or local cache. |
| **FeatureAgentNode** | Converts raw data into model-ready features. | `raw_data` | `features` | Consolidates logic from existing feature scripts. Should be configurable via TOML files. |
| **ForecastingNode** | Generates predictions using active models. | `features` | `forecasts` | Interfaces with a Model Registry to get the currently promoted models for each symbol/horizon. |
| **AnalyticsAgentNode** | Calculates performance, drift, and risk metrics. | `forecasts`, `features` | `performance_summary`, `drift_metrics`, `risk_kpis` | Purely numeric agent. The engine that feeds the decision-maker. |
| **DecisionAgentNode** | Interprets analytics and proposes strategic actions. | `performance_summary`, `drift_metrics`, `risk_kpis` | `recommended_actions` | **LLM-powered.** Takes JSON analytics and outputs structured JSON actions. The "brain" of the system. |
| **GuardrailAgentNode** | Vets proposed actions against hard-coded risk rules. | `recommended_actions` | `recommended_actions` (vetted) | A simple, rules-based sanity check to ensure stability. Can veto or downgrade action priority. |
| **ActionExecutorNode** | Dispatches vetted actions to the correct executor. | `recommended_actions` | `executed_actions`, `errors` | A router that calls other nodes (e.g., HPO, Model Registry) based on the action type. |
| **ReportingNode** | Generates a human-readable summary of the cycle. | `executed_actions`, `performance_summary` | (Logs, Markdown files) | Creates end-of-cycle reports for monitoring. |

