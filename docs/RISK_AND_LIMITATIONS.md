# System Risk & Limitations Assessment

This document outlines the current architectural limitations, risks, and known gaps in the `agentic_forecast` system. It serves as a roadmap for future stability and scalability improvements.

## 1. Data Integrity & Leakage

### TimeMachine & Point-in-Time Guarantees
*   **Current State**: The `TimeMachine` utility exists and is used in `FeatureAgent` to enforce cutoff dates. However, it is not universally enforced across all data consumers (e.g., `RegimeDetectionAgent`, `NewsDataAgent`).
*   **Risk**: Potential for look-ahead bias in backtests if agents access future data (e.g., calculating regime clusters on the full history).
*   **Target State**: A universal `DataProvider` interface that strictly enforces `as_of` timestamps for all data access. All feature engineering must be proven to be point-in-time safe via regression tests.

### Regime Detection
*   **Current State**: `RegimeDetectionAgent` uses rule-based logic (safe) but has methods for clustering (potentially unsafe if fitted on full history).
*   **Risk**: If clustering is enabled for historical feature generation without a rolling window, future market conditions will leak into past training labels.
*   **Target State**: Strict rolling-window or expanding-window approach for any learned regime labels (clustering/HMM).

## 2. Orchestration & Stability

### Supervisor "Definition of Done"
*   **Current State**: The Supervisor's termination logic is implicit ("no more work"). This has led to recursion limit errors when agents "argue" or loop indefinitely.
*   **Risk**: Pipeline hangs or crashes with `RecursionError`.
*   **Target State**: Explicit `run_status` state machine (`INIT` -> `RUNNING` -> `COMPLETED` / `FAILED`). The Supervisor stops scheduling agents immediately upon reaching a terminal state.

### State Management
*   **Current State**: Monolithic `GraphState` dictionary passing large DataFrames between nodes.
*   **Risk**: High memory usage and serialization overhead, limiting scalability.
*   **Target State**: Sharded state where large artifacts (DataFrames, Models) are stored in a local/remote store (Parquet/S3), and only references/IDs are passed in the graph state.

## 3. Production Reliability & Monitoring

### Silent Fallbacks
*   **Current State**: The system is designed to be resilient, often falling back to `BaselineLinear` if complex models fail. This fallback is currently silent/unmonitored.
*   **Risk**: The system could be running in a degraded state (simple regression only) for weeks without detection.
*   **Target State**: Prometheus metrics tracking `model_champion_family` distribution. Alerts for "Baseline Dominance" (e.g., if Baseline wins >90% of the time).

### Disaster Recovery
*   **Current State**: No formal DR plan for data provider failures (Alpha Vantage, FRED).
*   **Risk**: Pipeline failure or garbage data ingestion if APIs change or go offline.
*   **Target State**: `DataProvider` abstraction with circuit breakers, caching, and a "No Data -> No Trade" safety policy.

## 4. Reproducibility

### Golden Dataset
*   **Current State**: Backtests run on "fresh" data, leading to drift in results due to data updates or API changes.
*   **Risk**: Inability to distinguish between code regressions and data changes.
*   **Target State**: A frozen, versioned "Golden Dataset" (small subset of symbols/history) used in CI to guarantee historical reproducibility of metrics.

### Versioning
*   **Current State**: Implicit versioning via git and run IDs.
*   **Risk**: Difficulty in auditing exactly which model version and feature set produced a specific forecast.
*   **Target State**: Explicit Model Registry (MLflow) and Feature Store versioning. Every forecast should be traceable to `ModelID` + `FeatureViewID`.

## 5. Market Realities

### Symbol Lifecycle
*   **Current State**: No explicit modeling of delistings, acquisitions, or corporate actions.
*   **Risk**: Agents may crash or hold "dead" positions if a symbol stops trading.
*   **Target State**: `InstrumentState` tracking (`ACTIVE`, `HALTED`, `DELISTED`). Logic to force-close positions on delisting and exclude non-tradable assets.
