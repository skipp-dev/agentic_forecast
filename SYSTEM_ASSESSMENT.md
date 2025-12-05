# Critical System Assessment & Architectural Audit

**Date:** 2025-02-26
**Scope:** Full System Review (Code, Config, Architecture)
**Severity:** Critical

## 1. Critical Architectural Risks

### 1.1. Data Leakage in Feature Engineering
**Location:** `src/agents/feature_engineer_agent.py` -> `_clean_and_normalize_features`
**Issue:** The agent uses `.bfill()` (backward fill) after `.ffill()` to handle missing values.
```python
df = df.ffill().bfill()
```
**Impact:** In a time-series forecasting context, backward filling propagates *future* information into the past. If a model trains on this data, it will have look-ahead bias, leading to unrealistically high backtest performance that will collapse in live trading.
**Recommendation:** Strictly forbid backward filling. Drop initial rows with `NaN` or use forward-only interpolation.

### 1.2. Dangerous Mock Data Fallback
**Location:** `src/agents/feature_engineer_agent.py` -> `_fetch_market_data`
**Issue:** If the API call fails, the system silently generates random noise:
```python
except Exception as e:
    logger.error(f"Failed to fetch market data... {e}")
    # Return mock data for testing
    mock_data = { ... np.random.randn ... }
```
**Impact:** In a production or semi-production run, a network glitch could cause the system to train models on random noise. These models might then be registered as "Champions" if the noise happens to fit a pattern, corrupting the model registry.
**Recommendation:** Fail fast. Raise an exception if data cannot be fetched. Mock data should *only* be used in explicit test environments, never in the core agent logic.

## 2. "Agentic" Architecture Gap

### 2.1. Procedural vs. Agentic Orchestration
**Observation:** The project is named `agentic_forecast`, but `main.py` relies on hardcoded procedural pipelines:
```python
if ctx.run_type == RunType.DAILY:
    run_daily_pipeline(ctx, symbols, config)
```
**Issue:** The `SupervisorAgent` (which presumably contains logic for dynamic routing and decision making) is bypassed. The "Agents" (e.g., `FeatureEngineerAgent`) are currently just standard Python classes with methods, lacking autonomy or a control loop.
**Impact:** The system loses the benefits of an agentic framework (adaptability, error recovery, dynamic planning). It is currently just a complex script.

## 3. Infrastructure & Lifecycle Maturity

### 3.1. Model Registry is a Single Point of Failure
**Location:** `src/data/model_registry.py`
**Issue:** The registry is a simple JSON wrapper. It tracks metadata (e.g., "Champion Family") but does not appear to enforce:
*   **Artifact Versioning:** No links to specific `.pkl` or `.ckpt` files (e.g., `model_v1_20240101.pkl`).
*   **Lineage:** No record of *which* data version trained *which* model.
**Impact:** Reproducibility is impossible. If a model starts failing, you cannot rollback to the exact previous binary because the registry doesn't link to it.

### 3.2. Configuration Sprawl
**Observation:** Configuration is scattered across:
1.  `config.yaml` (Main)
2.  `config/quality.yml` (Quality gates)
3.  `config/settings.toml` (Likely unused or redundant)
4.  `main.py` (Hardcoded env var overrides like `SKIP_NEURALFORECAST`)
**Impact:** "Hidden" settings. A developer might change `config.yaml` expecting a change, but be overridden by `main.py` or `quality.yml`.

## 4. Codebase Hygiene

### 4.1. Test Clutter
**Observation:** The root directory is littered with `test_*.py` files (`test_20b.py`, `test_cuda.py`, `test_local_llm.py`).
**Recommendation:** Move all tests to a `tests/` directory. Separate `unit` (fast) from `integration` (slow/GPU) tests.

## 5. Action Plan

1.  **Immediate Fixes**:
    *   Remove `.bfill()` from Feature Engineer.
    *   Remove Mock Data fallback in Feature Engineer (raise Error instead).
2.  **Refactoring**:
    *   Consolidate Config: Merge `quality.yml` into `config.yaml` or establish a strict hierarchy.
    *   Clean Root: Move tests to `tests/`.
3.  **Architectural Upgrades**:
    *   Implement a real `ModelRegistry` (SQLite or MLflow based) that tracks file paths.
    *   Refactor `main.py` to use `SupervisorAgent` for decision making instead of `if/else` blocks.
