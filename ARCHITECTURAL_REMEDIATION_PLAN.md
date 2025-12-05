# Architectural Remediation Plan

Based on the Senior Architect's review and the User's clarifications, this plan outlines the steps to stabilize the `agentic_forecast` system.

## 1. Fix Model Target & Metrics (Priority: Critical)
**Context**: The current model yields ~94% MAPE, which is catastrophic. The user specified that the target *must* be Log Returns ($y_t = \ln(P_{t+1}/P_t)$). MAPE is mathematically unstable for returns (near zero) and should be replaced or only used on reconstructed prices.

**Tasks**:
- [x] **Audit Target Generation**: Verify `src/data_pipeline.py` calculates log returns correctly.
- [x] **Audit Metric Calculation**: Check `src/agents/forecast_agent.py`.
    - If target is returns, **disable MAPE** or calculate it only on *reconstructed prices*.
    - Ensure MAE/RMSE are the primary metrics.
- [x] **Verify Baselines**: Ensure `BaselineLinear` and `Naive` models are actually being trained and compared.

## 2. Fix Data Pipeline Leakage (Priority: High)
**Context**: The architect identified potential look-ahead bias in `RobustScaler` usage (fitting on the whole dataset?) and `fillna` methods.

**Tasks**:
- [x] **Split-then-Scale**: Ensure `RobustScaler` is `fit` ONLY on the training set, then `transform` applied to validation/test.
- [x] **Safe Imputation**: Verify `fillna` does not use future data (e.g., `bfill` or `ffill` across split boundaries).
- [x] **Corporate Actions**: Verify if "Adjusted Close" is used.

## 3. Formalize Strategy Configuration (Priority: Medium)
**Context**: Strategy selection (`momentum_growth`) must be deterministic and config-driven, not hallucinated.

**Tasks**:
- [x] **Create Config**: Create `config/strategies.yaml` defining valid strategies.
- [x] **Enforce Validation**: Update `StrategyPlannerAgent` (or equivalent) to validate selected strategy against this config.

## 4. System Hardening (Priority: Low)
**Context**: `PaperBroker` is fragile.

**Tasks**:
- [x] **Transaction Safety**: Add file locking to `paper_portfolio.json`.

## 5. Event-Driven Backtester (Priority: High)
**Context**: Need to simulate historical performance without look-ahead bias.

**Tasks**:
- [x] **Refactor Executor**: Update `BacktestExecutor` to run a day-by-day loop.
- [x] **Implement Time Travel**: Add `cutoff_date` to `GraphState` and filter data in `data_ingestion_node`.
- [x] **Isolate Broker**: Inject a dedicated `PaperBroker` instance for backtesting to avoid corrupting live portfolio.
- [x] **Verification**: Create and run `run_backtest.py` to prove the loop works.

## 6. Migrate to SQLite/Postgres (Priority: Medium)
**Context**: Move away from JSON files for persistence.

**Tasks**:
- [x] **Implement DatabaseService**: Create service to handle SQLite operations.
- [x] **Integrate with Broker**: Update `PaperBroker` to use `DatabaseService`.
- [x] **Integrate with Pipeline**: Update `DataPipeline` to save market data to DB.
- [x] **Verify Migration**: Ensure data is persisting correctly.

## 7. Implement Slippage & Fee Models (Priority: Medium)
**Context**: Make backtests realistic by including transaction costs.

**Tasks**:
- [x] **Create Models**: Implement `TransactionCostModel` classes.
- [x] **Create Liquidity Agent**: Implement `LiquidityAgent` for spread/volume analysis.
- [x] **Integrate with Broker**: Update `PaperBroker` to use cost models.
- [x] **Integrate with Pipeline**: Use `LiquidityAgent` for universe selection.

