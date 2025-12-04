# Action Plan for Agentic Forecast System Enhancement

## Phase 1: Technical Debt & Robustness (Immediate)

- [x] **1. Cleanup Legacy IBKR Code**
    - [x] Rename `watchlist_ibkr.csv` to `watchlist_main.csv`.
    - [x] Update `config.yaml`, `main.py`, `USER_GUIDE.md` to remove IBKR references.
    - [x] Delete legacy tests (`test_unified_ibkr.py`, `test_ib_direct.py`).
    - [x] Refactor `unified_ingestion_v2.py` to remove IBKR logic.

- [x] **2. Fix Data Serialization Risk**
    - [x] Modify `src/nodes/data_nodes.py` to save raw data as Parquet (`.parquet`) instead of pickle.
    - [x] Ensure `_load_data_sync` handles the new format.

- [x] **3. Implement Partial Failure Handling**
    - [x] Wrap symbol processing in `forecasting_node` (in `execution_nodes.py`) with try-except blocks.
    - [x] Ensure one symbol failure doesn't crash the entire batch.
    - [x] Log errors to `state['errors']`.

- [x] **4. Externalize Trust Score Heuristics**
    - [x] Move hardcoded weights and thresholds from `src/analytics/trust_score.py` to `config/quality.yml` (or `settings.toml`).
    - [x] Update `TrustScoreCalculator` to load config.

- [x] **5. Bridge HPO Execution Gap**
    - [x] Ensure `hpo_decision` from `decision_agent` actually triggers HPO execution.
    - [x] Verify `main_graph.py` routing logic for HPO.

## Phase 2: Advanced Features (Next Steps)

- [ ] Implement Macro/Regime Awareness.
- [ ] Enhance Cross-Asset Features.
