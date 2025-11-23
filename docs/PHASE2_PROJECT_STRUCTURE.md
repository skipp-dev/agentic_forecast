# Phase 2 — Cross-Asset Features & Global Training

This document describes where Phase 2 related code lives, how it integrates with the
rest of the repository, and notes for reviewers.

Top-level map (Phase 2 relevant files)
- `cross_asset_features.py` — Core cross-asset feature engineering class (crypto, AI, commodities, macro, correlations).
- `run_cross_asset_features.py` — Script to batch-generate cross-asset features and store them into the TimeSeriesFeatureStore.
- `data/feature_store.py` — TimeSeriesFeatureStore implementation: storage, retrieval, versioning and partitioning for feature sets.
- `run_global_models.py` — Global training pipeline updated to load features from the feature store (experiment: `cross_asset`) and to fall back to baseline features when necessary.
- `agents/global_model_agent.py` — Global model training agent (NHITS-style) used by `run_global_models.py` to train on multi-symbol datasets.

Where Phase 2 integrates with the architecture
- The feature engineer (`cross_asset_features.py`) pulls raw symbol time-series from `data/raw/alpha_vantage` (and other sources), creates aligned cross-asset indicators, and stores them into `data/feature_store`.
- The global training script (`run_global_models.py`) now queries the feature store for the `cross_asset` experiment and prepares a multi-symbol training dataset for `GlobalModelAgent`.
- The feature store acts as the canonical feature repository and is used throughout other agents and tests.

Developer notes & review checklist
- Verify feature sets exist in `data/feature_store/` — this implementation writes partitioned data and metadata to `data/feature_store/feature_store.db`.
- Cross-asset feature generation can be run with `python run_cross_asset_features.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD` or via the watchlist.
- The global trainer supports `--experiment cross_asset` and falls back to `baseline` parquet features when cross-asset features are missing.
- Review data alignment logic (pandas concat + ffill/bfill) in `cross_asset_features.py` — there are deliberate forward/backward fills to handle missing dates across asset classes.
- Check GPU compatibility: the trainer will attempt to use CUDA but falls back to CPU if incompatible.

Quick commands
```powershell
# Generate cross-asset features for watchlist
python run_cross_asset_features.py

# Train a global model using cross-asset features (uses watchlist_ibkr.csv by default)
python run_global_models.py --experiment cross_asset --max-symbols 100
```

Status
- Cross-asset features generation: implemented and tested on subsets (95 symbols generated during development).
- Global training pipeline: updated and tested on batches (50–100 symbols), with GPU attempted but CPU fallback handled.
