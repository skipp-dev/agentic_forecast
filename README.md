# IB Forecast – Alpha Vantage Agentic Forecasting System

## 1. What is this?

**IB Forecast** is a modular, agentic forecasting framework for equities, built around:

- **Alpha Vantage** as the primary market data source (OHLCV, FX, crypto, commodities).
- A **LangGraph-based workflow** that orchestrates:
  - data ingestion,
  - feature engineering,
  - model training & prediction,
  - monitoring & drift detection.
- A growing **Model Zoo** (Naive baselines, local ML models, global time-series models like NHITS/TFT).
- A **Strategy layer** with buckets (e.g. AI, defensive, energy) and regime-aware rules (rates, gold/oil, seasonality).
- A **Monitoring + LLM Research loop**, where a Macro-Aware LLM analyzes performance and suggests changes to features, models and strategies.

The system is explicitly designed to be:

- **Config-driven** (YAML configs for data sources, features, regimes, strategies, guardrails).
- **Agentic** (separate agents for Data, Features, Models, Regimes, Strategies, Monitoring, HPO, LLM-Research).
- **Incrementally extensible** (Phase 1: core skeleton; Phase 2: macro/regimes; Phase 3: portfolio/execution).

> Note: IBKR/TWS is not required – current design is Alpha Vantage only.

### API Limits

**Alpha Vantage Premium**:
- The system is configured for a **Premium** Alpha Vantage key with a limit of **300 calls per minute**.
- The client includes rate limiting logic to respect this constraint.
- If you are using a free key (5 calls/min), you must update `config.yaml` and expect significantly slower performance.

---

## 2. How to run the daily pipeline

**Prerequisites (Phase 1):**

- Python 3.12+ (recommended)
- `ALPHA_VANTAGE_API_KEY` set in `.env`
- Dependencies installed (see `requirements.txt`)
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- Folder structure created (`data/raw`, `data/features`, `data/models`, `data/metrics`, `config/`, `pipelines/`, `agents/`, `models/`)

### 2.1 One-time setup

1. Clone the repo and install dependencies:

   ```bash
   git clone <your-repo-url>.git
   cd <your-repo>
   pip install -r requirements.txt
   ```

2. Create a `.env` file with at least:

   ```bash
   ALPHA_VANTAGE_API_KEY=your_key_here
   ```

3. **GPU Setup (Optional but Recommended):**

   If you have an NVIDIA GPU, install PyTorch with CUDA support:

   ```bash
   # Check your CUDA version
   nvidia-smi

   # Install PyTorch with CUDA (adjust version based on your CUDA)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

   # Verify GPU is working
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

3. Adjust configs in `config/`:
   - `universe.yaml` – which symbols / buckets to use
   - `data_sources.yaml` – Alpha Vantage settings
   - `feature_config.yaml` – which feature groups to enable
   - `strategy_playbook.yaml` – basic strategies per bucket/horizon

### 2.2 Manual runs (per step)

You can run the main steps individually during development:

```bash
# 1) Download / update OHLCV data from Alpha Vantage
python pipelines/run_ingestion.py

# 2) Build feature matrices for all symbols
python pipelines/run_features.py

# 3) Train or update models
python pipelines/run_training.py

# 4) Generate predictions and evaluate
python pipelines/run_predictions_and_eval.py

# 5) Build monitoring snapshot (aggregated metrics)
python pipelines/build_performance_snapshot.py
```

### 2.3 Daily end-to-end run (LangGraph)

Once the graph is wired up, you can run the complete daily flow with a single command:

```bash
python main_daily.py
```

This will typically execute the following nodes:

1. `fetch_data_node` – ingestion for the current date range  
2. `build_features_node` – feature engineering for all symbols  
3. `train_models_node` – train/update models (if configured)  
4. `generate_predictions_node` – next-day forecasts  
5. `evaluate_node` – update evaluation metrics  
6. `monitoring_node` – compute rolling metrics, drift, guardrails

> In Phase 2+, the daily run will also apply regime tags and strategy selection.

---

## 3. Architecture & Roadmap

The system is organized into clear layers and phases:

### 3.1 Layers

- **Data Layer**
  - Agents: `AlphaVantageDataAgent`, `MacroDataAgent`, `CommodityDataAgent`
  - Outputs: cleaned OHLCV & macro/commodity series in Parquet

- **Feature Layer**
  - `FeatureAgent` with configurable feature groups:
    - Phase 1: `price_basic`, `tech_basic`, `seasonality_calendar`
    - Phase 2: `macro_rates`, `macro_labor`, `commodities_gold`, `commodities_oil`, `cross_asset` (BTC, NVDA/AI)

- **Modeling Layer**
  - `ModelAgent` + `model_zoo.py`:
    - Naive baselines, local ML models (e.g. LSTM / trees)
    - Global TS models (e.g. NHITS/TFT/NeuralForecast – Phase 2)
  - `HyperparameterSearchAgent` for structured HPO
  - Ensembles (`simple_avg`, `defensive_fallback`, `performance_weighted`)

- **Regimes & Strategies (Phase 2)**
  - `RegimeAgent`:
    - Tags like `rates_regime`, `labor_regime`, `commodity_regime_oil`, `commodity_regime_gold`, `seasonality_regime`, `market_regime`
  - `StrategySelector` + `strategy_playbook.yaml`:
    - Strategies per bucket & regime (e.g. energy_oil_spike, defensive_risk_off_gold, growth_tech_under_hiking)

- **Monitoring & LLM Loop**
  - `MonitoringAgent`:
    - Rolling metrics, drift detection, guardrail triggers
  - `performance_snapshot.json` as input to
  - Macro-Aware LLM Research Agent:
    - Produces weekly Markdown reports with recommendations
    - Feeds back into configs: `feature_config.yaml`, `strategy_playbook.yaml`, `guardrail_config.yaml`

### 3.2 Roadmap

- **Phase 1 – Skeleton**
  - Alpha Vantage only
  - Basic features & models
  - Simple strategies
  - Monitoring & first weekly reviews

- **Phase 2 – Context & Intelligence**
  - Macro, commodities & cross-asset features
  - Regime-aware strategies
  - Stronger model zoo + HPO
  - Guardrails & drift logic
  - Macro-aware LLM Research Loop

- **Phase 3 – Portfolio & Execution (planned)**
  - Signal/position sizing layer
  - Portfolio/risk management
  - Execution hooks (broker/API abstraction)
  - More advanced strategies (possibly RL/policy-based)

For more detail, see:

- `phase1_alpha_vantage_roadmap.md`
- `phase2_alpha_vantage_roadmap.md`
- `forecast_system_big_picture.md`


