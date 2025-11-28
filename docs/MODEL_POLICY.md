# Model Policy & Selection Guide
_For the agentic_forecast / ModelZoo stack_

## 1. Purpose

This document explains **how and why** different model families are used in the forecasting system.

It answers:

- Which model families do we have?
- When should each family be used?
- How do hardware & runtime constraints influence choices?
- How do cross-asset features (V1/V2) and regimes affect selection?
- How do agents (HPO, Orchestrator, Guardrail) interact with the ModelZoo?

The goal: the ModelZoo is **intentional**, not “train everything and hope for the best”.

---

## 2. Model tiers

We classify models into three tiers:

### 2.1 Core Models (default, always available)

Fast, robust, and good baselines. These should work on almost any machine.

- **AutoNHITS** – Neural hierarchical interpolation
- **AutoNBEATS** – Neural basis expansion
- **AutoDLinear** – Decomposed linear model
- **BaselineLinear (NLinear)** – Simple linear baseline
- **AutoARIMA**
- **AutoETS**
- **AutoTheta**

**Typical use:**

- Daily production runs
- CPU-only environments
- First line of defense when advanced models fail

---

### 2.2 Advanced Models (richer, more expensive)

More expressive, often best when covariates (including cross-asset features) are rich and GPU is available.

- **AutoTFT** – Temporal Fusion Transformer
- **PatchTST** – Patch-based time series transformer
- **iTransformer** – Inverted transformer architecture
- **CNNLSTM** – CNN + LSTM hybrid
- **Global Models** – Models trained across many symbols (global forecasting)

**Typical use:**

- GPU or strong CPU machines
- Research and “full” runs
- Cross-asset / regime-aware experiments
- When we want to exploit cross-asset V1/V2 features

---

### 2.3 Experimental Models (opt-in only)

These are powerful but heavier, more complex, or less battle-tested.

- **GNN** – Graph neural networks (PyTorch Geometric)  
- **DeepAR** – Probabilistic forecasting

**Typical use:**

- Dedicated research runs
- When graph or probabilistic structure is a specific focus
- Not enabled by default in daily production

---

## 3. Configuration: model tiers & profiles

Configuration controls what’s active:

```yaml
models:
  core:
    enabled: true
    families:
      - AutoNHITS
      - AutoNBEATS
      - AutoDLinear
      - BaselineLinear
      - AutoARIMA
      - AutoETS
      - AutoTheta

  advanced:
    enabled: true
    families:
      - AutoTFT
      - PatchTST
      - iTransformer
      - CNNLSTM
      - GlobalModels

  experimental:
    enabled: false
    families:
      - GNN
      - DeepAR

  profiles:
    # Per bucket / universe profile (examples)
    tech_trading:
      tiers: [core, advanced]
      metric_profile: trading
    macro_index:
      tiers: [core, advanced]
      metric_profile: macro
    experimental_graph:
      tiers: [core, experimental]
      metric_profile: research

Here you go – a ready-to-drop `docs/MODEL_POLICY.md` that fits your current system and what we just discussed:

````markdown
# Model Policy & Selection Guide
_For the agentic_forecast / ModelZoo stack_

## 1. Purpose

This document explains **how and why** different model families are used in the forecasting system.

It answers:

- Which model families do we have?
- When should each family be used?
- How do hardware & runtime constraints influence choices?
- How do cross-asset features (V1/V2) and regimes affect selection?
- How do agents (HPO, Orchestrator, Guardrail) interact with the ModelZoo?

The goal: the ModelZoo is **intentional**, not “train everything and hope for the best”.

---

## 2. Model tiers

We classify models into three tiers:

### 2.1 Core Models (default, always available)

Fast, robust, and good baselines. These should work on almost any machine.

- **AutoNHITS** – Neural hierarchical interpolation
- **AutoNBEATS** – Neural basis expansion
- **AutoDLinear** – Decomposed linear model
- **BaselineLinear (NLinear)** – Simple linear baseline
- **AutoARIMA**
- **AutoETS**
- **AutoTheta**

**Typical use:**

- Daily production runs
- CPU-only environments
- First line of defense when advanced models fail

---

### 2.2 Advanced Models (richer, more expensive)

More expressive, often best when covariates (including cross-asset features) are rich and GPU is available.

- **AutoTFT** – Temporal Fusion Transformer
- **PatchTST** – Patch-based time series transformer
- **iTransformer** – Inverted transformer architecture
- **CNNLSTM** – CNN + LSTM hybrid
- **Global Models** – Models trained across many symbols (global forecasting)

**Typical use:**

- GPU or strong CPU machines
- Research and “full” runs
- Cross-asset / regime-aware experiments
- When we want to exploit cross-asset V1/V2 features

---

### 2.3 Experimental Models (opt-in only)

These are powerful but heavier, more complex, or less battle-tested.

- **GNN** – Graph neural networks (PyTorch Geometric)  
- **DeepAR** – Probabilistic forecasting

**Typical use:**

- Dedicated research runs
- When graph or probabilistic structure is a specific focus
- Not enabled by default in daily production

---

## 3. Configuration: model tiers & profiles

Configuration controls what’s active:

```yaml
models:
  core:
    enabled: true
    families:
      - AutoNHITS
      - AutoNBEATS
      - AutoDLinear
      - BaselineLinear
      - AutoARIMA
      - AutoETS
      - AutoTheta

  advanced:
    enabled: true
    families:
      - AutoTFT
      - PatchTST
      - iTransformer
      - CNNLSTM
      - GlobalModels

  experimental:
    enabled: false
    families:
      - GNN
      - DeepAR

  profiles:
    # Per bucket / universe profile (examples)
    tech_trading:
      tiers: [core, advanced]
      metric_profile: trading
    macro_index:
      tiers: [core, advanced]
      metric_profile: macro
    experimental_graph:
      tiers: [core, experimental]
      metric_profile: research
````

The **HPOAgent** and **Orchestrator** use this to decide:

* Which families are *eligible* per bucket/universe
* Which tiers to include based on environment and profile

---

## 4. Hardware- & cost-aware gating

Each model family has internal metadata, for example:

```yaml
family_meta:
  AutoNHITS:
    requires_gpu: false
    approx_cost: medium
  AutoTFT:
    requires_gpu: true
    approx_cost: high
  GNN:
    requires_gpu: true
    requires_graph: true
    approx_cost: very_high
```

At runtime, an internal **ModelPolicy** checks:

* `has_gpu?`
* `has_graph_inputs?` (edge_index, node_features)
* allowed `approx_cost` for the run

**Examples:**

* On a CPU-only laptop:

  * Include: AutoNHITS, AutoNBEATS, AutoDLinear, BaselineLinear, AutoARIMA, AutoETS, AutoTheta
  * Skip: AutoTFT, PatchTST, iTransformer, GNN, DeepAR

* On a GPU machine with graph data and experimental mode enabled:

  * Include: core + advanced + GNN / DeepAR (if allowed by profile)

Skipped families should be logged with a reason, e.g.:

> “Skipped AutoTFT: no GPU detected.”
> “Skipped GNN: experimental mode disabled.”
> “Skipped GlobalModels: global_mode.enabled = false.”

---

## 5. When to use which family

### 5.1 Core Models

**AutoNHITS / AutoNBEATS**

* Use when:

  * You want strong baselines with good performance
  * Time series are reasonably long and structured
* Pros:

  * Good general-purpose models
  * Reasonable performance on CPU

**AutoDLinear**

* Use when:

  * You want speed + simple decomposition (trend, seasonality)
  * You have many symbols and limited compute

**BaselineLinear / ARIMA / ETS / Theta**

* Use when:

  * You need simple, interpretable baselines
  * You want stability and robustness checks
* They are essential for:

  * Benchmarking more complex models
  * Estimating OWA (vs naive baseline)

---

### 5.2 Advanced Models

**TFT, PatchTST, iTransformer**

* Use when:

  * You have rich covariates (cross-asset V1/V2, sentiment, macro)
  * You have GPU
  * You care about capturing complex interactions and regimes
* Ideal for:

  * Buckets where cross-asset context matters
  * Experiments involving regime switching and exogenous features

**CNNLSTM**

* Use when:

  * You want a hybrid approach (local convolution + memory)
  * Series are noisy but have local patterns

**GlobalModels**

* Use when:

  * You have many related symbols (e.g. large sector/universe)
  * You want to exploit cross-series patterns
  * You can accept global model constraints (same architecture across series)

---

### 5.3 Experimental Models

**GNN**

* Use when:

  * You explicitly want to model symbol relationships as a graph
  * You have a clear graph (edges: sector, correlation, supply chain, etc.)
  * GPU + PyTorch Geometric are available
* Recommended usage:

  * Prefer as **embedding provider** (graph-informed features) rather than main forecaster in prod.

**DeepAR**

* Use when:

  * You want probabilistic forecasts
  * You care about full predictive distribution, not just point estimates
* Often better in research / specialized scenarios than broad production.

---

## 6. Metric profiles & selection logic

We don’t only care about “lowest MAPE”. Different use cases prioritize different metrics.

### 6.1 Metrics used

* MAE (Mean Absolute Error)
* MAPE (Mean Absolute Percentage Error)
* SMAPE (Symmetric MAPE)
* OWA (Overall Weighted Average vs naive)
* Directional Accuracy (DA)
* Regime-specific metrics:

  * MAE/DA on shock vs normal days
  * High-vol vs low-vol regimes

### 6.2 Metric profiles

Example metric profiles:

```yaml
metric_profiles:
  trading:
    primary: ["directional_accuracy", "mae_shock"]
    secondary: ["mae", "mape"]
  macro:
    primary: ["mape", "owa"]
    secondary: ["mae"]
  research:
    primary: ["mape", "directional_accuracy"]
    secondary: ["smape", "owa"]
```

Buckets/universes select a profile via config (`models.profiles.*.metric_profile`).

**Selection logic per bucket:**

1. For each candidate family:

   * Train / HPO
   * Compute metrics
2. Rank models primarily by profile-specific metrics (e.g. DA + shock MAE for `trading`)
3. Promote:

   * Best-performing model as champion
   * Possibly build an ensemble from top N families (e.g. via inverse-MAPE or DA-weighted)

---

## 7. Cross-asset V1/V2 integration

The system has cross-asset features:

* **V1**: core peer- and sector-based context (peer returns, shocks, sector index returns, etc.)
* **V2**: extended context (longer horizon momentum, sector drawdowns, beta vs sector, ranks, etc.)

Model policy should:

* Ensure **simple baseline models** (ARIMA/ETS/Theta/BaselineLinear):

  * either don’t use cross-asset features or only very limited subsets.
* Ensure **advanced/global models** (TFT, PatchTST, iTransformer, GlobalModels):

  * fully ingest cross-asset V1+V2 as exogenous features.
* Use **analytics / feature importance** to decide:

  * which V2 features are actually useful (e.g. `sector_drawdown_60d`, `beta_vs_sector_60d`)
  * which should be kept or dropped in future runs.

Guardrails and reporting should surface:

> “Models leveraging cross-asset V2 features improved shock-day MAE by 12%. Keeping V2 enabled for this bucket.”

---

## 8. GNN as embedding provider (recommended pattern)

Instead of treating GNN as “just another forecasting family” in production, we prefer:

1. Build a graph of symbols (nodes) with edges representing:

   * sector, correlation, supply chain, etc.
2. Train a GNN to produce **node embeddings**.
3. Use these embeddings as features for:

   * NHITS / NBEATS / TFT / GlobalModels

Advantages:

* You get graph-aware information in *all* major models
* You avoid relying on a heavy GNN forecaster directly in production
* You can turn the embedding step on/off per profile

In config, think of:

```yaml
graph:
  embeddings:
    enabled: true
    use_in_models: [AutoNHITS, AutoNBEATS, AutoTFT, GlobalModels]
  gnn_forecaster:
    enabled: false   # default, research-only
```

---

## 9. Retraining cadence per tier

Not all models should retrain equally often.

### 9.1 Core models

* Retrain: **often** (e.g. daily or every few runs)
* HPO: modest (few trials)
* Purpose: stable backbone, always fresh enough

### 9.2 Advanced models

* Retrain: **less often** (e.g. weekly or on strong drift signals)
* HPO: more expensive, but less frequent
* Purpose: high-performance models for important buckets

### 9.3 Experimental models

* Retrain: **on-demand only**, when we explicitly run research workflows
* HPO: manual or research-driven
* Purpose: experimentation and new ideas, not daily production

The **Drift/Analytics Agent** and **HPO Agent** collaborate:

* If drift & performance degradation exceed thresholds:

  * schedule retrains for core (first),
  * then advanced (if needed),
  * experimental only if a research job is active.

---

## 10. Transparency & logging

For each training / HPO session, the system should log:

* Which families were **included**, and **why**
* Which families were **skipped**, and **why**

Example model roster log:

| Family      | Tier         | Included? | Reason                                     |
| ----------- | ------------ | --------- | ------------------------------------------ |
| AutoNHITS   | core         | ✅         | default core model                         |
| AutoNBEATS  | core         | ✅         | default core model                         |
| AutoDLinear | core         | ✅         | fast low-compute family                    |
| AutoTFT     | advanced     | ❌         | no GPU detected                            |
| PatchTST    | advanced     | ✅         | GPU + advanced enabled                     |
| GNN         | experimental | ❌         | experimental mode disabled in this profile |

This is crucial for:

* debugging,
* user trust,
* and understanding why a given run behaves as it does.

---

## 11. Agent roles in model policy

### 11.1 Orchestrator Agent

* Builds the overall graph: data → features → training → predictions → analytics → tuning
* Decides:

  * when to call HPO,
  * when to run cross-asset V2 experiments,
  * which profiles to use per bucket.

### 11.2 HPO Agent

* Uses this **model policy** to:

  * pick eligible families per bucket,
  * handle tiering & hardware gating,
  * run HPO with the right metric profile,
  * produce ranked models and ensembles.

### 11.3 Analytics / Drift Agent

* Evaluates models:

  * overall metrics,
  * regime metrics (shock vs normal, high vs low vol),
  * cross-asset feature importance.
* Feeds results into:

  * Guardrail decisions,
  * config suggestions,
  * experiment/tuning (e.g. cross-asset V2).

### 11.4 Config / Governance Agent

* Proposes config changes based on:

  * model performance trends,
  * experiments (e.g. V2 ON/OFF),
  * resource constraints.
* Optionally:

  * applies patches with backup,
  * or waits for human confirmation.

### 11.5 Guardrail Agent

* Uses:

  * model metrics,
  * regime metrics,
  * cross-asset features,
  * model tier information
* to generate:

  * risk warnings,
  * constraints,
  * human-readable narratives about reliability and risk.

---

## 12. Summary

* The ModelZoo is **tiered**: core, advanced, experimental.
* Hardware & cost constraints are respected automatically.
* Different use cases use different **metric profiles**.
* Cross-asset V1/V2 features are central to advanced/global models.
* GNNs are best treated as **embedding providers** in prod.
* Retraining cadence varies by tier.
* Agents make the policy **operational**:

  * Orchestrator, HPO, Analytics, Config, Guardrail
* Every run should clearly log:

  * which models were used,
  * which were skipped,
  * and why.

This makes the agentic forecasting framework both **powerful and understandable**, for you and for any future collaborators.

```

If you’d like, I can also create a super-short `MODEL_POLICY_SUMMARY.md` (1–2 pages) you can drop into a slide deck or README as a high-level view, while this full document stays as the “source of truth” for how the zoo is governed.
::contentReference[oaicite:0]{index=0}
```
