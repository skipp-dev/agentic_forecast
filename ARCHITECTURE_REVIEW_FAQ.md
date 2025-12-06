# Architecture Review FAQ

**Date:** December 6, 2025
**Context:** Senior ML/Quant Platform Architect Review

This document captures key architectural decisions, known limitations, and the "honest state" of the system regarding data, validation, and resilience.

---

## 1️⃣ “What is our *Golden Dataset*?”

> **Context:** Do we have a static, versioned dataset to benchmark every model change, or are we always fetching fresh data?

**Honest state today**

* You *do* run backtests, but they pull from **live-ish historical data** (vendor or local cache).
* There is **no small, frozen “golden dataset”** with:
  * a fixed set of symbols,
  * fixed time window,
  * and stored “expected” metrics.
* So if you change code or dependencies and rerun, you might see different results and not be sure whether:
  * the data changed,
  * the model changed,
  * or the backtester changed.

**Target State**

A proper Golden Dataset package:
* A small, versioned bundle under `tests/golden_dataset/`:
  * e.g. 5–10 symbols, 3–5 years of daily bars,
  * plus any macro/news/fundamental features you use.
* A fixed config, e.g. `configs/golden_backtest.yml`.
* Stored expected outputs:
  * summary metrics (avg MAE, DA, Sharpe, max DD, # trades),
  * maybe a few specific forecasts/weights for key dates.

Then in CI you run:
```bash
agentic_forecast backtest --config configs/golden_backtest.yml --data tests/golden_dataset/
```
and assert metrics match within tolerances.

> **Answer:** “Right now we don’t have a Golden Dataset; backtests use live historical data. We’re planning a small, frozen, versioned dataset + config that every code change must pass, so we can detect regressions and keep behaviour reproducible.”

---

## 2️⃣ “How do we validate the *Spectral Features*?”

> **Context:** `torch.fft` is cool, but have we proven FFT/frequency-domain features actually add alpha over plain moving averages for this asset class?

**Honest state today**

* Spectral features are effectively **experimental**:
  * implemented in the feature stack,
  * but not rigorously justified with **ablation tests**.
* There’s no written evidence like “adding FFT features improves out-of-sample Sharpe by X% on US equities”.
* So: right now you **cannot honestly claim** they add alpha; they may just add complexity.

**Target Validation**

A clean ablation study:
1. Define three feature sets:
   * **Baseline**: price returns, SMAs/EMAs, volatility.
   * **Baseline + Spectral**: add FFT-derived features.
   * **Spectral-only** (optional sanity check).
2. For a fixed universe + horizon, train the same model family on each feature set.
3. Compare **out-of-sample** metrics (MAE, Sharpe, stability).

> **Answer:** “Spectral features exist but are experimental; we haven’t yet proved they add alpha beyond simple moving averages. We need an ablation (baseline vs baseline+spectral) over a fixed golden dataset, and only keep them in the production feature view if they demonstrate consistent out-of-sample improvement.”

---

## 3️⃣ “What happens when the *Orchestrator* crashes?”

> **Context:** LangGraph has checkpoints, but are we persisting them? If the container dies, do we resume or restart?

**Honest state today**

* LangGraph supports checkpointing, but by default we use **in-memory** storage.
* If the container dies mid-run:
  * in practice you **lose the in-memory state** of that run.
  * you can restart the run from scratch, but you don’t “resume from step 17”.

**Target State**

* Use a **persistent checkpointer** (Postgres/Redis).
* Each agent transition writes state to DB.
* On restart, orchestrator loads last checkpoint and resumes idempotently.

> **Answer:** “Right now, if the orchestrator/container dies, we treat the run as failed and start over; we don’t yet resume from checkpoints. Production would use LangGraph’s persistent checkpointer backed by Postgres/Redis so we can resume safely after a crash rather than redoing the entire flow.”

---

## 4️⃣ “Who watches the *watchers*?”

> **Context:** DriftMonitorAgent checks the model. Who checks if the DriftMonitorAgent is broken or drifting itself?

**Honest state today**

* DriftMonitorAgent is trusted by default.
* There’s no separate agent or test harness continuously verifying its behaviour.
* If it silently stops updating or produces nonsense, we might not notice.

**Target State**

**Meta-monitoring + tests**, not another LLM agent:
1. **Make DriftMonitor deterministic and simple** (Pure math, no LLM core).
2. **Unit and regression tests** (Synthetic scenarios in CI).
3. **Meta-metrics** (Log `drift_monitor_runs_total`, `drift_flags_raised_total`).
4. **Guardrail on the guardrail** (If monitor errors, alert RED and fallback, don't kill models).

> **Answer:** “DriftMonitorAgent itself is treated as deterministic, tested code, not an LLM black box. We add unit tests, Prometheus metrics, and health checks for the monitor itself. If it stops running or behaves oddly, we raise a separate ‘monitor health’ alert and avoid automatic drastic actions until it’s fixed.”
