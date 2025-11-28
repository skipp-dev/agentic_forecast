# Metrics Policy & Evaluation Guide
_for the `agentic_forecast` platform_

## 1. Purpose

This document defines:

- **Which metrics** we use (MAE, RMSE, MAPE, SMAPE, MASE, Directional Accuracy, etc.)
- **How** they are computed and interpreted
- **When** they are considered reliable or unreliable
- **How** they flow into:
  - global scores,
  - quality checks,
  - guardrails,
  - and reporting/LLM explanations.

The goal is to make metrics **transparent, robust, and user-friendly**, especially for non-quants.

---

## 2. Core Metrics

### 2.1 MAE – Mean Absolute Error

**Definition**

\[
\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|
\]

- \( y_i \): actual value
- \( \hat{y}_i \): model prediction
- \( N \): number of samples

**Interpretation**

- Average **absolute deviation** between prediction and reality.
- Same **unit** as the target (e.g., normalized price, scaled returns).
- Lower is better.

**Usage**

- Primary *scale-dependent* error metric.
- Computed per `(symbol, horizon)` and aggregated globally.
- Used in:
  - per-symbol evaluation tables,
  - global performance summary,
  - overall model score.

---

### 2.2 RMSE – Root Mean Squared Error

**Definition**

\[
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}
\]

- Quadratic penalty on large errors.

**Interpretation**

- Similar to MAE, but **penalizes outliers** more strongly.
- Same unit as the target.
- Lower is better.
- RMSE ≥ MAE in typical scenarios.

**Usage**

- Complement to MAE to understand the impact of large mistakes.
- Used in:
  - per-symbol evaluation,
  - global performance summary,
  - overall model score.

---

### 2.3 MAPE – Mean Absolute Percentage Error (fragile)

**Definition (ideal)**

\[
\text{MAPE} = \frac{100}{N} \sum_{i=1}^N \left|\frac{y_i - \hat{y}_i}{y_i}\right|
\]

**Interpretation**

- Average **percentage error** relative to the actual value.
- 0.1 → ~10% error, 1.0 → ~100% error.

**Known Problems**

MAPE is **numerically fragile** when:

- actual values \( y_i \) are **zero or very small**,
- there are stock splits or extreme jumps,
- targets are **scaled/normalized** such that near-zero values are common.

This can cause:

- division by ~0,
- infinite or NaN values,
- distorted averages.

**Platform Policy**

- MAPE is considered **optional and fragile**.
- If we cannot compute a numerically stable MAPE, it is **marked as unreliable**, not blindly trusted.
- We do **not** rely solely on MAPE for model selection.

**Computation Safeguards**

When computing MAPE:

1. Use a minimum denominator:
   - Replace very small |\(y_i\)| by a small epsilon **in target units**.
2. Exclude known bad points (if available):
   - days with splits / extreme jumps.
3. If no valid points remain:
   - MAPE is set to `NaN`,
   - and a `mape_flag = "unreliable"` is recorded.

**Usage**

- Only used when its quality is acceptable.
- Quality is assessed by **QualityAgent** (see Section 4).
- If MAPE is flagged as unreliable, it is **dropped or downweighted** in the overall score and clearly marked in reports.

---

### 2.4 SMAPE – Symmetric Mean Absolute Percentage Error (recommended alternative)

**Definition**

\[
\text{SMAPE} = \frac{2}{N} \sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i| + \epsilon}
\]

- \( \epsilon \): small constant to avoid division by zero.

**Interpretation**

- Similar to MAPE, but denominator uses **actual + predicted**, not only actuals.
- Better behaved when values are near zero.
- Bounded (0 to ~2).

**Usage**

- Preferred percentage-type metric over MAPE when possible.
- More robust for normalized data and low price levels.
- Candidate to replace MAPE in the long term.

---

### 2.5 MASE – Mean Absolute Scaled Error (scale-free benchmark)

**Definition**

1. Compute the **naive error scale** using a random-walk baseline:

\[
Q = \frac{1}{N-1} \sum_{t=2}^N |y_t - y_{t-1}|
\]

2. Then:

\[
\text{MASE} = \frac{\text{MAE}}{Q}
\]

**Interpretation**

- MASE < 1 → model is **better than naive** (on average).
- MASE > 1 → model is **worse than naive**.
- Scale-free: comparable across symbols and horizons.

**Usage**

- Natural complement to MAE and RMSE.
- Makes "better than naive" comparison explicit.
- Particularly useful in reporting and guardrails:
  - "No model with MASE > 1.1 should be promoted to champion."

---

### 2.6 Directional Accuracy (DA)

**Definition**

We compare the **sign of changes**:

- Actual change: \( \Delta y_i = y_{i} - y_{i-1} \)
- Predicted change: \( \Delta \hat{y}_i = \hat{y}_{i} - \hat{y}_{i-1} \)

Directional Accuracy:

\[
\text{DA} = \frac{1}{N'} \sum_{i} \mathbb{1}\left(\text{sign}(\Delta y_i) = \text{sign}(\Delta \hat{y}_i)\right)
\]

- \( N' \): number of valid pairs.
- Values between 0 and 1.

**Interpretation**

- Fraction of times the model gets the **direction** (up/down) right.
- Particularly important for trading/risk use cases.

**Usage**

- First-class metric in evaluation and scoring.
- Used in:
  - symbol-level evaluation tables,
  - global performance reporting,
  - guardrails (e.g. stable DA above random baseline).

---

## 3. Where Metrics Are Computed

### 3.1 Per-Symbol Evaluation

- For each `(symbol, horizon)`:
  - A naive baseline (last-value persistence) is evaluated.
  - Metrics (MAE, RMSE, MAPE/SMAPE, MASE, DA) are computed using **adjusted prices** where available.
- Results are written to:
  - `data/metrics/evaluation_results_baseline_latest.csv` (and similar files for models).

Each row typically contains:

- `symbol`
- `target_horizon` / `horizon`
- `mae`, `rmse`, `mape` and/or `smape`, `mase`, `directional_accuracy`
- optional flags: `mape_flag`, `metric_quality_flag`, etc.

---

### 3.2 Global Aggregation (performance_reporting)

The **Performance Reporting** component:

- loads recent metrics,
- groups by `metric_name`,
- computes:
  - mean,
  - std,
  - min,
  - max,
  - count,
  - simple trend (last vs first value).

This produces a structure:

```python
performance = {
  "mae": {...},
  "rmse": {...},
  "mape": {...},
  "directional_accuracy": {...},
  "smape": {...},  # optional
  "mase": {...}    # optional
}
```

These aggregates feed:

* the global summary in reports,
* the **Overall Model Score**.

---

## 4. Metric Quality & Diagnostics

### 4.1 QualityAgent Responsibilities

The **QualityAgent** inspects evaluation metrics to detect issues such as:

* MAPE values ≥ 0.99 (indicative of:

  * near-zero denominators,
  * stock splits or data anomalies,
  * or fallbacks / unreliable calculations)
* too few unique values for a metric across symbols

  * e.g. `Only 3 unique MAE values` across hundreds of symbols → likely aggregation or join bug.
* all symbols having identical metrics for a given horizon

  * e.g. horizon 1: all `mae` equal → global computation incorrectly broadcast to all symbols.

It returns a structured result, for example:

```json
{
  "status": "failed",
  "severity": "high",
  "issues": [
    {"type": "mape_high", "count": 37},
    {"type": "identical_metrics_per_horizon", "horizon": 1, "metric": "mae"}
  ],
  "metrics_quality": {
    "mae": "ok",
    "rmse": "ok",
    "mape": "unreliable",
    "directional_accuracy": "ok"
  }
}
```

### 4.2 Metric Flags

When possible, each metrics row is annotated with flags, e.g.:

* `mape_flag`:

  * `"ok"` – numerically stable and in a plausible range
  * `"high"` – MAPE very large but still considered valid
  * `"unreliable"` – numeric instability or fallback
* `metric_quality_flag`:

  * `"normal"`, `"suspect"`, `"bad"` (based on QualityAgent checks)

These flags are used downstream by:

* GuardrailAgent
* ReportingLLM
* UI/Dashboards (color coding, warnings, tooltips)

---

## 5. Overall Model Score

### 5.1 Baseline Scoring Logic

The **Overall Model Score** aggregates metric performance using a weighted scheme:

* Example weights:

  * MAE: 0.3
  * RMSE: 0.3
  * MAPE / SMAPE: 0.2
  * Directional Accuracy: 0.2

* Directional Accuracy:

  * used as-is: higher is better (0–1).

* Error metrics:

  * transformed using `1 - mean_error` (after normalization), so lower error → higher score.

### 5.2 Handling Unreliable Metrics

If QualityAgent marks a metric as **unreliable** (e.g. `mape`):

* That metric is **dropped or heavily downweighted** from the score for that run.
* The remaining weights are **renormalized**.
* The score is accompanied by a warning, e.g.:

> "MAPE was excluded from the overall score for this run due to quality issues (37 symbols with unstable MAPE). The score is based on MAE, RMSE and Directional Accuracy only."

This prevents one broken metric from **poisoning the entire score**.

---

## 6. Guardrails & Reporting

### 6.1 GuardrailAgent

The **GuardrailAgent** consumes:

* Symbol-level metrics,
* QualityAgent result,
* global aggregates.

It enforces rules such as:

* "Do not promote a new champion model if:

  * MASE > 1.0 on key horizons, or
  * Directional Accuracy < baseline, or
  * metrics for this run failed quality checks."
* "If MAPE is unreliable, ignore it when deciding model rotations."

Guardrails can be **hard** (block actions) or **soft** (warnings requiring human review).

---

### 6.2 ReportingLLM & User-Facing Explanations

The **Reporting LLM** receives:

* aggregated metrics + quality flags,
* guardrail outcomes,
* score components.

It generates **human-friendly summaries**, for example:

> "For this evaluation run, models maintain stable performance on the 1-day horizon with MAE ≈ 0.016 and RMSE ≈ 0.020 across the main equity universe.
>
> MAPE was flagged as unreliable for 37 symbols due to near-zero prices or adjustment artifacts, so this report emphasizes MAE, RMSE and Directional Accuracy instead.
>
> Directional Accuracy remains above 60%, indicating that short-term direction is captured reasonably well, even though percentage-based error metrics should be interpreted with caution in this dataset."

This ensures non-technical users:

* understand **what the metrics mean**,
* know **which metrics are trustworthy right now**,
* and see **why** certain metrics are omitted or downweighted.

---

## 7. Best Practices & Recommendations

* **Always look at multiple metrics**:

  * MAE / RMSE for magnitude,
  * SMAPE / MAPE for percentage context,
  * MASE for "better than naive",
  * DA for directional correctness.
* **Treat MAPE with suspicion** when:

  * prices/targets are near zero,
  * there are many splits or structural breaks.
* **Respect QualityAgent outputs**:

  * if metrics are flagged as unreliable, treat decisions based on them as suspect.
* **Use MASE and DA** as robust anchors:

  * they often remain meaningful even when percentage metrics are unstable.
* **Document metric changes**:

  * if definitions or normalization change, update this policy and annotate runs accordingly.

---

## 8. Future Improvements

Planned or potential enhancements include:

* More systematic use of SMAPE and MASE across all pipelines.
* Regime-aware metrics (per volatility regime, shock flags, cross-asset context).
* Tail-focused metrics (e.g. error on shock days vs normal days).
* Per-bucket metric profiles:

  * different metric priorities for:

    * low-risk vs high-beta symbols,
    * short vs long horizons,
    * cross-asset/global vs symbol-level models.

These will be reflected in this policy as they are introduced.