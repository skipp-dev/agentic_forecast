# Metrics Quality Playbook
_for GuardrailAgent, QualityAgent & ReportingLLM_

This playbook describes **how the system should react** when metrics are:
- noisy,
- partially broken,
- or clearly unreliable.

It builds on the definitions in `METRICS_POLICY.md` and focuses on:
- **scenarios** (what went wrong),
- **detection** (how we know),
- **agent behavior** (what to do),
- **example messages** for human users.

---

## 1. Roles & Responsibilities

### 1.1 QualityAgent

**Goal:** Detect when metrics are suspicious or broken.

Responsibilities:
- Inspect evaluation tables (e.g. `evaluation_results_baseline_latest.csv`).
- Identify patterns such as:
  - extreme MAPE,
  - too few unique metric values,
  - identical metrics across symbols/horizons.
- Produce a **structured quality report** with:
  - `status` (`"passed"` / `"failed"`),
  - `severity` (`"low" | "medium" | "high"`),
  - `issues`: list of detailed findings,
  - `metrics_quality`: per-metric reliability labels.

### 1.2 GuardrailAgent

**Goal:** Use the quality report to decide **what is allowed**.

Responsibilities:
- Decide if:
  - model promotions,
  - HPO conclusions,
  - or risk decisions
  may proceed based on metric quality.
- Enforce **hard guardrails**:
  - e.g., block model rotation if MAE/DA metrics are unreliable.
- Attach warnings or "human review needed" flags when uncertainty is high.

### 1.3 ReportingLLM

**Goal:** Explain metric quality to humans.

Responsibilities:
- Read the metrics + quality report.
- Adjust narrative:
  - Emphasize metrics marked `"ok"`.
  - Downplay or explicitly mark `"unreliable"` metrics.
- Generate **clear, non-technical summaries** describing:
  - what is trustworthy,
  - what should be taken with caution,
  - how this impacts decisions.

---

## 2. Quality Levels & Labels

QualityAgent should produce:

```json
{
  "status": "failed" | "passed",
  "severity": "low" | "medium" | "high",
  "issues": [...],
  "metrics_quality": {
    "mae": "ok" | "suspect" | "unreliable",
    "rmse": "ok" | "suspect" | "unreliable",
    "mape": "ok" | "suspect" | "unreliable",
    "smape": "ok" | "suspect" | "unreliable",
    "mase": "ok" | "suspect" | "unreliable",
    "directional_accuracy": "ok" | "suspect" | "unreliable"
  }
}
```

### Suggested meanings

* **ok**
  Values look plausible and vary across symbols/horizons as expected.

* **suspect**
  Some warning signs:

  * unusual distribution,
  * partial identical values,
  * odd spikes – but not completely broken.

* **unreliable**
  Known serious problems:

  * many NaNs/inf,
  * fallback behavior,
  * identical across many symbols/horizons,
  * or structural bug suspected.

---

## 3. Scenario Playbook

Each scenario has:

* **Detection** – what QualityAgent sees
* **Guardrail behavior** – what actions are allowed/denied
* **Reporting behavior** – how to explain it

### Scenario A – MAPE is unreliable, others are fine

**Typical trigger**

* MAPE shows:

  * many values ≥ 0.99,
  * NaNs replaced or fallback,
  * or identical MAPE across symbols for a given horizon.

**Detection**

QualityAgent issues:

```json
{
  "metrics_quality": {
    "mae": "ok",
    "rmse": "ok",
    "mape": "unreliable",
    "directional_accuracy": "ok"
  },
  "issues": [
    {"type": "mape_high", "count": 37},
    {"type": "mape_identical_across_symbols", "horizon": 1}
  ],
  "status": "failed",
  "severity": "medium"
}
```

**GuardrailAgent behavior**

* Drop MAPE from:

  * overall score calculation, or
  * reduce its weight to near-zero.
* **Do not** base model promotion decisions on MAPE.
* Allow promotions if:

  * MAE, RMSE, MASE, DA are `"ok"`.
* Attach a "partial quality issue" note to the promotion decision.

**ReportingLLM behavior**

Emphasize:

* MAE, RMSE, MASE, DA
* Explain clearly that MAPE is **not trusted** for this run.

Example narrative:

> "In this evaluation, Mean Absolute Error (MAE ≈ 0.016) and RMSE (≈ 0.020) behave normally across symbols and horizons.
> However, MAPE is unreliable for 37 symbols due to near-zero prices or adjustment artifacts, so it is **not used** for scoring or model selection in this report.
> The conclusions are therefore based on MAE, RMSE, MASE and Directional Accuracy."

---

### Scenario B – Identical metrics across multiple symbols

**Typical trigger**

* All symbols at a given horizon share the same MAE/RMSE/MAPE.

**Detection**

QualityAgent issues:

```json
{
  "issues": [
    {
      "type": "identical_metrics_per_horizon",
      "horizon": 1,
      "metric": "mae"
    }
  ],
  "metrics_quality": {
    "mae": "unreliable",
    "rmse": "suspect",
    "mape": "unreliable"
  },
  "status": "failed",
  "severity": "high"
}
```

**GuardrailAgent behavior**

* Treat this run as **structurally compromised** for that horizon:

  * Block automatic model rotations based on horizon=1 metrics.
  * Disallow champion changes triggered by this evaluation.
* Mark evaluation as:

  * "Requires manual review."
* Optionally:

  * Allow other horizons (3d, 5d, etc.) to be evaluated independently if their metrics are fine.

**ReportingLLM behavior**

Explain clearly:

> "On the 1-day horizon, all symbols share identical MAE and MAPE values. This indicates that metrics were likely computed globally and broadcast to all symbols, rather than calculated per symbol.
> As a result, the 1-day metrics are **not reliable** and are excluded from model selection decisions in this report. Longer-horizon metrics (3/5/10 days) remain valid and are used for analysis."

---

### Scenario C – All metrics show low diversity (few unique values)

**Typical trigger**

* MAE or RMSE has fewer than N unique values across hundreds of symbols.

**Detection**

QualityAgent issues:

```json
{
  "issues": [
    {
      "type": "few_unique_values",
      "metric": "mae",
      "unique_values": 3
    }
  ],
  "metrics_quality": {
    "mae": "suspect",
    "rmse": "ok",
    "directional_accuracy": "ok"
  },
  "status": "failed",
  "severity": "medium"
}
```

**GuardrailAgent behavior**

* Mark MAE as `"suspect"` for this run:

  * still visible,
  * but **do not use it alone** to drive decisions.
* Prefer robust combinations:

  * e.g., require **agreement** between RMSE, DA, and MASE before rotating models.
* If other metrics are `"ok"`, **soft allow** decisions but annotate with caution.

**ReportingLLM behavior**

> "MAE values show unusually low variation across symbols, which may indicate aggregation issues or low sensitivity.
> We still include MAE in the tables for completeness, but rely more heavily on RMSE, MASE and Directional Accuracy when drawing conclusions for this run."

---

### Scenario D – Baseline / MASE indicates model worse than naive

**Typical trigger**

* MASE > 1.0 across many symbols or for a champion model.

**Detection**

QualityAgent issues:

```json
{
  "issues": [
    {
      "type": "mase_above_one",
      "symbols": 42,
      "context": "champion_model"
    }
  ],
  "metrics_quality": {
    "mase": "ok",
    "mae": "ok",
    ...
  },
  "status": "passed",
  "severity": "low"
}
```

(This is not a *metrics quality* failure; it's a **performance** signal.)

**GuardrailAgent behavior**

* Enforce guardrails:

  * "No model with median MASE > 1.0 should be champion."
* Suggest:

  * re-run HPO,
  * try a different model family,
  * or fall back to a simpler baseline.

**ReportingLLM behavior**

> "On several symbols, the current champion model's error is worse than a simple naive baseline when measured by MASE (> 1.0).
> This suggests that the model is not adding predictive value over a basic last-value forecast. A new training run or model family exploration is recommended before relying on these forecasts."

---

### Scenario E – Directional accuracy is unstable or near random

**Typical trigger**

* DA ~ 0.5 on a directional problem (e.g. classification of up/down),
* or big swings across runs.

**Detection**

QualityAgent issues:

```json
{
  "issues": [
    {
      "type": "low_directional_accuracy",
      "horizon": 1,
      "value": 0.52
    }
  ],
  "metrics_quality": {
    "directional_accuracy": "ok",
    "mae": "ok"
  },
  "status": "passed",
  "severity": "low"
}
```

(Metrics are **valid**, but performance is weak.)

**GuardrailAgent behavior**

* For strategies where direction matters:

  * e.g. "do not promote models with DA < 0.55 on the key horizon".
* Suggest more HPO, feature enrichment, or alternative architectures.

**ReportingLLM behavior**

> "While absolute error metrics (MAE/RMSE) are acceptable, the models' ability to get the direction right is only slightly better than random (~52% correct for next-day moves).
> For directional trading strategies, this level of directional accuracy should be treated with caution; further model improvements or feature enhancements are advisable before acting on those signals."

---

## 4. Default Agent Behaviors

### 4.1 GuardrailAgent – Decision Matrix (Simplified)

| Condition                                                        | Action                                                                        |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `status = failed` AND `severity = high`                          | Block automatic model rotations; require manual review                        |
| `mape = "unreliable"` BUT `mae`, `rmse`, `mase`, `da` are `"ok"` | Drop/ignore MAPE in decisions; continue based on the other metrics            |
| Identical metrics per horizon for ≥ 2 metrics                    | Treat that horizon as invalid; do not use it for decisions                    |
| `mase > 1.0` for champion on many symbols                        | Trigger warning; recommend HPO or model change; optionally block promotion    |
| `da < threshold` on key horizon (e.g. < 0.55)                    | Soft-block directional strategies; require explicit override or manual review |
| Only minor issues (few symbols affected, severity = low)         | Allow decisions; attach caution message                                       |

### 4.2 ReportingLLM – Narrative Guidelines

When quality issues exist:

* Always explicitly state:

  * **What’s wrong** (e.g., "MAPE unreliable", "metrics identical across symbols").
  * **What is still trusted** (e.g., "MAE/DA stable and used for conclusions").
  * **How decisions are adjusted** (e.g., "we excluded MAPE from the score").

Keep language:

* Simple,
* Honest,
* Non-alarmist but clear.

Example template:

> "Some metrics in this run show quality issues (details below), so the report emphasizes the more robust metrics and does not rely on the broken ones for decisions."

---

## 5. Implementation Hints (Conceptual)

* QualityAgent should:

  * run **after** each evaluation,
  * produce a JSON report under `data/metrics/quality_report_latest.json`.

* GuardrailAgent should:

  * read this JSON,
  * compute a **decision object**, e.g.:

    ```json
    {
      "can_rotate_models": false,
      "reason": "identical MAE across symbols on 1-day horizon",
      "warnings": [...]
    }
    ```

* ReportingLLM should:

  * get both:

    * the metrics aggregates,
    * the quality report,
  * and build its narrative accordingly.