# Metric Sanity Report – Forecast Evaluation

**Run ID:** 2025-11-28T21-02-30Z
**Evaluated at:** 2025-11-28T21:02:30.202926Z
**Source file:** `data/metrics/evaluation_results_baseline_latest.csv`
**Rows:** 1972 – **Symbols:** 986 – **Horizons:** 1, 5

---

## Overall Status
- **Status:** ❌ FAILED
- **Severity:** 🟢 Low
- **Issue count:** 1

**Summary:**
Found 1 quality issues

---

## Metric Summaries

### MAE
- Mean: `2.18`
- Std: `4.544`
- Min / Max: `0.0` / `58.658`
- Unique values: `832`
- Issues: _None detected_

### MAPE
- Mean: `0.039`
- Std: `0.04`
- Min / Max: `0.0` / `0.446`
- Unique values: `980`
- Issues: _None detected_

### SMAPE
- Mean: `0.038`
- Std: `0.038`
- Min / Max: `0.0` / `0.358`
- Unique values: `981`
- Issues: _None detected_

### SWASE
- Mean: `0.041`
- Std: `0.043`
- Min / Max: `0.0` / `0.479`
- Unique values: `981`
- Issues: _None detected_

### DIRECTIONAL_ACCURACY
- Mean: `0.013`
- Std: `0.067`
- Min / Max: `0.0` / `1.0`
- Unique values: `3`
- Issues: _None detected_

---

## Symbol Examples

- **ACDC_daily @ Horizon 5**
  - SWASE = 0.10 – far above peer average (0.04).

- **ACDC @ Horizon 5**
  - SWASE = 0.10 – far above peer average (0.04).

- **ACHR_daily @ Horizon 5**
  - SWASE = 0.10 – far above peer average (0.04).

- **ACHR @ Horizon 5**
  - SWASE = 0.10 – far above peer average (0.04).

- **ADAP_daily @ Horizon 1**
  - SWASE = 0.18 – far above peer average (0.04).

---

## Recommended Follow-Up

1. **SMAPE / SWASE implementation**
   - Review the SMAPE formula and ensure the denominator uses `(|pred| + |actual| + eps)`.
   - Confirm SWASE uses scaled errors with correct shock weights.
2. **Per-horizon loops**
   - Verify that metrics are computed per `(symbol, horizon)` and not accidentally broadcast or re-used.
3. **Evaluation dataset**
   - Inspect raw rows where SMAPE > 2.0 or SWASE > 3.0.
   - Check for zero or tiny actuals, or missing/incorrect shock flags.

---