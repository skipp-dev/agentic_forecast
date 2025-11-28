# Daily Forecast Health Report

**Run ID:** demo-daily-run-2025-01-28
**Evaluated at:** 2025-11-28T21:11:24.743240Z
**Symbols:** AAPL, MSFT, NVDA, TSLA, GOOGL
**Horizons:** 1, 5, 10

---

## 1. Metric Sanity

- **Status:** ✅ PASSED (🟢 Low)
- **Summary:** Metric sanity check passed with low severity.
- **Key Findings:**
  - All metrics show reasonable ranges and variability.
  - SMAPE and SWASE have appropriate unique value counts.
  - No critical calculation errors detected.

---

## 2. Model Performance (High Level)

- **Overall score:** 0.778 (0–1 scale, higher is better)
- **Headline metrics (average across symbols & horizons):**
  - MAE: 2.18
  - RMSE: 2.373
  - MAPE: 0.039
  - SMAPE: 0.038
  - SWASE: 0.041
  - DIRECTIONAL_ACCURACY: 0.013

- **Model comparison:**
  - **naive**: candidate

---

## 3. Cross-Asset Features V2

- **Status:** ✅ Enabled
- **Overall effect:** V2 improves overall performance.
- **Performance lift:**
  - delta_mae: +3.2679
  - delta_mape: +0.0340
  - delta_directional_accuracy: +0.0000

- **By regime:**
  - peer_shock_flag=0: {'delta_mae': 3.2621, 'delta_mape': 0.0338, 'delta_directional_accuracy': 0.0}

- **Most influential new features:**
  - `peer_mean_ret_10d`
  - `sector_drawdown_60d`
  - `beta_vs_sector_60d`
  - `ret_20d_rank_in_sector`

**Decision:** → Keep V2 **keep enabled**

---

## 4. Alerts & Automated Decisions

- **Auto-promotion of models:** ✅ **Enabled**
  - Reason: Metric sanity passed and performance meets thresholds.
- **Recommended engineering actions:**
  1. Proceed with automated model promotion.
  2. Monitor performance in production.
  3. Continue regular evaluation cycles.

---

## 5. Short Executive Summary

> The forecasting system shows metric sanity checks passed, strong model performance, V2 features providing benefits. The system is ready for automated operations.
