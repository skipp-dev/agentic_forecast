# LLM Analytics Summary – 2025-11-28 (DAILY)

## Global Summary

The overall health of the forecasting system appears stable with low variability in metrics. The mean MAE of 2.18 and MAPE of 0.039 indicate reasonable accuracy, though the maximum values suggest potential outliers. Directional accuracy is notably low with a mean of 0.013, indicating frequent mispredictions in direction. The absence of critical calculation errors and reasonable ranges in metrics suggest no immediate data quality issues. However, the lack of feature importance in both overall and shock regimes raises concerns about model interpretability and effectiveness.

## Metric Explanations

**MAE** – The MAE shows a mean error of 2.18, which is acceptable, but the maximum of 58.658 indicates significant outliers that could skew performance assessments.
**MAPE** – The MAPE mean of 0.039 suggests good performance overall, but the maximum of 0.446 indicates that some predictions are significantly off, warranting further investigation.
**SMAPE** – SMAPE is consistent with MAPE, having a mean of 0.038 and a maximum of 0.358, indicating that while most predictions are close, there are instances of substantial error.
**SWASE** – The SWASE mean of 0.041 and maximum of 0.479 suggest that while errors are generally low, there may be bias or asymmetric errors present, particularly in shock regimes.
**DIRECTIONAL_ACCURACY** – The directional accuracy mean of 0.013 indicates that the model is rarely correct in predicting the direction of changes, which is a significant concern for deployment.

## Regime Insights

### Regime: peer_shock_flag=0
- Performance: In this regime, the performance metrics show a delta MAE increase of 3.2621, indicating a decline in accuracy compared to previous periods.
- Risk: This regime is risky for deployment due to the increase in MAE and low directional accuracy, suggesting that the model may not perform reliably.

## Feature Insights

### Overall Top Features
- **peer_mean_ret_10d** – Despite having zero importance, it is a common feature that could be revisited for potential relevance.
- **sector_drawdown_60d** – Similar to the previous feature, it shows zero importance, indicating it may not be contributing to model performance.
- **beta_vs_sector_60d** – This feature also shows zero importance, suggesting a need for reevaluation or replacement.
- **ret_20d_rank_in_sector** – With zero importance, this feature may not be providing valuable insights for predictions.

## Recommendations

- **[FEATURES]** Reassess the relevance of features with zero importance, particularly 'peer_mean_ret_10d', 'sector_drawdown_60d', 'beta_vs_sector_60d', and 'ret_20d_rank_in_sector'.
  - Reason: These features may not be contributing to model performance and could be replaced or enhanced.
- **[RISK]** Consider delaying deployment until directional accuracy improves significantly.
  - Reason: The current low directional accuracy poses a substantial risk to model reliability.
