# LLM Analytics Summary – 2025-11-28 (DAILY)

## Global Summary

The overall health of the forecasting system appears to be low-risk, with metrics showing reasonable ranges and variability. However, the directional accuracy is alarmingly low, with a mean of only 0.013, indicating that the model struggles significantly to predict the correct direction of changes. The MAE and RMSE values are also low on average, but the maximum values suggest potential outliers or specific instances of poor performance. The feature importance metrics indicate that the features currently used are not contributing effectively, as they all have an importance score of zero.

## Metric Explanations

**MAE** – The MAE has a mean of 2.18, which suggests that on average, the model's predictions are reasonably close to actual values. However, the maximum MAE of 58.658 indicates that there are instances where the model performs poorly, which could skew overall performance.
**MAPE** – The MAPE mean of 0.039 suggests that the model's percentage error is low on average, but the maximum value of 0.446 indicates that there are significant outliers that need to be addressed.
**SMAPE** – SMAPE shows a mean of 0.038, which aligns closely with MAPE, indicating consistent performance across absolute and relative error metrics. The maximum value of 0.358 suggests that there are instances of high error that need further investigation.
**SWASE** – The mean SWASE is 0.041, which is low, but the maximum of 0.479 indicates potential bias or asymmetric errors that could be problematic in specific regimes.
**DIRECTIONAL_ACCURACY** – With a mean of 0.013, the directional accuracy is extremely low, suggesting that the model fails to predict the correct direction of changes most of the time. This could be a critical issue for deployment.

## Regime Insights

### Regime: peer_shock_flag=0
- Performance: In this regime, the model shows a delta MAE increase of 3.2621, indicating worsening performance compared to other regimes.
- Risk: This regime is risky for deployment due to the significant increase in MAE, suggesting that the model may not generalize well under these conditions.

## Feature Insights

### Overall Top Features
- **peer_mean_ret_10d** – Despite having an importance score of 0, this feature is typically relevant in predicting returns based on peer performance.
- **sector_drawdown_60d** – This feature is expected to capture sector-wide risks but currently shows no contribution to model performance.
- **beta_vs_sector_60d** – This feature usually helps in understanding volatility relative to the sector but is ineffective in the current model.
- **ret_20d_rank_in_sector** – This feature should help in ranking performance but is not contributing to the model's predictions.

## Recommendations

- **[FEATURES]** Re-evaluate and replace features with zero importance scores.
  - Reason: The current features are not contributing to model performance, indicating a need for better feature selection.
- **[HPO]** Run hyperparameter optimization to improve directional accuracy.
  - Reason: The extremely low directional accuracy suggests that the model's parameters may not be well-tuned.
- **[RISK]** Monitor performance closely in the peer_shock_flag=0 regime before deployment.
  - Reason: The increase in MAE in this regime indicates potential risks that need to be addressed.
