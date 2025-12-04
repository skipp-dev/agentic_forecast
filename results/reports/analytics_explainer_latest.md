# LLM Analytics Summary – 2025-11-28 (DAILY)

## Global Summary

The overall health of the forecasting system appears stable, with reasonable ranges and variability in the metrics. However, the directional accuracy is alarmingly low, with a mean of only 0.013, indicating that the model is rarely predicting the correct direction. The MAE and RMSE values are relatively low, suggesting that when predictions are made, they are close to actual values, but the overall performance is undermined by the poor directional accuracy. The absence of significant feature importance may indicate a lack of effective predictors in the current model setup. The metrics show no critical calculation errors, but the low variability in directional accuracy raises concerns about the model's reliability across different symbols or regimes.

## Metric Explanations

**MAE** – The MAE has a mean of 2.18, which indicates that on average, the model's predictions are off by about 2.18 units. The maximum value of 58.658 suggests that there are instances of significant errors, but the overall mean indicates that these are not the norm.
**MAPE** – The MAPE mean of 0.039 indicates that the model's predictions are off by about 3.9% on average. However, the maximum value of 0.446 suggests that there are outlier predictions that are much worse, which could be affecting overall performance.
**SMAPE** – The SMAPE mean of 0.038 is consistent with the MAPE, indicating that the model's percentage errors are similarly low on average. However, the maximum value of 0.358 suggests that there are instances of significant percentage errors.
**MASE** – Not available
**SWASE** – The SWASE mean of 0.041 indicates low asymmetric errors, but the maximum of 0.479 suggests that there are instances where the model significantly underestimates or overestimates values, particularly in shock regimes.
**DIRECTIONAL_ACCURACY** – The mean directional accuracy of 0.013 indicates that the model is only correct in predicting the direction of change 1.3% of the time, which is extremely low. This suggests a fundamental issue with the model's ability to capture trends or movements in the data.

## Regime Insights

### Regime: peer_shock_flag=0
- Performance: In this regime, the directional accuracy remains unchanged, but the MAE has increased by 3.2621, indicating that predictions are becoming less accurate.
- Risk: This regime is risky for deployment due to the increasing MAE, which suggests that the model may be losing its predictive power over time.

## Feature Insights

### Overall Top Features
- **peer_mean_ret_10d** – Despite having an importance score of 0, it is noteworthy that this feature was included, indicating it may not be contributing to the model's performance.
- **sector_drawdown_60d** – Similar to the previous feature, its zero importance suggests it is not effectively capturing relevant information for predictions.
- **beta_vs_sector_60d** – This feature's lack of importance indicates that it may not be relevant for the current forecasting context.
- **ret_20d_rank_in_sector** – The zero importance score suggests that this feature is not aiding the model's predictive capabilities.

## Recommendations

- **[FEATURES]** Re-evaluate the feature set and consider introducing new features that may better capture market dynamics.
  - Reason: The current features show zero importance, indicating they are not contributing to the model's performance.
- **[HPO]** Run hyperparameter optimization to improve model performance, especially focusing on directional accuracy.
  - Reason: The extremely low directional accuracy suggests that the model is not effectively capturing trends, and tuning may help.
