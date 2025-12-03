# LLM Analytics Summary – 2025-11-28 (DAILY)

## Global Summary

The overall health of the forecasting system appears stable, with metrics showing reasonable ranges and variability. However, the directional accuracy is alarmingly low, with a mean of only 0.013, indicating that the model struggles to predict the correct direction of price movements. The MAE and RMSE values suggest that while the absolute errors are manageable on average, there are instances of significantly larger errors, as indicated by the maximum values. The lack of variability in feature importance suggests that the model may not be leveraging relevant features effectively, which could be a concern for future predictions.

## Metric Explanations

**MAE** – The mean absolute error (MAE) is 2.18, indicating that on average, the model's predictions deviate from actual values by this amount. However, the maximum MAE of 58.658 suggests that there are outliers or specific instances where the model performs poorly.
**MAPE** – The mean absolute percentage error (MAPE) is 0.039, which is relatively low, but the maximum value of 0.446 indicates that there are cases where the model's predictions are significantly off, potentially skewing the average.
**SMAPE** – The symmetric mean absolute percentage error (SMAPE) has a mean of 0.038, closely aligning with MAPE, which suggests consistent performance across predictions. However, the maximum value of 0.358 indicates that some predictions are highly inaccurate.
**MASE** – Not available.
**SWASE** – The mean SWASE is 0.041, indicating a slight bias in predictions, but the maximum value of 0.479 suggests that in certain instances, the model may be significantly underestimating or overestimating outcomes.
**DIRECTIONAL_ACCURACY** – The directional accuracy is extremely low, with a mean of 0.013, indicating that the model is correct in predicting the direction of price movement only 1.3% of the time on average. This is a critical issue that needs addressing.

## Regime Insights

### Regime: peer_shock_flag=0
- Performance: In this regime, the model shows a delta MAE increase of 3.2621, indicating worsening performance compared to other regimes. The directional accuracy remains unchanged.
- Risk: This regime is risky for deployment due to the increasing MAE, suggesting that the model's predictions are becoming less reliable.

## Feature Insights

### Overall Top Features
- **peer_mean_ret_10d** – Despite having zero importance, it indicates that the model is not utilizing this feature effectively, which could be a missed opportunity for better predictions.
- **sector_drawdown_60d** – Similar to the previous feature, its zero importance suggests it is not contributing to the model's predictive capability.
- **beta_vs_sector_60d** – Again, with zero importance, this feature is not being leveraged, indicating potential model tuning opportunities.
- **ret_20d_rank_in_sector** – This feature also shows zero importance, suggesting a lack of relevance in the current model setup.

## Recommendations

- **[FEATURES]** Investigate and tune the feature set to include more relevant predictors, as current features show zero importance.
  - Reason: The lack of feature importance indicates that the model may not be capturing key drivers of the target variable, which could improve performance.
- **[RISK]** Reassess the deployment of the model given the low directional accuracy and increasing MAE in the peer shock regime.
  - Reason: The model's reliability is questionable, and deploying it in its current state could lead to significant forecasting errors.
