# LLM Analytics Summary – 2025-11-28 (DAILY)

## Global Summary

The overall health of the forecasting system appears stable, with metrics showing reasonable ranges and variability. However, the directional accuracy is alarmingly low, with a mean of only 0.013, indicating that the model struggles to predict the correct direction of price movements. The MAE and RMSE values are relatively low, suggesting that absolute errors are manageable, but the high maximum values indicate potential outliers. The absence of significant feature importance suggests that the model may not be leveraging relevant predictors effectively, which could be a point of concern for future improvements.

## Metric Explanations

**MAE** – The mean absolute error (MAE) is 2.18, which is acceptable, but the maximum value of 58.658 indicates that there are instances of significant errors. This suggests that while the model performs well on average, it may struggle with certain symbols or conditions.
**MAPE** – The mean absolute percentage error (MAPE) is 0.039, which is relatively low, but the maximum value of 0.446 indicates that some predictions are significantly off. This could be due to outliers or specific market conditions that the model fails to capture.
**SMAPE** – The symmetric mean absolute percentage error (SMAPE) has a mean of 0.038, closely aligning with MAPE, which indicates consistent performance across predictions. However, the maximum value of 0.358 suggests that there are instances where the model's predictions diverge significantly from actual values.
**MASE** – Not available.
**SWASE** – The mean SWASE is 0.041, indicating low bias in predictions, but the maximum of 0.479 suggests that there are periods of asymmetric errors, particularly in shock regimes.
**DIRECTIONAL_ACCURACY** – The directional accuracy is extremely low at a mean of 0.013, with a maximum of 1.0, indicating that while some predictions are correct, the vast majority are not. This poses a significant risk for deployment, as the model fails to predict the direction of price movements reliably.

## Regime Insights

### Regime: peer_shock_flag=0
- Performance: In this regime, the model shows a delta MAE increase of 3.2621, indicating a deterioration in performance compared to other regimes.
- Risk: This regime is risky for deployment due to the increased MAE, suggesting that the model may not generalize well under these conditions.

## Feature Insights

### Overall Top Features
- **peer_mean_ret_10d** – Despite having zero importance, it is noteworthy that this feature is included, indicating it may not contribute to the model's predictive power.
- **sector_drawdown_60d** – Similar to the previous feature, its zero importance suggests it is not being utilized effectively in the model.
- **beta_vs_sector_60d** – The lack of importance indicates that this feature may not be relevant for the current model setup.
- **ret_20d_rank_in_sector** – Again, zero importance suggests that this feature does not aid in improving model predictions.

## Recommendations

- **[FEATURES]** Re-evaluate the feature set and consider adding more relevant predictors to improve model performance.
  - Reason: The current features show zero importance, indicating that the model may not be leveraging available data effectively.
- **[HPO]** Run hyperparameter optimization for the current model to explore potential improvements in predictive accuracy.
  - Reason: Given the low directional accuracy and the variability in MAE and RMSE, tuning the model could yield better performance.
