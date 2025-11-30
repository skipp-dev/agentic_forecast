# LLM Analytics Summary – 2025-11-28 (DAILY)

## Global Summary

The stock forecasting system is currently performing within reasonable ranges, with low mean errors across various metrics. However, the directional accuracy is notably low, indicating challenges in predicting the correct direction of stock movements.

## Metric Explanations

**MAE** – Mean Absolute Error (MAE) measures the average magnitude of errors in predictions, without considering their direction. A lower MAE indicates better predictive accuracy.
**MAPE** – Mean Absolute Percentage Error (MAPE) expresses prediction errors as a percentage of actual values. It helps gauge the accuracy relative to the size of the actual values, with lower percentages indicating better performance.
**SMAPE** – Symmetric Mean Absolute Percentage Error (SMAPE) is similar to MAPE but adjusts for scale, providing a more balanced view of prediction accuracy. It is particularly useful when actual values can be close to zero.
**SWASE** – Symmetric Weighted Absolute Scaled Error (SWASE) is a scaled version of the absolute error that accounts for the variability in the data. It helps to understand how well the model performs relative to the variability of the actual values.

## Regime Insights

### Regime: normal
- Performance: The model shows a slight increase in MAE and MAPE during normal market conditions, suggesting some degradation in performance.
- Risk: The low directional accuracy indicates a significant risk in predicting stock movements accurately.

## Feature Insights

### Overall Top Features
- **peer_mean_ret_10d** – This feature is expected to provide insights into how similar stocks have performed recently, which can influence predictions.

## Recommendations

- **[monitoring]** Increase monitoring of directional accuracy metrics.
  - Reason: Given the low directional accuracy, it is crucial to understand the factors contributing to this issue.
- **[model_family]** Consider exploring different model families that may better capture market dynamics during periods of volatility.
  - Reason: The current model may not be adequately addressing the complexities of stock movements, especially in shock regimes.
- **[hpo]** Increase the hyperparameter optimization (HPO) budget for the model.
  - Reason: Enhancing the model's tuning could lead to improvements in predictive performance.
- **[data_quality]** Review the data quality and feature engineering processes.
  - Reason: Ensuring high-quality data and relevant features is essential for improving model accuracy.
