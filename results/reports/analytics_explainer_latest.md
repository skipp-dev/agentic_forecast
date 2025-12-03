# LLM Analytics Summary – 2025-11-28 (DAILY)

## Global Summary

The overall performance metrics indicate a concerning lack of predictive accuracy, particularly highlighted by the extremely low mean directional accuracy of 0.013. While MAE and RMSE show reasonable mean values, the high maximum values suggest potential outliers or specific symbols causing significant errors. The variability in metrics is low, indicating a consistent but poor performance across the board. The absence of feature importance in both overall and shock regimes raises concerns about the model's ability to leverage relevant information effectively. Overall, the system appears to be underperforming and may require significant adjustments.

## Metric Explanations

**MAE** – The mean absolute error (MAE) of 2.18 suggests that, on average, predictions are off by this amount. However, the maximum MAE of 58.658 indicates that some predictions are significantly worse, pointing to potential outliers or specific symbols that need investigation.
**MAPE** – The mean absolute percentage error (MAPE) of 0.039 is relatively low, but the maximum value of 0.446 suggests that certain predictions are highly inaccurate. This discrepancy indicates that while most predictions are reasonably accurate, a few are drastically off.
**SMAPE** – The symmetric mean absolute percentage error (SMAPE) mirrors MAPE's findings, with a mean of 0.038 and a maximum of 0.358. The low variability suggests that the model's performance is consistently poor across predictions.
**MASE** – Not available due to lack of baseline comparison metrics in the provided data.
**SWASE** – The mean SWASE of 0.041 indicates low asymmetric error, but the maximum value of 0.479 suggests that there are instances of significant bias in predictions, particularly in certain regimes.
**DIRECTIONAL_ACCURACY** – The mean directional accuracy of 0.013 indicates that the model is rarely correct in predicting the direction of changes. This is a critical issue, as it suggests that the model's predictions are not reliable for decision-making.

## Regime Insights

### Regime: peer_shock_flag=0
- Performance: In this regime, the performance metrics show a slight increase in MAE and MAPE, indicating that the model's accuracy decreases under these conditions.
- Risk: This regime is risky for deployment due to the observed increase in error metrics, suggesting that the model may not perform well when peer shocks are absent.

## Feature Insights

### Overall Top Features
- **peer_mean_ret_10d** – Despite having zero importance, this feature is expected to provide insights into peer performance, which could be critical if properly utilized.
- **sector_drawdown_60d** – This feature could potentially capture market trends but is currently not contributing to model performance.
- **beta_vs_sector_60d** – This feature might help in understanding volatility relative to the sector but lacks importance in the current model setup.
- **ret_20d_rank_in_sector** – Rankings can be useful for relative performance insights, yet it is not being effectively leveraged in the model.

## Recommendations

- **[FEATURES]** Re-evaluate and engineer features to improve predictive power, especially focusing on peer and sector-related metrics.
  - Reason: Current features show zero importance, indicating a need for better feature selection or engineering to enhance model performance.
- **[HPO]** Conduct hyperparameter optimization to explore different model configurations that may yield better accuracy.
  - Reason: Given the low directional accuracy and high variability in MAE, tuning model parameters could help improve performance.
- **[MODEL_SWITCH]** Consider switching to a different model family that may better capture the underlying patterns in the data.
  - Reason: The current model's performance is unsatisfactory, and exploring alternative algorithms could provide better results.
