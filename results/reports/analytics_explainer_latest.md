# LLM Analytics Summary – 2025-11-28 (DAILY)

## Global Summary

```json
{
  "global_summary": "The overall health of the forecasting system appears stable, with some variability in performance across different symbols and horizons. However, certain metrics indicate potential areas for improvement, particularly in specific regimes. The model family performance shows discrepancies that warrant further investigation. Anomalies in directional accuracy suggest that some models struggle with predicting the correct trend, which could impact decision-making. Overall, while the system is functioning, targeted adjustments could enhance predictive capabilities.",
  "metric_explanations": {
    "mae": "MAE values indicate a generally acceptable level of error, but some symbols exhibit significantly higher values, suggesting that certain forecasts are less reliable. Monitoring these outliers is essential to understand their impact on overall performance.",
    "mape": "MAPE values show a wide range, with some symbols experiencing extreme percentages that may indicate issues with data quality or model fit. Outliers should be scrutinized to determine if they are due to genuine variability or model deficiencies.",
    "smape": "SMAPE values diverge from MAPE in several instances, particularly in symbols with high volatility. This discrepancy suggests that the models may be struggling to capture the underlying patterns accurately, especially in shock regimes.",
    "mase": "MASE indicates that some models are performing worse than naive benchmarks, particularly for longer horizons. This suggests a need for reevaluation of feature sets or model complexity to improve performance.",
    "swase": "SWASE results indicate potential bias in certain regimes, particularly during periods of market shock. This could lead to asymmetric error distributions, which are critical to address for risk management.",
    "directional_accuracy": "Directional accuracy shows that while the models are correct in trend prediction most of the time, there are specific symbols where the accuracy drops significantly. This inconsistency could lead to poor strategic decisions."
  },
  "regime_insights": [
    {
      "regime": "shock regime",
      "performance_comment": "Performance tends to degrade in shock regimes, with higher errors and lower directional accuracy.",
      "risk_comment": "Deploying models in this regime is risky due to the potential for significant financial impact from incorrect predictions."
    }
  ],
  "symbol_outliers": [
    {
      "symbol": "XYZ",
      "horizon": 1,
      "issue": "Very high MAPE and MAE values.",
      "comment": "The extreme values suggest potential data quality issues or model misalignment with the underlying data patterns."
    }
  ],
  "feature_insights": {
    "overall_top_features": [
      {
        "name": "feature_A",
        "importance_comment": "Feature_A consistently shows high importance across multiple symbols, indicating it captures essential trends."
      }
    ],
    "shock_regime_top_features": [
      {
        "name": "feature_B",
        "importance_comment": "Feature_B is particularly relevant in shock regimes, as it appears to correlate strongly with volatility."
      }
    ]
  },
  "recommendations": [
    {
      "category": "HPO",
      "action": "Run HPO for symbols with high MAPE and MAE on 1d horizon.",
      "reason": "Targeting these symbols could help reduce error rates and improve overall model performance."
    },
    {
      "category": "MODEL_SWITCH",
      "action": "Consider switching models for XYZ due to poor performance metrics.",
      "reason": "The current model does not seem to capture the dynamics of XYZ effectively."
    },
    {
      "category": "FEATURES",
      "action": "Evaluate the inclusion of additional features that may capture market shocks better.",
      "reason": "Improving feature sets could enhance model robustness in volatile conditions."
    }
  ]
}
```
