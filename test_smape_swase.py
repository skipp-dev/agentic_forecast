#!/usr/bin/env python3
"""
Test the SMAPE and SWASE evaluation function.
"""

from analytics.evaluation_metrics import evaluate_forecast_series
import numpy as np

# Generate sample data
np.random.seed(42)
actual = np.random.randn(100) + 100  # Stock prices around 100
pred = actual + np.random.randn(100) * 2  # Predictions with some error

# Sample regime flags (10% shock days)
regime_flags = {
    "peer_shock_flag": np.random.choice([0, 1], size=100, p=[0.9, 0.1])
}

# Evaluate
results = evaluate_forecast_series(actual, pred, regime_flags)

print("SMAPE & SWASE Evaluation Test Results:")
print("=" * 40)
for k, v in results.items():
    print("20")

print("\nNote: SWASE currently uses equal weights (placeholder)")
print("In production, shock days would be weighted more heavily.")