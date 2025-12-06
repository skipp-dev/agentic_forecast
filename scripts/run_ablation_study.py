"""
Ablation Study Runner

Runs a comparative study to validate the impact of specific features (e.g., Spectral Features).
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.golden_dataset_manager import GoldenDatasetManager
from src.features.feature_engineer import FeatureEngineer
from src.data_pipeline import DataPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AblationStudy:
    
    def __init__(self):
        self.manager = GoldenDatasetManager()
        
    def run_spectral_ablation(self, symbol: str) -> Dict[str, float]:
        """
        Compare Baseline vs Baseline + Spectral features.
        """
        logger.info(f"Running Spectral Ablation for {symbol}")
        
        # Load Data
        df = self.manager.load_data(symbol)
        if df.empty:
            logger.error(f"No golden data for {symbol}")
            return {}
            
        # 1. Prepare Baseline Features
        df_baseline = FeatureEngineer.add_technical_features(df)
        df_baseline = df_baseline.dropna()
        
        # 2. Prepare Spectral Features
        df_spectral = FeatureEngineer.add_spectral_features(df_baseline)
        df_spectral = df_spectral.dropna()
        
        # Align indices (spectral drops more data due to window)
        common_index = df_spectral.index
        df_baseline = df_baseline.loc[common_index]
        
        # Define Target (Next Day Return)
        target = df_baseline['returns'].shift(-1).fillna(0)
        
        # Define Feature Sets
        baseline_cols = ['volatility', 'momentum_1m', 'momentum_3m', 'volume_ma']
        spectral_cols = baseline_cols + ['spectral_dominant_freq', 'spectral_entropy']
        
        # Train/Test Split (Simple time-based)
        split_idx = int(len(df_baseline) * 0.8)
        
        # Baseline Model
        mae_baseline = self._train_eval(df_baseline, baseline_cols, target, split_idx)
        
        # Spectral Model
        mae_spectral = self._train_eval(df_spectral, spectral_cols, target, split_idx)
        
        improvement = (mae_baseline - mae_spectral) / mae_baseline * 100
        
        logger.info(f"Results for {symbol}:")
        logger.info(f"  Baseline MAE: {mae_baseline:.6f}")
        logger.info(f"  Spectral MAE: {mae_spectral:.6f}")
        logger.info(f"  Improvement:  {improvement:.2f}%")
        
        return {
            "baseline_mae": mae_baseline,
            "spectral_mae": mae_spectral,
            "improvement_pct": improvement
        }
        
    def _train_eval(self, df: pd.DataFrame, features: List[str], target: pd.Series, split_idx: int) -> float:
        """
        Train a simple linear model and return MAE.
        """
        X = df[features].values
        y = target.loc[df.index].values
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        return mae

def main():
    study = AblationStudy()
    
    # Check if golden dataset exists
    if not study.manager.metadata_file.exists():
        logger.error("Golden Dataset not found. Run scripts/generate_golden_dataset.py first.")
        return

    # Load symbols from metadata
    import json
    with open(study.manager.metadata_file, 'r') as f:
        meta = json.load(f)
    symbols = meta.get("symbols", [])
    
    results = {}
    for symbol in symbols:
        res = study.run_spectral_ablation(symbol)
        if res:
            results[symbol] = res
            
    # Summary
    print("\n=== Ablation Study Summary ===")
    avg_improvement = np.mean([r['improvement_pct'] for r in results.values()])
    print(f"Average Improvement with Spectral Features: {avg_improvement:.2f}%")
    
    if avg_improvement > 1.0:
        print("CONCLUSION: Spectral features show promise ( > 1% improvement).")
    else:
        print("CONCLUSION: Spectral features do not show significant improvement.")

if __name__ == "__main__":
    main()
