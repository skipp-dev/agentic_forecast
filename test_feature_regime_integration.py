
import sys
import os
import pandas as pd
import logging
import numpy as np

# Add root to path
sys.path.append(os.getcwd())

from src.agents.macro_data_agent import MacroDataAgent
from src.agents.regime_detection_agent import RegimeDetectionAgent
from src.agents.feature_agent import FeatureAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_integration():
    # 1. Get Macro Data
    logger.info("1. Fetching Macro Data...")
    macro_agent = MacroDataAgent()
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=150)
    macro_data = macro_agent.get_macro_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if 'processed_features' not in macro_data:
        logger.error("No processed features")
        return

    # 2. Detect Historical Regimes
    logger.info("2. Detecting Historical Regimes...")
    regime_agent = RegimeDetectionAgent()
    historical_regimes = regime_agent.detect_regimes(macro_data['processed_features'])
    
    # Convert to serializable format (as in macro_nodes.py)
    serializable_history = {}
    for name, series in historical_regimes.items():
        series_copy = series.copy()
        series_copy.index = series_copy.index.strftime('%Y-%m-%d')
        serializable_history[name] = series_copy.to_dict()
        
    logger.info(f"Detected regimes: {list(serializable_history.keys())}")

    # 3. Generate Features with Regimes
    logger.info("3. Generating Features...")
    feature_agent = FeatureAgent()
    
    # Mock raw data
    dates = pd.date_range(end=end_date, periods=100)
    raw_data = {
        'AAPL': pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 2000, 100)
        }, index=dates)
    }
    
    features = feature_agent.generate_features(
        raw_data,
        macro_data=macro_data,
        historical_regimes=serializable_history
    )
    
    # 4. Verify Features
    logger.info("4. Verifying Features...")
    df = features['AAPL']
    logger.info(f"Feature columns: {df.columns.tolist()}")
    
    # Check for regime columns (one-hot encoded)
    regime_cols = [c for c in df.columns if 'regime' in c]
    logger.info(f"Regime columns found: {regime_cols}")
    
    if regime_cols:
        logger.info("SUCCESS: Regime features integrated!")
    else:
        logger.error("FAILURE: No regime features found!")

if __name__ == "__main__":
    test_integration()
