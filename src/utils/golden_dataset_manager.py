"""
Golden Dataset Manager

Utilities to create, load, and validate the 'Golden Dataset' for regression testing.
"""

import os
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

class GoldenDatasetManager:
    """
    Manages the lifecycle of the Golden Dataset.
    """
    
    def __init__(self, base_dir: str = "tests/golden_dataset"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_dir / "metadata.json"
        
    def create_dataset(self, symbols: List[str], start_date: str, end_date: str, 
                       pipeline: Optional[DataPipeline] = None):
        """
        Fetch data and save it as the Golden Dataset.
        WARNING: This overwrites the existing golden dataset.
        """
        if pipeline is None:
            pipeline = DataPipeline()
            
        logger.info(f"Creating Golden Dataset for {symbols} from {start_date} to {end_date}")
        
        # Calculate period roughly
        # This is a simplification, ideally we pass exact dates to fetcher if supported
        # or filter after fetching
        
        for symbol in symbols:
            try:
                # Fetch max to ensure we cover the range, then slice
                df = pipeline.fetch_stock_data(symbol, period='5y', interval='daily')
                
                if df.empty:
                    logger.error(f"Failed to fetch data for {symbol}")
                    continue
                
                # Slice to exact range
                mask = (df.index >= start_date) & (df.index <= end_date)
                df_slice = df.loc[mask]
                
                if df_slice.empty:
                    logger.warning(f"No data found for {symbol} in range {start_date}-{end_date}")
                    continue
                
                # Save to parquet
                save_path = self.base_dir / f"{symbol}.parquet"
                df_slice.to_parquet(save_path)
                logger.info(f"Saved {symbol} to {save_path} ({len(df_slice)} rows)")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Save metadata
        import json
        metadata = {
            "created_at": datetime.now().isoformat(),
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "version": "1.0"
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        Load data for a symbol from the Golden Dataset.
        """
        path = self.base_dir / f"{symbol}.parquet"
        if not path.exists():
            logger.error(f"Golden data not found for {symbol} at {path}")
            return pd.DataFrame()
            
        return pd.read_parquet(path)

    def validate_current_performance(self, current_metrics: Dict[str, float], tolerance: float = 0.05) -> bool:
        """
        Compare current run metrics against stored golden metrics.
        """
        # Load expected metrics (if they exist)
        metrics_path = self.base_dir / "expected_metrics.json"
        if not metrics_path.exists():
            logger.warning("No expected metrics found. Saving current as baseline.")
            import json
            with open(metrics_path, 'w') as f:
                json.dump(current_metrics, f, indent=2)
            return True
            
        import json
        with open(metrics_path, 'r') as f:
            expected = json.load(f)
            
        passed = True
        for key, val in current_metrics.items():
            if key in expected:
                target = expected[key]
                # Check if within tolerance (relative)
                if abs(val - target) / (abs(target) + 1e-9) > tolerance:
                    logger.error(f"Regression detected in {key}: Current {val:.4f} vs Expected {target:.4f}")
                    passed = False
                else:
                    logger.info(f"Metric {key} passed: {val:.4f} (Target {target:.4f})")
            else:
                logger.warning(f"New metric {key} not in golden baseline.")
                
        return passed
