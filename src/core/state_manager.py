"""
State Manager

Handles persistence of large artifacts (DataFrames) to disk to keep LangGraph state lightweight.
"""

import os
import pandas as pd
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages storage and retrieval of large state artifacts.
    """
    
    def __init__(self, run_id: str, base_dir: str = "data/runs"):
        self.run_id = run_id
        self.run_dir = Path(base_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def save_dataframe(self, df: pd.DataFrame, name: str, subdir: str = "data") -> str:
        """
        Save a DataFrame to parquet and return the path.
        """
        if df.empty:
            logger.warning(f"Attempting to save empty DataFrame for {name}")
            return ""
            
        save_dir = self.run_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = save_dir / f"{name}.parquet"
        try:
            df.to_parquet(file_path)
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save DataFrame {name}: {e}")
            raise

    def load_dataframe(self, path: str) -> pd.DataFrame:
        """
        Load a DataFrame from a path.
        """
        if not path or not os.path.exists(path):
            logger.warning(f"Path not found: {path}")
            return pd.DataFrame()
            
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.error(f"Failed to load DataFrame from {path}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_artifact_path(state: Dict[str, Any], category: str, key: str) -> Optional[str]:
        """
        Helper to safely get a path from state.
        e.g. category='data', key='AAPL'
        """
        artifacts = state.get(category, {})
        return artifacts.get(key)
