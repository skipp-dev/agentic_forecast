import json
import os
import time
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Manages model metadata, including champion models and HPO run timestamps.
    """
    def __init__(self, registry_path: str = "data/model_registry.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                return {}
        return {}

    def _save_registry(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def get_symbol_metadata(self, symbol: str) -> Dict[str, Any]:
        return self.registry.get(symbol, {})

    def update_symbol_metadata(self, symbol: str, metadata: Dict[str, Any]):
        if symbol not in self.registry:
            self.registry[symbol] = {}
        self.registry[symbol].update(metadata)
        self._save_registry()

    def get_last_hpo_run(self, symbol: str) -> Optional[float]:
        return self.registry.get(symbol, {}).get("last_hpo_run_ts")

    def set_last_hpo_run(self, symbol: str, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        self.update_symbol_metadata(symbol, {"last_hpo_run_ts": timestamp})

    def get_champion_model(self, symbol: str) -> Optional[str]:
        return self.registry.get(symbol, {}).get("champion_model_family")

    def set_champion_model(self, symbol: str, model_family: str, reason: Optional[str] = None):
        metadata = {"champion_model_family": model_family}
        if reason:
            metadata["champion_selection_reason"] = reason
        self.update_symbol_metadata(symbol, metadata)
