import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class PolicyOptimizer:
    """
    Analyzes model performance and generates a dynamic model selection policy.
    """
    
    def __init__(self, default_policy: Dict[str, Any] = None):
        self.default_policy = default_policy or {}

    def optimize_policy(self, performance_summary: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates a model family policy based on performance metrics.
        
        Args:
            performance_summary: DataFrame with columns ['symbol', 'model_family', 'mape', 'smape', 'directional_accuracy']
            
        Returns:
            A dictionary matching the structure of model_families.yaml
        """
        if performance_summary.empty:
            logger.warning("Performance summary is empty. Returning default policy.")
            return self.default_policy
            
        # Calculate composite score
        # Lower score is better
        # Score = w1 * MAPE + w2 * SMAPE + w3 * (1 - Directional_Accuracy)
        
        w_mape = 0.4
        w_smape = 0.3
        w_da = 0.3
        
        # Ensure columns exist
        for col in ['mape', 'smape', 'directional_accuracy']:
            if col not in performance_summary.columns:
                performance_summary[col] = 0.0 if col != 'directional_accuracy' else 0.5
                
        # Group by family
        grouped = performance_summary.groupby('model_family').agg({
            'mape': 'mean',
            'smape': 'mean',
            'directional_accuracy': 'mean'
        })
        
        grouped['score'] = (
            w_mape * grouped['mape'] + 
            w_smape * grouped['smape'] + 
            w_da * (1 - grouped['directional_accuracy'])
        )
        
        family_perf = grouped['score'].sort_values()
        
        logger.info(f"Model family performance ranking (Score): {family_perf.to_dict()}")
        
        ranked_families = family_perf.index.tolist()
        
        # Filter out families that are not suitable for certain roles if needed
        # For now, we assume any trained model can be Primary/Secondary
        
        # 2. Construct Policy
        # We will apply this ranking primarily to the Medium horizon bucket.
        # We will also use it to inform Short and Long, but maybe with some heuristics.
        
        # Heuristic: 
        # - Top 1 -> Primary
        # - Top 2 -> Secondary
        # - Baseline -> Keep as BaselineLinear or AutoDLinear if available
        
        primary = [ranked_families[0]]
        secondary = [ranked_families[1]] if len(ranked_families) > 1 else []
        
        # Special handling for graph_stgcnn
        # If graph_stgcnn is in the top 3, ensure it's included in secondary for medium/long
        if 'graph_stgcnn' in ranked_families[:3] and 'graph_stgcnn' not in primary and 'graph_stgcnn' not in secondary:
             secondary.append('graph_stgcnn')
        
        # Identify baseline
        baseline = ["BaselineLinear"]
        if "AutoDLinear" in ranked_families:
             baseline = ["AutoDLinear"]
        elif "BaselineLinear" in ranked_families:
             baseline = ["BaselineLinear"]
             
        # Construct the new policy structure
        new_policy = {
            "default_policy": {
                "short_horizon": {
                    "primary": primary,
                    "secondary": secondary,
                    "baseline": baseline,
                    "ensemble_weights": {"primary": 0.7, "secondary": 0.3}
                },
                "medium_horizon": {
                    "primary": primary,
                    "secondary": secondary,
                    "baseline": baseline,
                    "ensemble_weights": {"primary": 0.5, "secondary": 0.3, "graph_stgcnn": 0.2 if 'graph_stgcnn' in secondary else 0.0}
                },
                "long_horizon": {
                    "primary": primary,
                    "secondary": secondary,
                    "baseline": baseline,
                    "ensemble_weights": {"primary": 0.6, "secondary": 0.2, "graph_stgcnn": 0.2 if 'graph_stgcnn' in secondary else 0.0}
                }
            },
            "enable_ensembling": True,
            "risk_baseline_family": baseline[0],
            "graph_model_settings": self.default_policy.get("graph_model_settings", {"enabled": True})
        }
        
        # Preserve buckets from default policy if they exist, as we don't have per-bucket metrics yet
        if "buckets" in self.default_policy:
            new_policy["buckets"] = self.default_policy["buckets"]
            
        logger.info(f"Generated optimized policy: Primary={primary}, Secondary={secondary}")
        
        return new_policy
