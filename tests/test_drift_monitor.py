"""
Test Suite for Drift Monitor Agent

Verifies that the DriftMonitorAgent behaves deterministically and handles errors gracefully.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.drift_monitor_agent import DriftMonitorAgent

class TestDriftMonitor:
    
    def setup_method(self):
        # Mock dependencies
        self.mock_gpu = MagicMock()
        self.mock_data = MagicMock()
        self.mock_registry = MagicMock()
        self.mock_inference = MagicMock()
        
        self.agent = DriftMonitorAgent(
            gpu_services=self.mock_gpu,
            data_pipeline=self.mock_data,
            model_registry=self.mock_registry,
            inference_service=self.mock_inference
        )
        
    def test_drift_calculation_deterministic(self):
        """
        Ensure drift calculation is deterministic given fixed inputs.
        """
        # Mock internal methods to return fixed values
        with patch.object(self.agent, '_check_performance_drift') as mock_perf, \
             patch.object(self.agent, '_check_data_drift') as mock_data, \
             patch.object(self.agent, '_check_spectral_drift') as mock_spec:
                 
            mock_perf.return_value = {'drift_detected': True, 'drift_score': 0.5}
            mock_data.return_value = {'drift_detected': False, 'drift_score': 0.1}
            mock_spec.return_value = {'drift_detected': False, 'drift_score': 0.0}
            
            # Run check twice
            res1 = self.agent.comprehensive_drift_check("AAPL")
            res2 = self.agent.comprehensive_drift_check("AAPL")
            
            # Assert equality
            assert res1['overall_drift_score'] == res2['overall_drift_score']
            assert res1['overall_drift_score'] == (0.5 + 0.1 + 0.0) / 3.0
            
    def test_error_handling_guardrail(self):
        """
        Ensure agent returns safe default on internal error instead of crashing.
        """
        # Force an exception in one of the sub-checks
        with patch.object(self.agent, '_check_performance_drift', side_effect=Exception("Boom")):
            
            result = self.agent.comprehensive_drift_check("AAPL")
            
            # Should not raise exception
            assert 'error' in result
            assert result['error'] == "Boom"
            assert result['overall_drift_score'] == 0.0 # Safe default
            assert result['regime_change'] is False
