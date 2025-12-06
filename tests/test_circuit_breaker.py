import unittest
import os
import sys
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.risk.circuit_breaker import CircuitBreaker

class TestCircuitBreaker(unittest.TestCase):
    def test_drawdown_trip(self):
        # Disable daily loss check for this test
        cb = CircuitBreaker(max_drawdown_pct=0.10, max_daily_loss_pct=1.0)
        cb.update_equity(100000.0) # Peak
        
        # Drop to 95k (5% DD) - Safe
        self.assertTrue(cb.check_risk(95000.0, 0))
        
        # Drop to 89k (11% DD) - Trip
        self.assertFalse(cb.check_risk(89000.0, 0))
        self.assertTrue(cb.is_tripped)
        self.assertIn("Max Drawdown", cb.trip_reason)
        
    def test_daily_loss_trip(self):
        # Disable drawdown check for this test
        cb = CircuitBreaker(max_daily_loss_pct=0.05, max_drawdown_pct=1.0)
        cb.update_equity(100000.0, is_new_day=True)
        
        # Drop to 96k (4% Loss) - Safe
        self.assertTrue(cb.check_risk(96000.0, 0))
        
        # Drop to 94k (6% Loss) - Trip
        self.assertFalse(cb.check_risk(94000.0, 0))
        self.assertTrue(cb.is_tripped)
        self.assertIn("Max Daily Loss", cb.trip_reason)

if __name__ == '__main__':
    unittest.main()
