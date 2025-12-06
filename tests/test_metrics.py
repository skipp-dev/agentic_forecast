import unittest
from unittest.mock import MagicMock, patch
import time
import os
from src.monitoring.metrics import PIPELINE_LATENCY, SYSTEM_ERRORS, PORTFOLIO_VALUE, TRADE_COUNT
from src.agents.orchestrator_agent import OrchestratorAgent
from src.brokers.paper_broker import PaperBroker
from src.interfaces.broker_interface import Order

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Reset metrics for testing
        # Note: Prometheus client doesn't easily allow resetting global registry, 
        # so we check for increments/changes.
        pass

    @patch('src.services.model_registry_service.ModelRegistryService')
    def test_pipeline_latency_decorator(self, mock_registry):
        agent = OrchestratorAgent()
        # Mock internal methods to avoid complex setup
        agent._check_gpu_status = MagicMock(return_value={'available': True})
        agent._analyze_context = MagicMock(return_value={})
        agent._advanced_decision_making = MagicMock(return_value="next_step")
        agent._optimize_gpu_resources = MagicMock()

        # Get initial count
        # Using collect() to inspect metrics safely
        metrics = PIPELINE_LATENCY.collect()
        initial_count = 0
        if metrics:
            for sample in metrics[0].samples:
                if sample.name.endswith('_count'):
                    initial_count = sample.value
                    break
        
        # Run method
        agent.coordinate_workflow({})
        
        # Check if count incremented
        metrics = PIPELINE_LATENCY.collect()
        new_count = 0
        if metrics:
            for sample in metrics[0].samples:
                if sample.name.endswith('_count'):
                    new_count = sample.value
                    break
                    
        self.assertEqual(new_count, initial_count + 1)

    @patch('src.services.model_registry_service.ModelRegistryService')
    def test_system_errors_metric(self, mock_registry):
        agent = OrchestratorAgent()
        agent._check_gpu_status = MagicMock(side_effect=Exception("Test Error"))
        
        initial_errors = SYSTEM_ERRORS.labels(component='orchestrator')._value.get()
        
        # This should trigger the exception handler and increment the counter
        result = agent.coordinate_workflow({})
        
        new_errors = SYSTEM_ERRORS.labels(component='orchestrator')._value.get()
        self.assertEqual(new_errors, initial_errors + 1)
        self.assertEqual(result, "end")

    @patch('src.brokers.paper_broker.DatabaseService')
    def test_broker_metrics(self, mock_db_cls):
        # Mock DB to return None for get_latest_portfolio, forcing fresh init
        mock_db = mock_db_cls.return_value
        mock_db.get_latest_portfolio.return_value = None
        
        # Use a non-existent file to prevent loading from legacy JSON
        broker = PaperBroker(state_file="non_existent_portfolio.json")
        # Correct labels: symbol, action, status
        initial_trades = TRADE_COUNT.labels(symbol='AAPL', action='BUY', status='FILLED')._value.get()
        
        # Mock circuit breaker to allow trade
        broker.circuit_breaker.check_limits = MagicMock(return_value=True)
        broker.circuit_breaker.check_risk = MagicMock(return_value=True) # Also mock check_risk
        
        # Create Order object
        order = Order(
            symbol="AAPL",
            quantity=10,
            action="BUY",
            order_type="market",
            price=150.0
        )
        
        broker.place_order(order)
        
        new_trades = TRADE_COUNT.labels(symbol='AAPL', action='BUY', status='FILLED')._value.get()
        self.assertEqual(new_trades, initial_trades + 1)

if __name__ == '__main__':
    unittest.main()
