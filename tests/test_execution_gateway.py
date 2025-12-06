import unittest
from unittest.mock import MagicMock, patch
from src.brokers.execution_gateway import ExecutionGateway
from src.interfaces.broker_interface import Order, Fill
from datetime import datetime
import asyncio

class TestExecutionGateway(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {"mode": "paper", "max_retries": 2, "retry_delay": 0.01} # Fast retry for tests
        self.gateway = ExecutionGateway(self.config)
        # Mock the internal broker
        self.gateway.broker = MagicMock()

    def test_initialization(self):
        gateway = ExecutionGateway({"mode": "paper"})
        from src.brokers.paper_broker import PaperBroker
        self.assertIsInstance(gateway.broker, PaperBroker)

    async def test_place_order_success(self):
        order = Order("AAPL", "BUY", 10, "MARKET", 150.0)
        expected_fill = Fill("1", "AAPL", "BUY", 10, 150.0, datetime.now(), 0.0, "FILLED")
        
        self.gateway.broker.place_order.return_value = expected_fill
        
        fill = await self.gateway.place_order(order)
        self.assertEqual(fill.status, "FILLED")
        self.gateway.broker.place_order.assert_called_once()

    async def test_place_order_retry(self):
        order = Order("AAPL", "BUY", 10, "MARKET", 150.0)
        expected_fill = Fill("1", "AAPL", "BUY", 10, 150.0, datetime.now(), 0.0, "FILLED")
        
        # Fail first time, succeed second time
        self.gateway.broker.place_order.side_effect = [Exception("Network Error"), expected_fill]
        
        fill = await self.gateway.place_order(order)
        self.assertEqual(fill.status, "FILLED")
        self.assertEqual(self.gateway.broker.place_order.call_count, 2)

    async def test_place_order_max_retries_exceeded(self):
        order = Order("AAPL", "BUY", 10, "MARKET", 150.0)
        
        # Always fail
        self.gateway.broker.place_order.side_effect = Exception("Network Error")
        
        fill = await self.gateway.place_order(order)
        self.assertEqual(fill.status, "FAILED_ERROR")
        self.assertEqual(self.gateway.broker.place_order.call_count, 2) # max_retries=2

if __name__ == '__main__':
    unittest.main()
