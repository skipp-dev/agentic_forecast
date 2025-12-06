import unittest
from src.brokers.transaction_costs import VolatilityAdjustedSlippageModel, LinearSlippageModel
from src.brokers.paper_broker import PaperBroker
from src.interfaces.broker_interface import Order

class TestTransactionCosts(unittest.TestCase):
    def test_volatility_adjusted_slippage(self):
        # Base spread 2bps, VolFactor 0.05, ImpactFactor 0.01
        model = VolatilityAdjustedSlippageModel(base_spread_bps=2.0, vol_factor=0.05, impact_factor=0.01)
        
        # Case 1: Low Volatility (1%), Small Order (1% of Volume)
        # Vol Component: 0.01 * 10000 * 0.05 = 5 bps
        # Impact Component: sqrt(0.01) * 10000 * 0.01 = 0.1 * 100 = 10 bps
        # Total: 2 + 5 + 10 = 17 bps
        cost = model.get_slippage_cost(order_value=10000, quantity=100, price=100, volatility=0.01, avg_volume=10000)
        expected_bps = 2.0 + (0.01 * 10000 * 0.05) + (0.01 * (100/10000)**0.5 * 10000)
        # Wait, participation is 100/10000 = 0.01. sqrt(0.01) = 0.1.
        # Impact = 0.01 * 0.1 * 10000 = 10 bps.
        # Total = 2 + 5 + 10 = 17 bps.
        # Cost = 10000 * 17/10000 = 17.0
        
        self.assertAlmostEqual(cost, 17.0, places=2)

        # Case 2: High Volatility (5%), Large Order (10% of Volume)
        # Vol Component: 0.05 * 10000 * 0.05 = 25 bps
        # Impact Component: sqrt(0.1) * 10000 * 0.01 = 0.3162 * 100 = 31.62 bps
        # Total: 2 + 25 + 31.62 = 58.62 bps
        cost_high = model.get_slippage_cost(order_value=10000, quantity=1000, price=10, volatility=0.05, avg_volume=10000)
        # Participation = 1000/10000 = 0.1
        
        expected_bps_high = 2.0 + (0.05 * 10000 * 0.05) + (0.01 * (0.1)**0.5 * 10000)
        expected_cost_high = 10000 * expected_bps_high / 10000
        
        self.assertAlmostEqual(cost_high, expected_cost_high, places=2)
        self.assertTrue(cost_high > cost)

    def test_paper_broker_integration(self):
        import os
        import uuid
        
        # Use unique paths to ensure clean state
        unique_id = str(uuid.uuid4())
        db_path = f"tests/test_db_{unique_id}.sqlite"
        state_file = f"tests/test_portfolio_{unique_id}.json"
        
        try:
            model = VolatilityAdjustedSlippageModel()
            broker = PaperBroker(state_file=state_file, initial_cash=100000, cost_model=model, db_path=db_path)
            
            # Place order with kwargs
            order = Order(symbol="TEST", action="BUY", quantity=100, price=100)
            fill = broker.place_order(order, volatility=0.02, avg_volume=100000)
            
            self.assertEqual(fill.status, "FILLED")
            # Check if cash was deducted correctly (Price + Slippage + Comm)
            # Slippage for 2% vol, 0.1% participation
            # Vol: 0.02 * 10000 * 0.05 = 10 bps
            # Impact: sqrt(0.001) * 10000 * 0.01 = 0.0316 * 100 = 3.16 bps
            # Base: 2 bps
            # Total: 15.16 bps
            # Cost: 10000 * 15.16/10000 = 15.16
            # Comm: 100 * 0.005 = 0.5 (min 1.0) -> 1.0
            # Total Cost: 10000 + 15.16 + 1.0 = 10016.16
            
            expected_cash = 100000 - (10000 + 15.16 + 1.0)
            # Allow some float error
            self.assertTrue(abs(broker.get_cash() - expected_cash) < 1.0)
            
        finally:
            # Cleanup
            try:
                # Close DB connection if possible (not exposed in PaperBroker, so we might fail)
                if hasattr(broker, 'db_service') and hasattr(broker.db_service, 'engine'):
                    broker.db_service.engine.dispose()
            except:
                pass

            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
            except Exception as e:
                print(f"Warning: Could not remove temp DB: {e}")

            try:
                if os.path.exists(state_file):
                    os.remove(state_file)
                if os.path.exists(state_file + ".lock"):
                    os.remove(state_file + ".lock")
            except:
                pass

if __name__ == '__main__':
    unittest.main()
