import unittest
import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.database_service import DatabaseService

class TestDatabaseService(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_forecast.db"
        self.service = DatabaseService(db_path=self.test_db)
        
    def tearDown(self):
        # Dispose engine to release lock
        if hasattr(self.service, 'engine'):
            self.service.engine.dispose()
            
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except PermissionError:
                pass # Windows file lock issues
            
    def test_market_data_save_load(self):
        df = pd.DataFrame({
            'open': [100.0], 'high': [105.0], 'low': [95.0], 'close': [102.0],
            'volume': [1000], 'adjusted_close': [102.0],
            'sma': [100.0], 'ema': [100.0], 'rsi': [50.0], 'macd': [0.0]
        }, index=pd.to_datetime(['2023-01-01']))
        
        self.service.save_market_data('TEST', df)
        
        loaded = self.service.get_market_data('TEST')
        self.assertFalse(loaded.empty)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.iloc[0]['close'], 102.0)
        
    def test_portfolio_save_load(self):
        self.service.save_portfolio_state(10000.0, {'AAPL': 10}, 11500.0, is_backtest=True)
        
        port = self.service.get_latest_portfolio(is_backtest=True)
        self.assertIsNotNone(port)
        self.assertEqual(port['cash'], 10000.0)
        self.assertEqual(port['positions']['AAPL'], 10)
        
    def test_clear_backtest(self):
        self.service.save_portfolio_state(10000.0, {'AAPL': 10}, 11500.0, is_backtest=True)
        self.service.clear_backtest_data()
        
        port = self.service.get_latest_portfolio(is_backtest=True)
        self.assertIsNone(port)

if __name__ == '__main__':
    unittest.main()
