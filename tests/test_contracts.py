import unittest
import os
import sys
import pandas as pd
import pandera as pa

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.database_service import DatabaseService
from src.data.contracts import MarketDataSchema

class TestDataContracts(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_contracts.db"
        self.service = DatabaseService(db_path=self.test_db)
        
    def tearDown(self):
        if hasattr(self.service, 'engine'):
            self.service.engine.dispose()
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except:
                pass
            
    def test_valid_data(self):
        df = pd.DataFrame({
            'open': [100.0], 'high': [105.0], 'low': [95.0], 'close': [102.0],
            'volume': [1000], 'adjusted_close': [102.0]
        }, index=pd.to_datetime(['2023-01-01']))
        df.index.name = 'date'
        
        # Should not raise
        self.service.save_market_data('TEST', df)
        
        loaded = self.service.get_market_data('TEST')
        self.assertEqual(len(loaded), 1)
        
    def test_invalid_high_low(self):
        # High < Low
        df = pd.DataFrame({
            'open': [100.0], 'high': [90.0], 'low': [95.0], 'close': [92.0],
            'volume': [1000], 'adjusted_close': [92.0]
        }, index=pd.to_datetime(['2023-01-01']))
        df.index.name = 'date'
        
        # Should log error and NOT save
        self.service.save_market_data('BAD_DATA', df)
        
        loaded = self.service.get_market_data('BAD_DATA')
        self.assertTrue(loaded.empty)

    def test_negative_price(self):
        # Close < 0
        df = pd.DataFrame({
            'open': [100.0], 'high': [105.0], 'low': [95.0], 'close': [-10.0],
            'volume': [1000], 'adjusted_close': [-10.0]
        }, index=pd.to_datetime(['2023-01-01']))
        df.index.name = 'date'
        
        self.service.save_market_data('NEG_PRICE', df)
        
        loaded = self.service.get_market_data('NEG_PRICE')
        self.assertTrue(loaded.empty)

if __name__ == '__main__':
    unittest.main()
