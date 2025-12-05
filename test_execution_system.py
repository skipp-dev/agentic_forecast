
import os
import sys
import logging
import pandas as pd
import shutil
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

from src.agents.execution_agent import ExecutionAgent
from src.brokers.paper_broker import PaperBroker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_execution_system():
    logger.info("--- Starting Execution System Test ---")
    
    # 1. Setup Test Environment
    test_state_file = "data/test_paper_portfolio.json"
    if os.path.exists(test_state_file):
        os.remove(test_state_file)
        
    # Mock DatabaseService to return None, forcing file/init fallback
    with patch('src.brokers.paper_broker.DatabaseService') as MockDB:
        mock_db_instance = MockDB.return_value
        mock_db_instance.get_latest_portfolio.return_value = None
        
        # Initialize Broker with $100,000
        broker = PaperBroker(state_file=test_state_file, initial_cash=100000.0)
        logger.info(f"Initial Cash: ${broker.get_cash():,.2f}")
        
        # Initialize Agent
        agent = ExecutionAgent(broker=broker)
        
        # 2. Prepare Mock Data
        # Create dummy DataFrames for AAPL and MSFT
        data = {
            'AAPL': pd.DataFrame({'close': [150.0]}, index=[datetime.now()]),
            'MSFT': pd.DataFrame({'close': [300.0]}, index=[datetime.now()])
        }
        
        # 3. Define Target Portfolio (50% AAPL, 30% MSFT, 20% Cash)
        target_orders = [
            {'symbol': 'AAPL', 'target_weight': 0.5, 'action': 'BUY'},
            {'symbol': 'MSFT', 'target_weight': 0.3, 'action': 'BUY'}
        ]
        
        logger.info(f"Target Orders: {target_orders}")
        
        # 4. Execute Orders
        logger.info("Executing orders...")
        results = agent.execute_orders(target_orders, data)
        
        # 5. Verify Results
        logger.info(f"Execution Results: {len(results)} trades executed")
        for res in results:
            logger.info(f"  {res}")
            
        # Check Portfolio State
        cash = broker.get_cash()
        positions = broker.get_positions()
        
        logger.info(f"Final Cash: ${cash:,.2f}")
        logger.info(f"Final Positions: {positions}")
        
        # Assertions
        # Expected AAPL: $50,000 / $150 = 333 shares
        # Expected MSFT: $30,000 / $300 = 100 shares
        # Expected Cost: (333 * 150) + (100 * 300) = 49950 + 30000 = 79950
        # Expected Cash: 100000 - 79950 = 20050 (approx, ignoring fees/slippage)
        
        assert 'AAPL' in positions, "AAPL should be in positions"
        assert 'MSFT' in positions, "MSFT should be in positions"
        
        assert positions['AAPL'] == 333, f"Expected 333 AAPL, got {positions['AAPL']}"
        assert positions['MSFT'] == 100, f"Expected 100 MSFT, got {positions['MSFT']}"
        
        # Check Persistence
        if os.path.exists(test_state_file):
            logger.info("✅ State file created successfully")
        else:
            logger.error("❌ State file NOT created")
            
        logger.info("✅ Test Passed!")

    # Cleanup
    if os.path.exists(test_state_file):
        os.remove(test_state_file)
    if os.path.exists(test_state_file + ".lock"):
        os.remove(test_state_file + ".lock")

if __name__ == "__main__":
    try:
        test_execution_system()
    except Exception as e:
        logger.error(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
