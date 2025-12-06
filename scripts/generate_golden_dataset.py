"""
Script to generate the Golden Dataset.
"""

import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.golden_dataset_manager import GoldenDatasetManager

logging.basicConfig(level=logging.INFO)

def main():
    manager = GoldenDatasetManager()
    
    # Define Golden Set
    symbols = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    print(f"Generating Golden Dataset for {len(symbols)} symbols...")
    manager.create_dataset(symbols, start_date, end_date)
    print("Done. Dataset saved to tests/golden_dataset/")

if __name__ == "__main__":
    main()
