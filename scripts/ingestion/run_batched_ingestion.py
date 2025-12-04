#!/usr/bin/env python3
"""Batched ingestion runner
Processes symbols from `watchlist_main.csv` in configurable batches and
saves processed data to `data/processed/batched/`. Writes simple ingestion
metrics to SQLite MetricsDatabase.
"""
import os
import sys
import time
import pandas as pd
from datetime import datetime
sys.path.append(os.path.dirname(__file__))

from src.data.unified_ingestion_v2 import UnifiedDataIngestion
from data.metrics_database import MetricsDatabase

WATCHLIST = 'watchlist_main.csv'
OUTPUT_DIR = 'data/processed/batched'

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '50'))
BATCH_DELAY = float(os.environ.get('BATCH_DELAY', '5'))  # seconds between batches


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_watchlist(limit=None):
    if not os.path.exists(WATCHLIST):
        print(f"Watchlist not found: {WATCHLIST}")
        return []
    df = pd.read_csv(WATCHLIST)
    if 'Symbol' not in df.columns:
        print(f"Watchlist missing 'Symbol' column")
        return []
    symbols = df['Symbol'].tolist()
    if limit:
        symbols = symbols[:limit]
    return symbols


def run_batches(symbols):
    ingestion = UnifiedDataIngestion(use_real_data=True, config={})
    metrics_db = MetricsDatabase(db_path='data/metrics/metrics.db')
    print("Initializing ingestion...")
    ingestion.initialize()
    print(f"Primary source: {ingestion.primary_source}")

    total = len(symbols)
    for i in range(0, total, BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}: {len(batch)} symbols")
        for symbol in batch:
            try:
                print(f"  -> Fetching {symbol}")
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
                df = ingestion.get_historical_data(symbol, start_date, end_date, timeframe='1 day')
                success = False
                rows = 0
                source = ingestion.primary_source
                if df is not None and not df.empty:
                    rows = len(df)
                    out_file = os.path.join(OUTPUT_DIR, f"{symbol}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    df.to_csv(out_file)
                    success = True
                    print(f"    ✅ Saved {rows} rows to {out_file}")
                else:
                    print(f"    ⚠️  No data for {symbol}")

                # Store metrics in SQLite database
                metrics_db.store_metric(
                    metric_name='ingestion.success_rate',
                    value=1 if success else 0,
                    tags={'symbol': symbol, 'source': source, 'batch': str(i//BATCH_SIZE + 1)},
                    metadata={'rows': rows, 'file_path': out_file if success else None}
                )

                if success:
                    metrics_db.store_metric(
                        metric_name='ingestion.rows_processed',
                        value=rows,
                        tags={'symbol': symbol, 'source': source},
                        metadata={'file_path': out_file}
                    )

            except Exception as e:
                print(f"    ❌ Error fetching {symbol}: {e}")
                metrics_db.store_metric(
                    metric_name='ingestion.errors',
                    value=1,
                    tags={'symbol': symbol, 'error': str(e)[:100]},
                    metadata={'error_details': str(e)}
                )
        print(f"Batch complete. Sleeping {BATCH_DELAY}s before next batch...")
        time.sleep(BATCH_DELAY)

    ingestion.disconnect()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run batched ingestion')
    parser.add_argument('--limit', type=int, default=100, help='Limit number of symbols to process')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Symbols per batch')
    parser.add_argument('--batch-delay', type=float, default=BATCH_DELAY, help='Seconds between batches')
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    BATCH_DELAY = args.batch_delay

    ensure_dirs()
    syms = load_watchlist(limit=args.limit)
    if not syms:
        print('No symbols to process')
        sys.exit(1)
    run_batches(syms)