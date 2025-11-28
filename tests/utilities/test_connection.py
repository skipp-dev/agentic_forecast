#!/usr/bin/env python3

from src.data.unified_ingestion_v2 import UnifiedDataIngestion

print("Testing data ingestion initialization...")

try:
    ui = UnifiedDataIngestion()
    print(f"IB available: {ui.ib_available}")
    print(f"AV available: {ui.av_available}")
    print(f"Primary source: {ui.primary_source}")
    print("✅ Initialization successful")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()