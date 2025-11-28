#!/usr/bin/env python3
"""
Quick Component Verification Script

Tests basic functionality of Phase 2 components without full execution.
"""

import os
import sys
import yaml
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all Phase 2 components can be imported"""
    print("Testing component imports...")

    try:
        from agents.macro_data_agent import MacroDataAgent
        print("✅ MacroDataAgent imported successfully")
    except ImportError as e:
        print(f"❌ MacroDataAgent import failed: {e}")
        return False

    try:
        from agents.regime_detection_agent import RegimeDetectionAgent
        print("✅ RegimeDetectionAgent imported successfully")
    except ImportError as e:
        print(f"❌ RegimeDetectionAgent import failed: {e}")
        return False

    try:
        from agents.strategy_selection_agent import StrategySelectionAgent
        print("✅ StrategySelectionAgent imported successfully")
    except ImportError as e:
        print(f"❌ StrategySelectionAgent import failed: {e}")
        return False

    return True

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")

    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Check Phase 2 configurations
        if 'macro' in config:
            print("✅ Macro configuration found")
            fred_key = config['macro'].get('fred_api_key', '')
            if fred_key and fred_key != 'your_fred_api_key_here':
                print("✅ FRED API key configured")
            else:
                print("⚠️  FRED API key not configured (using placeholder)")
        else:
            print("❌ Macro configuration missing")

        if 'scaling' in config:
            print("✅ Scaling configuration found")
            max_symbols = config['scaling'].get('max_symbols', 50)
            print(f"   Max symbols: {max_symbols}")
        else:
            print("❌ Scaling configuration missing")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_access():
    """Test basic data access"""
    print("\nTesting data access...")

    try:
        # Check if watchlist exists
        if os.path.exists('watchlist_ibkr.csv'):
            import pandas as pd
            df = pd.read_csv('watchlist_ibkr.csv')
            symbol_count = len(df)
            print(f"✅ Watchlist loaded: {symbol_count} symbols")
        else:
            print("❌ Watchlist file not found")
            return False

        # Check data directories
        data_dirs = ['data/raw', 'data/features', 'data/models', 'logs']
        for dir_path in data_dirs:
            if os.path.exists(dir_path):
                print(f"✅ Directory exists: {dir_path}")
            else:
                print(f"⚠️  Directory missing: {dir_path}")

        return True

    except Exception as e:
        print(f"❌ Data access test failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("Phase 2 Component Verification")
    print("=" * 40)

    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_config),
        ("Data Access Tests", test_data_access),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        result = test_func()
        results.append(result)

    print("\n" + "=" * 40)
    print("VERIFICATION SUMMARY")
    print("=" * 40)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ All tests passed ({passed}/{total})")
        print("\n🚀 System ready for Phase 2 operations!")
        print("\nNext steps:")
        print("1. Configure FRED API key in config.yaml")
        print("2. Run: python scripts/run_phase2_pipeline.py")
        print("3. Setup automation: powershell scripts/setup_task_scheduler.ps1")
    else:
        print(f"⚠️  Some tests failed ({passed}/{total})")
        print("Please resolve the issues above before proceeding.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)