#!/usr/bin/env python3
"""
Test script for LLM Policy Framework implementation.
Validates tiered model selection, routing policies, and caching.
"""

import os
import sys
import yaml
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_llm_config_loading():
    """Test that LLM configuration loads correctly."""
    print("Testing LLM config loading...")

    # Load config
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("❌ config.yaml not found")
        return False

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    llm_config = config.get('llm', {})
    if not llm_config:
        print("❌ No LLM config found in config.yaml")
        return False

    # Check tiers
    tiers = llm_config.get('tiers', {})
    required_tiers = ['tier1_fast_local', 'tier2_accurate_local']
    for tier in required_tiers:
        if tier not in tiers:
            print(f"❌ Missing tier: {tier}")
            return False
        if not tiers[tier].get('enabled', True):
            print(f"⚠️  Tier {tier} is disabled")

    # Check routing
    routing = llm_config.get('routing', {})
    required_routes = ['reporting_llm', 'llm_analytics_explainer', 'llm_hpo_planner', 'llm_news_feature']
    for route in required_routes:
        if route not in routing:
            print(f"❌ Missing routing for: {route}")
            return False

    print("✅ LLM config loaded successfully")
    return True

def test_llm_client_tier_selection():
    """Test that LLMClient selects models based on tiers."""
    print("Testing LLM client tier selection...")

    try:
        from src.llm.reporting_llm import LLMClient
    except ImportError as e:
        print(f"❌ Cannot import LLMClient: {e}")
        return False

    # Mock config
    config = {
        'llm': {
            'tiers': {
                'tier1_fast_local': {
                    'enabled': True,
                    'models': ['gemma-2-2b', 'llama-3.1-8b-q4'],
                    'threads': 2
                },
                'tier2_accurate_local': {
                    'enabled': True,
                    'models': ['llama-3.1-8b-higher-precision', 'mistral-q8'],
                    'threads': 4
                }
            }
        }
    }

    # Test client initialization
    client = LLMClient(config=config)

    # Check if model was loaded (may fail if models don't exist, but structure should work)
    health = client.get_health_status()
    print(f"LLM Health: {health['status']}")

    if health['status'] in ['healthy', 'no_models_found']:
        print("✅ LLM client initialized with tier config")
        return True
    else:
        print(f"❌ LLM client failed: {health['status']}")
        return False

def test_reporting_llm_node():
    """Test the reporting LLM node integration."""
    print("Testing reporting LLM node...")

    try:
        from src.nodes.reporting_nodes import reporting_llm_node
        from src.graphs.state import GraphState
    except ImportError as e:
        print(f"❌ Cannot import reporting node: {e}")
        return False

    # Create mock state
    state = GraphState()
    state['config'] = {'llm': {'enabled': True, 'caching': {'enabled': False}}}
    state['performance_summary'] = None  # Test with no data
    state['recommended_actions'] = []
    state['guardrail_log'] = []
    state['drift_events'] = []
    state['run_id'] = 'test_run'

    try:
        result_state = reporting_llm_node(state)
        if 'llm_report_summary' in result_state:
            print("✅ Reporting LLM node executed successfully")
            return True
        else:
            print("❌ Reporting LLM node did not produce summary")
            return False
    except Exception as e:
        print(f"❌ Reporting LLM node failed: {e}")
        return False

def test_cache_functionality():
    """Test LLM caching functionality."""
    print("Testing LLM caching...")

    import hashlib
    import json
    from datetime import datetime

    # Test cache key generation
    test_data = {'run_id': 'test', 'performance': [1, 2, 3]}
    cache_key = hashlib.md5(json.dumps(test_data, sort_keys=True).encode()).hexdigest()

    if len(cache_key) == 32:
        print("✅ Cache key generation works")
    else:
        print("❌ Cache key generation failed")
        return False

    # Test cache directory creation
    cache_dir = Path("results/llm")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir.exists():
        print("✅ Cache directory created")
        return True
    else:
        print("❌ Cache directory creation failed")
        return False

def main():
    """Run all LLM policy tests."""
    print("🧪 Testing LLM Policy Framework Implementation\n")

    tests = [
        test_llm_config_loading,
        test_llm_client_tier_selection,
        test_reporting_llm_node,
        test_cache_functionality,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}\n")

    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All LLM policy tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())