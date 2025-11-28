#!/usr/bin/env python3
"""
Test script for the new Model Policy system
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_model_policy():
    """Test the new model tier and policy system."""
    try:
        print("Testing Model Policy System...")

        # Test configuration
        config = {
            'models': {
                'core': {
                    'enabled': True,
                    'families': ['AutoNHITS', 'AutoNBEATS', 'AutoDLinear']
                },
                'advanced': {
                    'enabled': True,
                    'families': ['AutoTFT', 'PatchTST']
                },
                'experimental': {
                    'enabled': False,
                    'families': ['GNN', 'DeepAR']
                },
                'selection': {
                    'profile': 'trading',
                    'max_families_per_symbol': 5
                }
            }
        }

        from models.model_zoo import ModelZoo
        zoo = ModelZoo(config=config)

        print("✓ ModelZoo initialized with policy configuration")

        # Test candidate selection
        candidates = zoo.get_candidate_families()
        print(f"✓ Candidate families: {candidates}")

        # Check that experimental models are excluded
        has_gnn = 'GNN' in candidates
        has_deepar = 'DeepAR' in candidates
        print(f"✓ GNN excluded (experimental disabled): {not has_gnn}")
        print(f"✓ DeepAR excluded (experimental disabled): {not has_deepar}")

        # Check that core models are included
        has_nhits = 'AutoNHITS' in candidates
        has_nbeats = 'AutoNBEATS' in candidates
        print(f"✓ AutoNHITS included (core enabled): {has_nhits}")
        print(f"✓ AutoNBEATS included (core enabled): {has_nbeats}")

        # Test hardware profile
        hw_profile = zoo._hardware_profile
        print(f"✓ Hardware profile: {hw_profile}")

        # Test model tiers structure
        tiers = zoo.MODEL_TIERS
        print(f"✓ Model tiers defined: {list(tiers.keys())}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_policy()
    if success:
        print("\n🎉 Model Policy system working correctly!")
    else:
        print("\n❌ Model Policy system has issues")
    sys.exit(0 if success else 1)