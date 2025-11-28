#!/usr/bin/env python3
"""
Test script for GNN setup and graph construction
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_gnn_setup():
    """Test GNN dependencies and components."""
    try:
        print("Testing PyTorch...")
        import torch
        print(f"✓ PyTorch {torch.__version__} available")

        print("Testing PyTorch Geometric...")
        from torch_geometric.nn import GCNConv
        from torch_geometric.data import Data
        print("✓ PyTorch Geometric available")

        print("Testing GNN model...")
        from models.gnn_model import GNNModel
        model = GNNModel(num_node_features=10, hidden_channels=64, num_predictions=24)
        print("✓ GNN model created successfully")

        print("Testing graph construction agent...")
        from agents.graph_construction_agent import GraphConstructionAgent
        agent = GraphConstructionAgent()
        print("✓ Graph construction agent created successfully")

        print("Testing ModelZoo GNN support...")
        from models.model_zoo import ModelZoo
        zoo = ModelZoo()
        supports_gnn = zoo.supports_gnn()
        print(f"✓ ModelZoo GNN support: {supports_gnn}")

        core_families = zoo.get_core_model_families()
        has_gnn = "GNN" in core_families
        print(f"✓ GNN in core families: {has_gnn} (families: {core_families})")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gnn_setup()
    if success:
        print("\n🎉 All GNN components are working!")
    else:
        print("\n❌ GNN setup has issues")
    sys.exit(0 if success else 1)