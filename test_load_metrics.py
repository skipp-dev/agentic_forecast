import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from src.agents.reporting_agent import LLMReportingAgent

def test_load_metrics():
    agent = LLMReportingAgent()
    metrics = agent._load_evaluation_metrics()
    
    if metrics:
        print(f"Loaded metrics keys: {metrics.keys()}")
        print(f"Evaluation results: {metrics.get('evaluation_results')}")
        
        comparison = metrics.get('model_comparison', {})
        print("\nModel Comparison Summary:")
        print(f"Baseline Wins: {comparison.get('baseline_wins')}")
        print(f"Challenger Wins: {comparison.get('challenger_wins')}")
        print(f"Promotions: {len(comparison.get('promotions', []))}")
        
        if comparison.get('promotions'):
            print("\nSample Promotions:")
            for p in comparison['promotions'][:5]:
                print(p)
    else:
        print("No metrics loaded.")

if __name__ == "__main__":
    test_load_metrics()
