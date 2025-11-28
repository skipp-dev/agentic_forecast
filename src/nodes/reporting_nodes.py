import pandas as pd
from ..graphs.state import GraphState
import os

def generate_report_node(state: GraphState) -> GraphState:
    """
    Generates a comprehensive CSV report of model performance.
    """
    print("--- Node: Generate Comprehensive Report ---")
    
    hpo_results = state.get('hpo_results', {})
    all_performance_data = []

    for symbol, results in hpo_results.items():
        for model_family, result in results.items():
            if result:
                performance_data = {
                    'symbol': symbol,
                    'model_family': model_family,
                    'mape': result.best_val_mape,
                    'mae': result.best_val_mae,
                    'model_id': result.best_model_id,
                    'artifact_path': result.artifact_info.artifact_uri
                }
                all_performance_data.append(performance_data)

    if all_performance_data:
        report_df = pd.DataFrame(all_performance_data)
        
        # Save report to disk
        report_path = "/app/data/reports"
        os.makedirs(report_path, exist_ok=True)
        report_filename = os.path.join(report_path, "full_model_evaluation_report.csv")
        
        report_df.to_csv(report_filename, index=False)
        
        print(f"✅ Comprehensive performance report saved to {report_filename}")
        
        # For display, show the top 5 models by MAPE
        print("\nTop 5 Performing Models (by MAPE):")
        print(report_df.sort_values(by='mape').head(5))
    else:
        print("⚠️ No performance data available to generate a report.")

    return state
