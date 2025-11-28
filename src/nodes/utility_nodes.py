from ..graphs.state import GraphState

def reporting_node(state: GraphState) -> GraphState:
    """
    Generates a report of the forecasting pipeline's execution.
    """
    print("--- Node: Reporting ---")
    
    hpo_results = state.get('hpo_results', {})
    if hpo_results:
        print("HPO Results Summary:")
        for symbol, results in hpo_results.items():
            print(f"  {symbol}:")
            for model, result in results.items():
                if result:
                    print(f"    {model}: MAPE={result.best_val_mape:.4f}, MAE={result.best_val_mae:.4f}")
    
    shap_results = state.get('shap_results', {})
    if shap_results:
        print("SHAP Explainability Summary:")
        for symbol, results in shap_results.items():
            print(f"  {symbol} ({results.get('model_family', 'Unknown Model')}):")
            feature_importance = results.get('feature_importance')
            if feature_importance is not None and not feature_importance.empty:
                print("    Top 5 Most Important Features:")
                for idx, row in feature_importance.head(5).iterrows():
                    print(f"      {row['feature']}: {row['importance']:.4f}")
            print(f"    Sample Size: {results.get('sample_size', 'N/A')}")
            print("    SHAP analysis provides insights into feature contributions to predictions.")
    
    print("✅ Generated report.")
    return state
