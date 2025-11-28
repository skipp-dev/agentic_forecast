#!/usr/bin/env python3
"""
Full cycle test script to run the IB Forecast pipeline with all symbols
"""

import os
import sys
import yaml
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.unified_ingestion_v2 import UnifiedDataIngestion
from src.agents.feature_agent import FeatureAgent
from src.nodes.execution_nodes import forecasting_node
from src.nodes.agent_nodes import analytics_agent_node
from src.graphs.state import GraphState

def main():
    print("🚀 Starting full cycle IB Forecast pipeline with all symbols...")

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load ALL symbols from watchlist
    symbols = pd.read_csv('watchlist_ibkr.csv')['Symbol'].tolist()
    print(f"📊 Loaded {len(symbols)} symbols from IBKR watchlist")

    # Initialize data ingestion
    print("--- Loading Data ---")
    data_ingestion = UnifiedDataIngestion(use_real_data=True, config=config, skip_sentiment=True)
    data_ingestion.initialize()

    # Load raw data for all symbols
    raw_data = {}
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')

    successful_loads = 0
    for i, symbol in enumerate(symbols):
        if (i + 1) % 50 == 0:  # Progress update every 50 symbols
            print(f"Loading progress: {i+1}/{len(symbols)} symbols ({successful_loads} successful)")

        try:
            data = data_ingestion.get_historical_data(symbol, start_date, end_date, '1 day')
            if data is not None and not data.empty:
                raw_data[symbol] = data
                successful_loads += 1
            else:
                print(f"❌ Failed to load {symbol}")
        except Exception as e:
            print(f"❌ Error loading {symbol}: {e}")

    print(f"✅ Loaded data for {len(raw_data)}/{len(symbols)} symbols")

    # Generate features
    print("\n--- Generating Features ---")
    feature_agent = FeatureAgent()
    features = feature_agent.generate_features(raw_data)
    print(f"✅ Generated features for {len(features)} symbols")

    # Create state for next steps
    state = GraphState(
        symbols=list(raw_data.keys()),  # Only symbols with data
        config=config,
        raw_data=raw_data,
        features=features,
        forecasts={},
        performance_summary=pd.DataFrame(),
        drift_metrics=pd.DataFrame(),
        risk_kpis=pd.DataFrame(),
        anomalies={},
        recommended_actions=[],
        executed_actions=[],
        retrained_models={},
        best_models={},
        errors=[],
        hpo_results={},
        shap_results={},
        analytics_summary=pd.DataFrame(),
        hpo_decision={},
        retraining_history=[],
        guardrail_log=[],
        hpo_triggered=False,
        drift_detected=False,
        edge_index=None,
        node_features=None,
        symbol_to_idx={}
    )

    # Run forecasting
    print("\n--- Running Forecasting ---")
    try:
        forecast_state = forecasting_node(state)
        print("✅ Forecasting completed")
        print(f"📈 Generated forecasts for {len(forecast_state.get('forecasts', {}))} symbols")
    except Exception as e:
        print(f"❌ Forecasting failed: {e}")
        return

    # Run analytics
    print("\n--- Running Analytics ---")
    try:
        analytics_state = analytics_agent_node(forecast_state)
        print("✅ Analytics completed")

        # Save results
        if not analytics_state.get('analytics_summary').empty:
            results_file = f"full_cycle_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            analytics_state['analytics_summary'].to_csv(results_file)
            print(f"💾 Results saved to {results_file}")

            # Show top results
            print("\n🏆 Top 10 Models by MAPE:")
            top_results = analytics_state['analytics_summary'].sort_values('mape').head(10)
            for _, row in top_results.iterrows():
                print(".4f")

    except Exception as e:
        print(f"❌ Analytics failed: {e}")

    print("\n🎉 Full cycle pipeline completed!")

if __name__ == "__main__":
    main()