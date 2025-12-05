import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.backtesting.backtest_executor import BacktestExecutor

class TestBacktestExecutor:
    
    @pytest.fixture
    def mock_graph(self):
        graph = MagicMock()
        
        def side_effect(state):
            cutoff = state.get('cutoff_date')
            # Return dummy forecast
            forecast_df = pd.DataFrame({
                'ds': [pd.Timestamp(cutoff) + pd.Timedelta(days=1)],
                'Baseline': [100.0]
            })
            
            return {
                'forecasts': {
                    'TEST': {
                        'Baseline': forecast_df
                    }
                },
                'recommended_actions': ['Promote Baseline TEST'],
                'performance_summary': pd.DataFrame([{
                    'symbol': 'TEST',
                    'model_family': 'Baseline',
                    'mape': 0.05
                }])
            }
            
        graph.invoke.side_effect = side_effect
        return graph

    def test_backtest_run(self, mock_graph):
        start_date = '2023-01-01'
        end_date = '2023-01-03'
        
        executor = BacktestExecutor(mock_graph, start_date, end_date)
        initial_state = {'raw_data': {}}
        
        results = executor.run(initial_state)
        
        assert len(results) == 3 # 3 days
        assert 'date' in results.columns
        assert 'forecast' in results.columns
        assert results.iloc[0]['date'] == '2023-01-01'
        assert results.iloc[0]['forecast'] == 100.0
        
        # Verify graph was called with correct cutoffs
        assert mock_graph.invoke.call_count == 3
        calls = mock_graph.invoke.call_args_list
        assert calls[0][0][0]['cutoff_date'] == '2023-01-01'
        assert calls[1][0][0]['cutoff_date'] == '2023-01-02'
        assert calls[2][0][0]['cutoff_date'] == '2023-01-03'

    def test_save_results(self, mock_graph, tmp_path):
        start_date = '2023-01-01'
        end_date = '2023-01-01'
        executor = BacktestExecutor(mock_graph, start_date, end_date)
        executor.run({})
        
        output_dir = tmp_path / "results"
        executor.save_results(str(output_dir))
        
        assert (output_dir / "forecasts.csv").exists()
        assert (output_dir / "performance.csv").exists()
