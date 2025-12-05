import pytest
import pandas as pd
import numpy as np
from pipelines.run_features import FeatureEngineer
from src.utils.time_machine import TimeMachine

class TestTimeMachineIntegration:
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2022-11-01', end='2023-04-10', freq='D') # Enough history for SMA50
        df = pd.DataFrame({
            'open': np.random.rand(len(dates)) * 100,
            'high': np.random.rand(len(dates)) * 100,
            'low': np.random.rand(len(dates)) * 100,
            'close': np.random.rand(len(dates)) * 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        return df

    def test_time_machine_cutoff(self, sample_data):
        """Test that TimeMachine correctly filters data based on cutoff date."""
        cutoff = '2023-02-01'
        tm = TimeMachine(sample_data)
        filtered_df = tm.get_data_as_of(cutoff)
        
        assert filtered_df.index.max() <= pd.Timestamp(cutoff)
        # Should have data up to cutoff
        assert len(filtered_df) > 0

    def test_feature_engineer_cutoff(self, sample_data):
        """Test that FeatureEngineer respects the cutoff_date parameter."""
        engineer = FeatureEngineer()
        cutoff = '2023-02-01'
        
        # Run feature engineering with cutoff
        features = engineer.engineer_features_for_symbol(
            symbol='TEST',
            data=sample_data,
            experiment='baseline',
            cutoff_date=cutoff
        )
        
        # Check that the last index in features is not after the cutoff
        assert not features.empty
        assert features.index.max() <= pd.Timestamp(cutoff)

    def test_feature_engineer_no_cutoff(self, sample_data):
        """Test that FeatureEngineer uses all data when no cutoff is provided."""
        engineer = FeatureEngineer()
        
        # Run feature engineering without cutoff
        features = engineer.engineer_features_for_symbol(
            symbol='TEST',
            data=sample_data,
            experiment='baseline'
        )
        
        # Check that we have data beyond the cutoff used in the other test
        cutoff = '2023-02-01'
        assert not features.empty
        assert features.index.max() > pd.Timestamp(cutoff)
