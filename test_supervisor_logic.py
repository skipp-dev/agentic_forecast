
import unittest
from unittest.mock import MagicMock
from src.agents.supervisor_agent import SupervisorAgent
from src.graphs.state import GraphState

class TestSupervisorLogic(unittest.TestCase):
    def setUp(self):
        self.agent = SupervisorAgent()
        self.base_state: GraphState = {
            'run_status': 'RUNNING',
            'supervisor_iterations': 1,
            'errors': [],
            'hpo_triggered': False,
            'drift_detected': False,
            'horizon_forecasts': {},
            'interpreted_forecasts': {},
            'hpo_results': {},
            'retrained_models': {},
            'deep_research_conducted': False,
            'next_step': None
        }

    def test_completion_condition(self):
        """Test that presence of interpreted_forecasts leads to 'end'."""
        state = self.base_state.copy()
        state['interpreted_forecasts'] = {'AAPL': 'Some forecast'}
        
        decision = self.agent.coordinate_workflow(state)
        self.assertEqual(decision, 'end')

    def test_hpo_trigger(self):
        """Test that HPO trigger leads to 'hpo'."""
        state = self.base_state.copy()
        state['hpo_triggered'] = True
        
        decision = self.agent.coordinate_workflow(state)
        self.assertEqual(decision, 'hpo')

    def test_hpo_loop_prevention(self):
        """Test that HPO trigger is ignored if results exist."""
        state = self.base_state.copy()
        state['hpo_triggered'] = True
        state['hpo_results'] = {'some': 'results'}
        # Should fall through to next check. If nothing else, 'continue' or 'end' if forecasts exist.
        # Here we don't have forecasts, so it might return 'continue'
        
        decision = self.agent.coordinate_workflow(state)
        self.assertNotEqual(decision, 'hpo')
        self.assertEqual(decision, 'continue')

    def test_drift_trigger(self):
        """Test that Drift trigger leads to 'retrain'."""
        state = self.base_state.copy()
        state['drift_detected'] = True
        
        decision = self.agent.coordinate_workflow(state)
        self.assertEqual(decision, 'retrain')

    def test_drift_loop_prevention(self):
        """Test that Drift trigger is ignored if retrained models exist."""
        state = self.base_state.copy()
        state['drift_detected'] = True
        state['retrained_models'] = {'model': 'path'}
        
        decision = self.agent.coordinate_workflow(state)
        self.assertNotEqual(decision, 'retrain')
        self.assertEqual(decision, 'continue')

    def test_deep_research_trigger(self):
        """Test that low confidence leads to 'deep_research'."""
        state = self.base_state.copy()
        # Mock forecast object with low confidence
        mock_forecast = MagicMock()
        mock_forecast.confidence = "Low"
        state['horizon_forecasts'] = {'AAPL': [mock_forecast]}
        
        decision = self.agent.coordinate_workflow(state)
        self.assertEqual(decision, 'deep_research')

    def test_deep_research_loop_prevention(self):
        """Test that low confidence is ignored if deep research already conducted."""
        state = self.base_state.copy()
        mock_forecast = MagicMock()
        mock_forecast.confidence = "Low"
        state['horizon_forecasts'] = {'AAPL': [mock_forecast]}
        state['deep_research_conducted'] = True
        
        decision = self.agent.coordinate_workflow(state)
        self.assertNotEqual(decision, 'deep_research')
        self.assertEqual(decision, 'continue')

    def test_run_status_completed(self):
        """Test that COMPLETED status returns 'end'."""
        state = self.base_state.copy()
        state['run_status'] = 'COMPLETED'
        
        decision = self.agent.coordinate_workflow(state)
        self.assertEqual(decision, 'end')

if __name__ == '__main__':
    unittest.main()
