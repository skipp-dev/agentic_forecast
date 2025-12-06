
import unittest
from unittest.mock import MagicMock, patch
from src.agents.orchestrator_agent import OrchestratorAgent
from src.core.state import PipelineGraphState

class TestOrchestratorExtended(unittest.TestCase):
    """Extended test suite for Orchestrator Agent logic."""

    def setUp(self):
        self.mock_gpu_services = MagicMock()
        self.mock_gpu_services.get_memory_stats.return_value = {'available': True}
        
        with patch('src.agents.orchestrator_agent.get_gpu_services', return_value=self.mock_gpu_services):
            self.agent = OrchestratorAgent()

    def test_gpu_check_logic(self):
        """Test that orchestrator correctly checks GPU status."""
        # Mock internal check
        self.agent._check_gpu_status = MagicMock(return_value={'available': True})
        self.agent._analyze_context = MagicMock(return_value={})
        self.agent._decide_next_step = MagicMock(return_value="next_step")
        
        state = PipelineGraphState(symbols=['AAPL'])
        self.agent.coordinate_workflow(state)
        
        self.agent._check_gpu_status.assert_called_once()

    def test_cpu_fallback_logic(self):
        """Test that orchestrator handles GPU unavailability."""
        self.agent._check_gpu_status = MagicMock(return_value={'available': False})
        self.agent._analyze_context = MagicMock(return_value={})
        self.agent._decide_next_step = MagicMock(return_value="next_step")
        
        state = PipelineGraphState(symbols=['AAPL'])
        self.agent.coordinate_workflow(state)
        
        self.assertIn('error', state)
        self.assertEqual(state['error'], "GPU not available")

if __name__ == '__main__':
    unittest.main()
