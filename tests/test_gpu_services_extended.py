
import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from src.gpu_services import GPUServices, SpectralFeatureService

class TestGPUServicesExtended(unittest.TestCase):
    """Extended test suite for GPU Services."""

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_gpu_initialization_success(self, mock_props, mock_is_available):
        """Test successful GPU initialization with optimizations."""
        mock_props.return_value.name = "Test GPU"
        mock_props.return_value.total_memory = 8 * 1024**3
        mock_props.return_value.major = 8
        mock_props.return_value.minor = 0
        
        service = GPUServices()
        
        self.assertEqual(service.device.type, 'cuda')
        self.assertTrue(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)

    @patch('torch.cuda.is_available', return_value=False)
    def test_gpu_fallback_to_cpu(self, mock_is_available):
        """Test fallback to CPU when GPU is unavailable."""
        service = GPUServices()
        self.assertEqual(service.device.type, 'cpu')

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1024**3)
    @patch('torch.cuda.memory_reserved', return_value=2*1024**3)
    @patch('torch.cuda.get_device_properties')
    def test_memory_stats(self, mock_props, mock_reserved, mock_allocated, mock_is_available):
        """Test memory statistics reporting."""
        mock_props.return_value.total_memory = 8 * 1024**3
        
        service = GPUServices()
        stats = service.get_memory_stats()
        
        self.assertEqual(stats['allocated'], 1.0)
        self.assertEqual(stats['reserved'], 2.0)
        self.assertEqual(stats['total'], 8.0)

    def test_optimization_modes(self):
        """Test switching between training and inference modes."""
        # Mock torch.cuda.is_available to return True for this test context
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_properties'):
            
            service = GPUServices()
            
            # Test Training Mode
            service.optimize_for_training()
            self.assertTrue(torch.backends.cudnn.benchmark)
            
            # Test Inference Mode
            service.optimize_for_inference()
            self.assertFalse(torch.backends.cudnn.benchmark)

if __name__ == '__main__':
    unittest.main()
