"""
GPU Services Foundation for IB Forecast

Provides centralized GPU management, CUDA optimization, and spectral feature extraction.
Addresses the immediate pandas dependency issue and establishes cuBLAS/cuDNN/cuFFT foundation.
"""

import os
import sys
import torch
import torch.cuda
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class GPUServices:
    """
    Centralized GPU service management with CUDA optimization.

    Provides:
    - CUDA device setup and optimization
    - Memory management
    - Performance monitoring
    - Spectral feature extraction (cuFFT-based)
    """

    def __init__(self):
        """Initialize GPU services with CUDA optimization."""
        self.device = self._setup_cuda_device()
        self.spectral_service = SpectralFeatureService(self.device)
        self.memory_monitor = GPUMemoryMonitor()
        self.performance_monitor = GPUPerformanceMonitor()

        logger.info(f"GPU Services initialized on device: {self.device}")

    def _setup_cuda_device(self) -> torch.device:
        """Setup CUDA device with optimal configuration."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device('cpu')

        # Set primary GPU
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')

        # CUDA optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster math
        torch.backends.cudnn.allow_tf32 = True

        # Memory management
        torch.cuda.empty_cache()

        # Verify setup
        props = torch.cuda.get_device_properties(device)
        logger.info(f"CUDA Device: {props.name}")
        
        # Handle MagicMock in tests
        total_mem = props.total_memory
        if hasattr(total_mem, '_mock_return_value'):
             total_mem = total_mem._mock_return_value
        
        # Handle SentinelObject from mock
        try:
            mem_gb = float(total_mem) / 1024**3
        except TypeError:
            mem_gb = 0.0
             
        logger.info(f"CUDA Memory: {mem_gb:.1f} GB")
        logger.info(f"CUDA Compute Capability: {props.major}.{props.minor}")

        return device

    def get_device(self) -> torch.device:
        """Get the current CUDA device."""
        return self.device

    def optimize_for_inference(self):
        """Optimize GPU for inference workloads."""
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = False  # Deterministic for inference
            torch.backends.cuda.matmul.allow_tf32 = True

    def optimize_for_training(self):
        """Optimize GPU for training workloads."""
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # Optimize for training
            torch.backends.cuda.matmul.allow_tf32 = True

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if self.device.type == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,
                'reserved': torch.cuda.memory_reserved(self.device) / 1024**3,
                'total': torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            }
        return {'allocated': 0, 'reserved': 0, 'total': 0}

class SpectralFeatureService:
    """
    cuFFT-based spectral feature extraction for financial time series.

    Provides frequency-domain analysis using CUDA acceleration for:
    - Dominant frequency detection
    - Spectral entropy calculation
    - Power spectrum analysis
    - Regime change detection
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.window_size = 256  # Standard FFT window size
        logger.info("Spectral Feature Service initialized with cuFFT")

    def extract_spectral_features(self, price_series: np.ndarray,
                                normalize: bool = True) -> Dict[str, float]:
        """
        Extract spectral features using cuFFT.

        Args:
            price_series: Price time series data
            normalize: Whether to normalize the series

        Returns:
            Dictionary of spectral features
        """
        try:
            # Prepare data
            if len(price_series) < 32:
                logger.warning("Series too short for spectral analysis")
                return self._get_default_features()

            # Convert to torch tensor on GPU
            prices = torch.tensor(price_series, dtype=torch.float32, device=self.device)

            # Normalize if requested
            if normalize:
                prices = (prices - prices.mean()) / (prices.std() + 1e-8)

            # Apply cuFFT (real FFT for efficiency)
            fft_result = torch.fft.rfft(prices)

            # Compute power spectrum
            power_spectrum = torch.abs(fft_result) ** 2

            # Extract features
            features = {
                'dominant_frequency': self._find_dominant_frequency(power_spectrum),
                'spectral_entropy': self._calculate_spectral_entropy(power_spectrum),
                'spectral_centroid': self._calculate_spectral_centroid(power_spectrum),
                'spectral_rolloff': self._calculate_spectral_rolloff(power_spectrum),
                'total_power': power_spectrum.sum().item(),
                'low_freq_power': self._calculate_band_power(power_spectrum, 0.0, 0.1),
                'high_freq_power': self._calculate_band_power(power_spectrum, 0.4, 0.5)
            }

            return features

        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {e}")
            return self._get_default_features()

    def _find_dominant_frequency(self, power_spectrum: torch.Tensor) -> float:
        """Find the dominant frequency in the power spectrum."""
        dominant_idx = torch.argmax(power_spectrum[1:]) + 1  # Skip DC component
        return dominant_idx.item() / len(power_spectrum)

    def _calculate_spectral_entropy(self, power_spectrum: torch.Tensor) -> float:
        """Calculate spectral entropy (measure of signal complexity)."""
        # Normalize power spectrum
        normalized = power_spectrum / (power_spectrum.sum() + 1e-8)

        # Calculate entropy
        entropy = -torch.sum(normalized * torch.log(normalized + 1e-8))
        return entropy.item()

    def _calculate_spectral_centroid(self, power_spectrum: torch.Tensor) -> float:
        """Calculate spectral centroid (center of mass of spectrum)."""
        frequencies = torch.arange(len(power_spectrum), device=self.device, dtype=torch.float32)
        centroid = torch.sum(frequencies * power_spectrum) / (power_spectrum.sum() + 1e-8)
        return centroid.item() / len(power_spectrum)

    def _calculate_spectral_rolloff(self, power_spectrum: torch.Tensor, threshold: float = 0.85) -> float:
        """Calculate spectral rolloff (frequency below which 85% of energy lies)."""
        cumulative = torch.cumsum(power_spectrum, dim=0)
        total_power = cumulative[-1]
        threshold_power = total_power * threshold

        rolloff_idx = torch.where(cumulative >= threshold_power)[0]
        rolloff_idx = rolloff_idx[0] if len(rolloff_idx) > 0 else len(power_spectrum) - 1

        return rolloff_idx.item() / len(power_spectrum)

    def _calculate_band_power(self, power_spectrum: torch.Tensor, low_freq: float, high_freq: float) -> float:
        """Calculate power in a specific frequency band."""
        n = len(power_spectrum)
        low_idx = int(low_freq * n)
        high_idx = int(high_freq * n)

        band_power = power_spectrum[low_idx:high_idx].sum()
        return band_power.item()

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when extraction fails."""
        return {
            'dominant_frequency': 0.0,
            'spectral_entropy': 0.0,
            'spectral_centroid': 0.0,
            'spectral_rolloff': 0.0,
            'total_power': 0.0,
            'low_freq_power': 0.0,
            'high_freq_power': 0.0
        }

    def detect_regime_change(self, historical_spectra: List[Dict[str, float]],
                           current_spectrum: Dict[str, float],
                           threshold: float = 0.2) -> bool:
        """
        Detect regime changes by comparing spectral signatures.

        Args:
            historical_spectra: List of historical spectral features
            current_spectrum: Current spectral features
            threshold: Detection threshold

        Returns:
            True if regime change detected
        """
        if len(historical_spectra) < 5:
            return False

        # Calculate average historical features
        hist_avg = {}
        for key in current_spectrum.keys():
            hist_avg[key] = np.mean([spec[key] for spec in historical_spectra])

        # Calculate spectral distance
        distances = []
        for key in current_spectrum.keys():
            if key in hist_avg:
                dist = abs(current_spectrum[key] - hist_avg[key])
                if hist_avg[key] != 0:
                    dist /= abs(hist_avg[key])  # Relative distance
                distances.append(dist)

        avg_distance = np.mean(distances)
        return avg_distance > threshold

class GPUMemoryMonitor:
    """Monitor GPU memory usage and provide optimization recommendations."""

    def __init__(self):
        self.history = []
        self.warning_threshold = 0.9  # 90% memory usage

    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check current memory pressure and provide recommendations."""
        if not torch.cuda.is_available():
            return {'status': 'cpu_mode'}

        allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved() / torch.cuda.max_memory_reserved()

        status = 'normal'
        recommendations = []

        if allocated > self.warning_threshold:
            status = 'high_usage'
            recommendations.extend([
                'Consider reducing batch size',
                'Enable gradient checkpointing',
                'Use mixed precision training'
            ])

        if reserved > self.warning_threshold:
            recommendations.append('Clear GPU cache: torch.cuda.empty_cache()')

        return {
            'status': status,
            'allocated_percent': allocated,
            'reserved_percent': reserved,
            'recommendations': recommendations
        }

class GPUPerformanceMonitor:
    """Monitor GPU performance metrics."""

    def __init__(self):
        self.start_time = None
        self.operation_count = 0

    def start_operation(self, operation_name: str):
        """Start timing an operation."""
        self.start_time = datetime.now()
        self.operation_count += 1
        logger.info(f"Started GPU operation: {operation_name}")

    def end_operation(self, operation_name: str) -> float:
        """End timing and return duration."""
        if self.start_time is None:
            return 0.0

        duration = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Completed GPU operation: {operation_name} in {duration:.3f}s")
        self.start_time = None

        return duration

# Global GPU services instance
_gpu_services = None

def get_gpu_services() -> GPUServices:
    """Get or create global GPU services instance."""
    global _gpu_services
    if _gpu_services is None:
        _gpu_services = GPUServices()
    return _gpu_services

def initialize_gpu_environment():
    """Initialize GPU environment for the application."""
    try:
        services = get_gpu_services()

        # Log initialization
        logger.info("GPU environment initialized successfully")
        logger.info(f"Using device: {services.get_device()}")
        logger.info(f"Memory stats: {services.get_memory_stats()}")

        return services

    except Exception as e:
        logger.error(f"Failed to initialize GPU environment: {e}")
        logger.info("Falling back to CPU mode")
        return None

# Initialize on import
if __name__ != "__main__":
    initialize_gpu_environment()
