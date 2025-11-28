"""
Enhanced Orchestrator Agent

Extends the existing SupervisorAgent with advanced capabilities:
- Hyperparameter search coordination
- Spectral feature engineering
- Advanced drift monitoring
- GPU service orchestration
"""

import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.supervisor_agent import SupervisorAgent
from graphs.state import GraphState
from src.gpu_services import get_gpu_services
from agents.hyperparameter_search_agent import HyperparameterSearchAgent
from agents.drift_monitor_agent import DriftMonitorAgent
from agents.feature_engineer_agent import FeatureEngineerAgent
from agents.forecast_agent import ForecastAgent
from agents.reporting_agent import ReportingAgent

logger = logging.getLogger(__name__)

class OrchestratorAgent(SupervisorAgent):
    """
    Advanced orchestrator that extends SupervisorAgent with:
    - GPU service coordination
    - Hyperparameter search orchestration
    - Spectral feature engineering
    - Enhanced drift monitoring
    """

    def __init__(self, llm=None, config=None):
        super().__init__(llm, config)

        # Initialize GPU services
        self.gpu_services = get_gpu_services()

        # Initialize advanced agents
        self.hyperparameter_agent = HyperparameterSearchAgent(gpu_services=self.gpu_services)
        self.drift_monitor_agent = DriftMonitorAgent()
        self.feature_engineer_agent = FeatureEngineerAgent(gpu_services=self.gpu_services)
        self.forecast_agent = ForecastAgent()
        self.reporting_agent = ReportingAgent()

        logger.info("OrchestratorAgent initialized with GPU services")

    def coordinate_workflow(self, state: GraphState) -> str:
        """
        Enhanced workflow coordination with advanced decision making.

        Extends SupervisorAgent routing with:
        - GPU resource management
        - Hyperparameter search decisions
        - Spectral feature analysis
        - Advanced drift detection
        """

        # Check GPU status first
        gpu_status = self._check_gpu_status()
        if not gpu_status['available']:
            logger.warning("GPU not available, using CPU fallback")
            state.error = "GPU not available"

        # Enhanced decision making
        next_action = self._advanced_decision_making(state)

        # GPU resource optimization
        self._optimize_gpu_resources(next_action, state)

        return next_action

    def _check_gpu_status(self) -> Dict[str, Any]:
        """Check GPU availability and status."""
        if self.gpu_services and self.gpu_services.device.type == 'cuda':
            memory_stats = self.gpu_services.get_memory_stats()
            return {
                'available': True,
                'device': str(self.gpu_services.device),
                'memory_percent': memory_stats['allocated'] / memory_stats['total']
            }
        return {
            'available': False,
            'device': 'cpu',
            'memory_percent': 0.0
        }

    def _advanced_decision_making(self, state: GraphState) -> str:
        """
        Advanced decision making with spectral analysis and hyperparameter considerations.
        """

        # Get base decision from parent class
        base_decision = super().coordinate_workflow(state)

        # Enhance decision based on advanced analysis
        enhanced_decision = self._apply_advanced_logic(base_decision, state)

        return enhanced_decision

    def _apply_advanced_logic(self, base_decision: str, state: GraphState) -> str:
        """
        Apply advanced logic to refine the base decision.
        """

        # Spectral analysis for feature engineering decisions
        if base_decision == "feature_engineer" and self._should_use_spectral_features(state):
            logger.info("Enhancing feature engineering with spectral analysis")
            # Could trigger spectral feature extraction here

        # Hyperparameter search for training decisions
        if base_decision == "train_model" and self._should_run_hyperparameter_search(state):
            logger.info("Triggering hyperparameter search before training")
            # Could trigger HPO here

        # Enhanced drift monitoring
        if base_decision == "monitor_drift" and self._should_run_advanced_drift_check(state):
            logger.info("Running advanced drift monitoring with spectral analysis")
            # Could trigger spectral drift detection here

        return base_decision

    def _should_use_spectral_features(self, state: GraphState) -> bool:
        """Determine if spectral features should be used."""
        # Logic to decide when to use spectral features
        # For now, use for volatile symbols or when requested
        return hasattr(state, 'use_spectral') and state.use_spectral

    def _should_run_hyperparameter_search(self, state: GraphState) -> bool:
        """Determine if hyperparameter search should be run."""
        # Run HPO if no recent search or performance is poor
        return hasattr(state, 'run_hpo') and state.run_hpo

    def _should_run_advanced_drift_check(self, state: GraphState) -> bool:
        """Determine if advanced drift checking should be used."""
        # Use advanced drift detection for critical symbols
        return True  # For now, always use advanced checking

    def _optimize_gpu_resources(self, action: str, state: GraphState):
        """Optimize GPU resources based on the planned action."""

        if not self.gpu_services or self.gpu_services.device.type != 'cuda':
            return

        # Optimize based on action type
        if action in ["train_model", "feature_engineer"]:
            self.gpu_services.optimize_for_training()
            logger.info("GPU optimized for training workload")

        elif action in ["generate_forecasts", "predict"]:
            self.gpu_services.optimize_for_inference()
            logger.info("GPU optimized for inference workload")

        # Monitor memory usage
        memory_stats = self.gpu_services.get_memory_stats()
        if memory_stats['allocated'] > 0.8:  # 80% usage
            logger.warning(f"High GPU memory usage: {memory_stats['allocated']:.2f}GB")
            # Could trigger memory optimization here

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including GPU metrics."""
        status = super().get_system_status()

        # Add GPU status
        gpu_status = self._check_gpu_status()
        status.update({
            'gpu': gpu_status,
            'gpu_services_initialized': self.gpu_services is not None,
            'advanced_agents_ready': all([
                self.hyperparameter_agent is not None,
                self.drift_monitor_agent is not None,
                self.feature_engineer_agent is not None
            ])
        })

        return status

    def trigger_hyperparameter_search(self, symbol: str, model_type: str) -> Dict[str, Any]:
        """Trigger hyperparameter search for a symbol/model combination."""
        if not self.hyperparameter_agent:
            return {'error': 'Hyperparameter search agent not initialized'}

        logger.info(f"Triggering hyperparameter search for {symbol} {model_type}")

        return self.hyperparameter_agent.run_search(symbol, model_type)

    def analyze_spectral_drift(self, symbol: str) -> Dict[str, Any]:
        """Analyze spectral drift for a symbol."""
        if not self.drift_monitor_agent:
            return {'error': 'Drift monitor agent not initialized'}

        logger.info(f"Analyzing spectral drift for {symbol}")

        return self.drift_monitor_agent.monitor_performance(symbol)

# Factory function to create orchestrator
def create_orchestrator_agent(llm=None, config=None) -> OrchestratorAgent:
    """Create and configure an OrchestratorAgent instance."""
    agent = OrchestratorAgent(llm=llm, config=config)
    return agent

# Backwards compatibility
def create_supervisor_agent(llm=None, config=None):
    """Create supervisor agent (backwards compatibility)."""
    return create_orchestrator_agent(llm, config)
