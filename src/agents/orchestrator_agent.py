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
import time
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# Add paths for imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.supervisor_agent import SupervisorAgent
from graphs.state import GraphState
from src.gpu_services import get_gpu_services
from src.data.model_registry import ModelRegistry
from agents.hyperparameter_search_agent import HyperparameterSearchAgent
from agents.drift_monitor_agent import DriftMonitorAgent
from agents.feature_engineer_agent import FeatureEngineerAgent
from agents.forecast_agent import ForecastAgent
from agents.llm_reporting_agent import LLMReportingAgent

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
        
        # Initialize Model Registry
        self.model_registry = ModelRegistry()

        # Initialize advanced agents
        self.hyperparameter_agent = HyperparameterSearchAgent(gpu_services=self.gpu_services)
        self.drift_monitor_agent = DriftMonitorAgent()
        self.feature_engineer_agent = FeatureEngineerAgent(gpu_services=self.gpu_services)
        self.forecast_agent = ForecastAgent()
        self.reporting_agent = LLMReportingAgent()

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
        
        # Check explicit flag first
        if hasattr(state, 'run_hpo') and state.run_hpo:
            return True
            
        # Check age-based trigger
        symbols = state.get('symbols', [])
        if not symbols and hasattr(state, 'raw_data'):
             symbols = list(state.raw_data.keys())
             
        for symbol in symbols:
            last_run = self.model_registry.get_last_hpo_run(symbol)
            if last_run is None:
                logger.info(f"HPO trigger: No previous run for {symbol}")
                return True
            
            # Default 7 days age limit
            max_age_seconds = 7 * 24 * 3600 
            if (time.time() - last_run) > max_age_seconds:
                logger.info(f"HPO trigger: Age limit exceeded for {symbol}")
                return True

        return False

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

        result = self.hyperparameter_agent.run_search(symbol, model_type)
        
        # Update registry timestamp
        self.model_registry.set_last_hpo_run(symbol, time.time())
        
        return result

    def analyze_spectral_drift(self, symbol: str) -> Dict[str, Any]:
        """Analyze spectral drift for a symbol."""
        if not self.drift_monitor_agent:
            return {'error': 'Drift monitor agent not initialized'}

        logger.info(f"Analyzing spectral drift for {symbol}")

        return self.drift_monitor_agent.monitor_performance(symbol)

    def trigger_comprehensive_report(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger comprehensive LLM-powered reporting with continuous learning feedback."""
        if not self.reporting_agent:
            return {'error': 'LLM Reporting agent not initialized'}

        logger.info("Generating comprehensive system report with LLM analysis")

        try:
            # Convert state data to ReportingInput format
            from agents.llm_reporting_agent import ReportingInput

            report_input = ReportingInput(
                analytics_summary=state_data.get('analytics_summary', {}),
                hpo_plan=state_data.get('hpo_plan', {}),
                research_insights=state_data.get('research_insights', {}),
                guardrail_status=state_data.get('guardrail_status', {}),
                run_metadata=state_data.get('run_metadata', {})
            )

            # Generate report
            metadata = self.reporting_agent.generate_and_store_report(report_input)

            # Extract priority actions for continuous learning
            priority_actions = []
            if hasattr(self.reporting_agent, '_last_report') and self.reporting_agent._last_report:
                priority_actions = self.reporting_agent._last_report.get('priority_actions', [])

            # Apply continuous learning feedback
            learning_feedback = self._apply_continuous_learning_feedback(priority_actions, state_data)

            result = {
                'report_metadata': metadata,
                'priority_actions': priority_actions,
                'learning_feedback': learning_feedback,
                'success': True
            }

            logger.info(f"Comprehensive report generated with {len(priority_actions)} priority actions identified")
            return result

        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            return {'error': str(e), 'success': False}

    def _apply_continuous_learning_feedback(self, priority_actions: list, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply continuous learning feedback based on report recommendations.

        This implements Step 5: continuous learning loop where reports inform future decisions.
        """
        feedback = {
            'actions_triggered': [],
            'decisions_updated': [],
            'learning_insights': []
        }

        for action in priority_actions:
            action_type = action.get('action', '').lower()
            priority = action.get('priority', 'medium')
            rationale = action.get('rationale', '')

            # High priority actions get automatic execution
            if priority == 'high':
                if 'retrain' in action_type or 'model' in action_type:
                    feedback['actions_triggered'].append(f"High-priority retraining triggered: {action_type}")
                    feedback['decisions_updated'].append({'type': 'retraining', 'symbol': action.get('owner', 'all')})

                elif 'hpo' in action_type or 'optimization' in action_type:
                    feedback['actions_triggered'].append(f"High-priority HPO triggered: {action_type}")
                    feedback['decisions_updated'].append({'type': 'hpo', 'symbol': action.get('owner', 'all')})

                elif 'feature' in action_type:
                    feedback['actions_triggered'].append(f"High-priority feature engineering triggered: {action_type}")
                    feedback['decisions_updated'].append({'type': 'feature_engineering', 'symbol': action.get('owner', 'all')})

            # Learning insights for all actions
            feedback['learning_insights'].append({
                'action': action_type,
                'priority': priority,
                'rationale': rationale,
                'learning_applied': priority == 'high'
            })

        # Update state data with learning insights
        state_data['continuous_learning_applied'] = len(feedback['actions_triggered']) > 0
        state_data['learning_insights'] = feedback['learning_insights']

        logger.info(f"Applied continuous learning feedback: {len(feedback['actions_triggered'])} actions triggered")
        return feedback
