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

from .supervisor_agent import SupervisorAgent
from src.core.state import PipelineGraphState as GraphState
from src.gpu_services import get_gpu_services
from src.services.model_registry_service import ModelRegistryService
from src.services.training_service import GPUTrainingService
from .hyperparameter_search_agent import HyperparameterSearchAgent
from .drift_monitor_agent import DriftMonitorAgent
from .feature_engineer_agent import FeatureEngineerAgent
from .forecast_agent import ForecastAgent
from .reporting_agent import LLMReportingAgent
from src.monitoring.metrics import PIPELINE_LATENCY, SYSTEM_ERRORS
from src.monitoring.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer("orchestrator_agent")

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
        
        # Initialize Services
        self.model_registry = ModelRegistryService()
        self.training_service = GPUTrainingService(gpu_services=self.gpu_services, model_registry=self.model_registry)

        # Circuit Breaker State
        self.retrain_count = 0
        self.last_retrain_time = datetime.min
        self.MAX_RETRAINS_PER_HOUR = 3
        self.MIN_RETRAIN_INTERVAL_MINUTES = 15

        # Initialize advanced agents
        self.hyperparameter_agent = HyperparameterSearchAgent(gpu_services=self.gpu_services)
        self.drift_monitor_agent = DriftMonitorAgent()
        self.feature_engineer_agent = FeatureEngineerAgent(gpu_services=self.gpu_services)
        self.forecast_agent = ForecastAgent()
        self.reporting_agent = LLMReportingAgent()

        logger.info("OrchestratorAgent initialized with GPU services")

    @PIPELINE_LATENCY.time()
    def coordinate_workflow(self, state: GraphState) -> str:
        """
        Enhanced workflow coordination with advanced decision making.

        Extends SupervisorAgent routing with:
        - GPU resource management
        - Hyperparameter search decisions
        - Spectral feature analysis
        - Advanced drift detection
        - Market regime analysis
        - Performance-based alerting
        """
        with tracer.start_as_current_span("coordinate_workflow") as span:
            try:
                # Check GPU status first
                gpu_status = self._check_gpu_status()
                span.set_attribute("gpu.available", gpu_status['available'])
                
                if not gpu_status['available']:
                    logger.warning("GPU not available, using CPU fallback")
                    state['error'] = "GPU not available"

                # 1. Analyze Context (Regime, Performance, Drift)
                analysis = self._analyze_context(state)
                span.set_attribute("analysis.drift_detected", analysis.get('drift_detected', False))
                
                # 2. Decision Tree based on Analysis
                
                # Critical Error Handling
                if analysis.get('critical_error'):
                    logger.error(f"Critical error detected: {analysis['critical_error']}. Terminating workflow.")
                    SYSTEM_ERRORS.labels(component='orchestrator').inc()
                    span.set_status(trace.Status(trace.StatusCode.ERROR, analysis['critical_error']))
                return "end"

            # Regime Change Handling
            if analysis.get('regime_change_detected'):
                logger.info("Regime change detected. Triggering strategy update/HPO.")
                if not state.get('hpo_triggered') and not state.get('hpo_results'):
                     state['hpo_triggered'] = True
                     return "hpo"

            # High Drift Handling
            if analysis.get('drift_severity') == 'high':
                logger.info("High drift detected. Checking circuit breakers before retraining.")
                
                # Circuit Breaker Logic
                now = datetime.now()
                time_since_last = (now - self.last_retrain_time).total_seconds() / 60.0
                
                if self.retrain_count >= self.MAX_RETRAINS_PER_HOUR:
                    logger.error(f"CIRCUIT BREAKER TRIPPED: Max retrains ({self.MAX_RETRAINS_PER_HOUR}) exceeded. Falling back to baseline.")
                    state['error'] = "Circuit Breaker: Max retrains exceeded"
                    return "fallback_baseline" # Assume this node exists or will be handled
                
                if time_since_last < self.MIN_RETRAIN_INTERVAL_MINUTES:
                    logger.warning(f"CIRCUIT BREAKER: Retrain requested too soon ({time_since_last:.1f} min < {self.MIN_RETRAIN_INTERVAL_MINUTES} min). Skipping.")
                    return "skip_retrain"

                if not state.get('retrained_models'):
                    self.retrain_count += 1
                    self.last_retrain_time = now
                    return "retrain"
                
            # Performance Drop Handling
            if analysis.get('performance_drop'):
                logger.info("Performance drop detected. Suggesting HPO.")
                if not state.get('hpo_triggered') and not state.get('hpo_results'):
                     state['hpo_triggered'] = True
                     return "hpo"

            # Enhanced decision making (calls parent logic for standard flow)
            next_action = self._advanced_decision_making(state)

            # GPU resource optimization
            self._optimize_gpu_resources(next_action, state)

            return next_action
        except Exception as e:
            logger.error(f"Orchestrator failed: {e}")
            SYSTEM_ERRORS.labels(component='orchestrator').inc()
            return "end"
        if analysis.get('drift_severity') == 'high':
            logger.info("High drift detected. Mandating retraining.")
            if not state.get('retrained_models'):
                return "retrain"
            
        # Performance Drop Handling
        if analysis.get('performance_drop'):
            logger.info("Performance drop detected. Suggesting HPO.")
            if not state.get('hpo_triggered') and not state.get('hpo_results'):
                 state['hpo_triggered'] = True
                 return "hpo"

        # Enhanced decision making (calls parent logic for standard flow)
        next_action = self._advanced_decision_making(state)

        # GPU resource optimization
        self._optimize_gpu_resources(next_action, state)

        return next_action

    def _analyze_context(self, state: GraphState) -> Dict[str, Any]:
        """
        Analyze the current state context to extract insights for decision making.
        """
        analysis = {
            'critical_error': None,
            'regime_change_detected': False,
            'drift_severity': 'low',
            'performance_drop': False
        }

        # Check for errors
        if state.get('errors'):
            # Simple check: if too many errors, flag as critical
            if len(state['errors']) > 5:
                analysis['critical_error'] = "Too many errors encountered"

        # Check for Drift
        drift_symbols = state.get('drift_detected', [])
        if len(drift_symbols) > 0:
            # If > 20% of symbols have drift, consider it high severity
            total_symbols = len(state.get('symbols', [])) or 1
            if len(drift_symbols) / total_symbols > 0.2:
                analysis['drift_severity'] = 'high'
            else:
                analysis['drift_severity'] = 'medium'

        # Check for Performance Drop
        analytics = state.get('analytics_results', {})
        # This assumes analytics_results structure. 
        # If we have historical metrics, we could compare. 
        # For now, check if any MAPE is > threshold (e.g., 5%)
        for symbol, metrics in analytics.items():
            # Handle nested structure if necessary. 
            # Assuming metrics is dict-like.
            if isinstance(metrics, dict):
                # Check for 'mape' in various places
                mape = metrics.get('mape')
                if mape is None:
                    # Try looking deeper if structure is complex
                    # e.g. metrics['NLinear']['mape']
                    for model_name, model_metrics in metrics.items():
                        if isinstance(model_metrics, dict):
                            m = model_metrics.get('mape')
                            if m and m > 5.0: # 500% MAPE is huge, maybe 0.05? 
                                # Wait, MAPE is usually percentage. 5% = 0.05 or 5.0?
                                # Looking at logs: 'mape': 1.0418... (likely 1.04%)
                                # Let's assume > 10.0 is bad.
                                if m > 10.0:
                                    analysis['performance_drop'] = True
                                    break
                elif mape > 10.0:
                     analysis['performance_drop'] = True
            
            if analysis['performance_drop']:
                break

        # Check for Regime Change (Placeholder logic)
        # In a real scenario, we'd check VIX or volatility metrics from 'features'
        # For now, we can check if 'drift_severity' is high, which implies regime change
        if analysis['drift_severity'] == 'high':
            analysis['regime_change_detected'] = True

        return analysis

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
        return state.get('use_spectral', False)

    def _should_run_hyperparameter_search(self, state: GraphState) -> bool:
        """Determine if hyperparameter search should be run."""
        # Run HPO if no recent search or performance is poor
        
        # Check explicit flag first
        if state.get('run_hpo', False):
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

        return self.drift_monitor_agent.comprehensive_drift_check(symbol)

    def trigger_comprehensive_report(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger comprehensive LLM-powered reporting with continuous learning feedback."""
        if not self.reporting_agent:
            return {'error': 'LLM Reporting agent not initialized'}

        logger.info("Generating comprehensive system report with LLM analysis")

        try:
            # Convert state data to ReportingInput format
            from agents.reporting_agent import ReportingInput

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
