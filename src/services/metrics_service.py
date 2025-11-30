# services/metrics_service.py
"""
Metrics service for tracking LLM usage and system metrics.

Provides functionality to record LLM calls, report generation, and other system metrics.
"""

import json
import os
import time
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime


class MetricsService:
    """Service for tracking and storing system metrics."""

    def __init__(self, metrics_dir: str = "data/metrics"):
        self.metrics_dir = metrics_dir
        self.llm_metrics_file = os.path.join(metrics_dir, "llm_metrics.json")
        self.report_metrics_file = os.path.join(metrics_dir, "report_metrics.json")

        # Ensure metrics directory exists
        os.makedirs(metrics_dir, exist_ok=True)

        # Load existing metrics
        self.llm_metrics = self._load_metrics(self.llm_metrics_file)
        self.report_metrics = self._load_metrics(self.report_metrics_file)

    def _load_metrics(self, filepath: str) -> Dict[str, Any]:
        """Load metrics from JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_metrics(self, metrics: Dict[str, Any], filepath: str) -> None:
        """Save metrics to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving metrics to {filepath}: {e}")

    def record_llm_call(self, agent_name: str, model: str, tokens_used: int,
                       cost: float, success: bool, duration: float) -> None:
        """
        Record an LLM API call.

        Args:
            agent_name: Name of the agent making the call
            model: Model used (e.g., 'gpt-4o-mini')
            tokens_used: Number of tokens consumed
            cost: Estimated cost in USD
            success: Whether the call was successful
            duration: Call duration in seconds
        """
        if agent_name not in self.llm_metrics:
            self.llm_metrics[agent_name] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_tokens': 0,
                'total_cost': 0.0,
                'total_duration': 0.0,
                'last_call_timestamp': None,
                'models_used': set(),
                'calls_by_model': defaultdict(int)
            }

        agent_metrics = self.llm_metrics[agent_name]
        agent_metrics['total_calls'] += 1
        agent_metrics['total_tokens'] += tokens_used
        agent_metrics['total_cost'] += cost
        agent_metrics['total_duration'] += duration
        agent_metrics['last_call_timestamp'] = datetime.now().isoformat()

        if success:
            agent_metrics['successful_calls'] += 1
        else:
            agent_metrics['failed_calls'] += 1

        agent_metrics['models_used'].add(model)
        agent_metrics['calls_by_model'][model] += 1

        # Convert set to list for JSON serialization
        agent_metrics['models_used'] = list(agent_metrics['models_used'])

        self._save_metrics(self.llm_metrics, self.llm_metrics_file)

    def record_report_generation(self, report_type: str, success: bool,
                               duration: float, size_bytes: int = 0) -> None:
        """
        Record report generation metrics.

        Args:
            report_type: Type of report (e.g., 'daily', 'weekly')
            success: Whether generation was successful
            duration: Generation duration in seconds
            size_bytes: Size of generated report in bytes
        """
        timestamp = datetime.now().isoformat()

        report_entry = {
            'timestamp': timestamp,
            'report_type': report_type,
            'success': success,
            'duration': duration,
            'size_bytes': size_bytes
        }

        if 'report_generations' not in self.report_metrics:
            self.report_metrics['report_generations'] = []

        self.report_metrics['report_generations'].append(report_entry)

        # Keep only last 1000 entries to prevent file from growing too large
        if len(self.report_metrics['report_generations']) > 1000:
            self.report_metrics['report_generations'] = self.report_metrics['report_generations'][-1000:]

        # Update summary stats
        generations = self.report_metrics['report_generations']
        successful = sum(1 for g in generations if g['success'])
        total = len(generations)

        self.report_metrics['summary'] = {
            'total_reports': total,
            'successful_reports': successful,
            'failed_reports': total - successful,
            'success_rate': successful / total if total > 0 else 0,
            'last_generation': timestamp
        }

        self._save_metrics(self.report_metrics, self.report_metrics_file)

    def get_llm_metrics(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get LLM metrics for a specific agent or all agents.

        Args:
            agent_name: Specific agent name, or None for all agents

        Returns:
            Dictionary containing LLM metrics
        """
        if agent_name:
            return self.llm_metrics.get(agent_name, {})
        return self.llm_metrics

    def get_report_metrics(self) -> Dict[str, Any]:
        """Get report generation metrics."""
        return self.report_metrics

    def get_system_health_metrics(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        llm_health = {}
        for agent, metrics in self.llm_metrics.items():
            total_calls = metrics.get('total_calls', 0)
            successful_calls = metrics.get('successful_calls', 0)
            success_rate = successful_calls / total_calls if total_calls > 0 else 0
            llm_health[agent] = {
                'success_rate': success_rate,
                'total_calls': total_calls,
                'last_call': metrics.get('last_call_timestamp')
            }

        report_health = self.report_metrics.get('summary', {})

        return {
            'llm_health': llm_health,
            'report_health': report_health,
            'timestamp': datetime.now().isoformat()
        }


# Global instance for easy access
_metrics_service = None

def get_metrics_service() -> MetricsService:
    """Get the global metrics service instance."""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service