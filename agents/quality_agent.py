#!/usr/bin/env python3
"""
Quality Assurance Agent

Monitors the agentic forecasting system for data quality issues, calculation errors,
and system health problems. Automatically detects and reports anomalies.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import requests
import subprocess
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
import sqlite3

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import requests
import subprocess
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
import sqlite3

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class QualityAssuranceAgent:
    """
    Quality Assurance Agent that monitors the forecasting system for:
    - Data quality issues (MAPE=1.0, identical values across symbols)
    - Calculation errors and anomalies
    - System health (hanging processes, stuck pipelines)
    - Data freshness and staleness
    - Streamlit dashboard functionality
    """

    def __init__(self, db_path: str = 'data/quality_monitoring.db'):
        """
        Initialize the Quality Assurance Agent.

        Args:
            db_path: Path to SQLite database for storing quality metrics
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Quality thresholds
        self.thresholds = {
            'max_identical_symbols': 10,  # Max symbols with identical metrics
            'max_mape_value': 0.99,       # MAPE values >= 1.0 indicate problems
            'min_unique_mae_values': 5,   # Minimum unique MAE values expected
            'max_staleness_hours': 24,    # Max hours before data is considered stale
            'max_pipeline_runtime': 3600, # Max seconds for pipeline to run
            'streamlit_check_interval': 300, # Check streamlit every 5 minutes
        }

        # Monitoring state
        self.last_checks = {}
        self.active_alerts = []
        self.monitoring_active = False

        # Initialize database
        self._init_database()

        logger.info("Quality Assurance Agent initialized")

    def _init_database(self):
        """Initialize SQLite database for quality monitoring."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS quality_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    check_type TEXT,
                    status TEXT,
                    details TEXT,
                    severity TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT,
                    message TEXT,
                    severity TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT,
                    value REAL,
                    check_type TEXT
                )
            ''')

    def start_monitoring(self, interval_seconds: int = 300):
        """
        Start continuous monitoring in a background thread.

        Args:
            interval_seconds: Monitoring interval in seconds
        """
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Quality monitoring started with {interval_seconds}s interval")

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        logger.info("Quality monitoring stopped")

    def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self.run_full_quality_check()
            except Exception as e:
                logger.error(f"Error in quality monitoring: {e}")
                self._log_alert('monitoring_error', f'Monitoring loop error: {e}', 'high')

            time.sleep(interval)

    def run_full_quality_check(self) -> Dict[str, Any]:
        """
        Run comprehensive quality checks on the system.

        Returns:
            Dictionary with check results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'alerts': [],
            'overall_status': 'healthy'
        }

        # Run all quality checks
        check_functions = [
            self._check_evaluation_metrics_quality,
            self._check_data_freshness,
            self._check_pipeline_health,
            self._check_streamlit_status,
            self._check_calculation_consistency,
            self._check_data_staleness
        ]

        for check_func in check_functions:
            try:
                check_name = check_func.__name__.replace('_check_', '')
                check_result = check_func()
                results['checks'][check_name] = check_result

                # Log to database
                self._log_check_result(check_name, check_result)

                # Check for alerts
                if check_result.get('status') == 'failed':
                    severity = check_result.get('severity', 'medium')
                    results['alerts'].append({
                        'type': check_name,
                        'message': check_result.get('message', 'Check failed'),
                        'severity': severity
                    })

                    if severity in ['high', 'critical']:
                        results['overall_status'] = 'unhealthy'

            except Exception as e:
                logger.error(f"Error in {check_func.__name__}: {e}")
                results['checks'][check_func.__name__.replace('_check_', '')] = {
                    'status': 'error',
                    'message': str(e),
                    'severity': 'high'
                }

        # Store results
        self._store_check_results(results)

        return results

    def _check_evaluation_metrics_quality(self) -> Dict[str, Any]:
        """Check evaluation metrics for quality issues with structured output."""
        try:
            # Load evaluation results
            eval_file = Path('data/metrics/evaluation_results_baseline_latest.csv')
            if not eval_file.exists():
                return {
                    'status': 'failed',
                    'message': 'Evaluation results file not found',
                    'severity': 'high'
                }

            df = pd.read_csv(eval_file)

            issues = []
            metrics_quality = {
                'mae': 'ok',
                'rmse': 'ok',
                'mape': 'ok',
                'smape': 'ok',
                'mase': 'ok',
                'directional_accuracy': 'ok'
            }

            # Check for MAPE issues (high values or unreliable flags)
            if 'mape_flag' in df.columns:
                unreliable_mape = df[df['mape_flag'] == 'unreliable']
                if len(unreliable_mape) > 0:
                    issues.append({
                        'type': 'mape_unreliable',
                        'count': len(unreliable_mape),
                        'description': f"{len(unreliable_mape)} symbols have unreliable MAPE"
                    })
                    metrics_quality['mape'] = 'unreliable'

                high_mape = df[df['mape_flag'] == 'high']
                if len(high_mape) > 0:
                    issues.append({
                        'type': 'mape_high',
                        'count': len(high_mape),
                        'description': f"{len(high_mape)} symbols have high MAPE"
                    })
                    if metrics_quality['mape'] == 'ok':
                        metrics_quality['mape'] = 'suspect'
            else:
                # Fallback to old logic if no flags
                mape_issues = df[df['mape'] >= self.thresholds['max_mape_value']]
                if len(mape_issues) > 0:
                    issues.append({
                        'type': 'mape_high',
                        'count': len(mape_issues),
                        'description': f"{len(mape_issues)} symbols have MAPE >= {self.thresholds['max_mape_value']}"
                    })
                    metrics_quality['mape'] = 'unreliable'

            # Check for identical values across symbols (per metric)
            for metric in ['mae', 'rmse', 'mape', 'smape', 'mase']:
                if metric in df.columns:
                    unique_values = len(df[metric].dropna().unique())
                    if unique_values < self.thresholds['min_unique_mae_values']:
                        issues.append({
                            'type': 'few_unique_values',
                            'metric': metric,
                            'unique_values': unique_values,
                            'description': f"Only {unique_values} unique {metric.upper()} values"
                        })
                        metrics_quality[metric] = 'suspect'

            # Check for all symbols having identical metrics per horizon
            for horizon in df['target_horizon'].unique():
                h_df = df[df['target_horizon'] == horizon]
                for metric in ['mae', 'rmse', 'mape', 'smape', 'mase']:
                    if metric in h_df.columns:
                        unique_vals = len(h_df[metric].dropna().unique())
                        if unique_vals == 1 and len(h_df) > 1:
                            issues.append({
                                'type': 'identical_metrics_per_horizon',
                                'horizon': horizon,
                                'metric': metric,
                                'description': f"Horizon {horizon}: All symbols have identical {metric.upper()}"
                            })
                            metrics_quality[metric] = 'unreliable'

            # Check MASE quality (should be scale-free benchmark)
            if 'mase' in df.columns:
                mase_values = df['mase'].dropna()
                if len(mase_values) > 0:
                    high_mase = mase_values[mase_values > 2.0]  # Much worse than naive
                    if len(high_mase) > len(mase_values) * 0.5:  # More than half
                        issues.append({
                            'type': 'mase_very_high',
                            'count': len(high_mase),
                            'description': f"{len(high_mase)} symbols have MASE > 2.0 (much worse than naive)"
                        })
                        metrics_quality['mase'] = 'suspect'

            # Check SMAPE quality
            smape_result = self._check_smape_quality(df)
            if smape_result['issues']:
                issues.extend(smape_result['issues'])
                if smape_result['severity'] == 'high':
                    metrics_quality['smape'] = 'unreliable'
                elif smape_result['severity'] == 'medium':
                    metrics_quality['smape'] = 'suspect'

            # Check SWASE quality
            swase_result = self._check_swase_quality(df)
            if swase_result['issues']:
                issues.extend(swase_result['issues'])
                if swase_result['severity'] == 'high':
                    metrics_quality['swase'] = 'unreliable'
                elif swase_result['severity'] == 'medium':
                    metrics_quality['swase'] = 'suspect'

            # Check cross-metric consistency
            consistency_result = self._check_cross_metric_consistency(df)
            if consistency_result['issues']:
                issues.extend(consistency_result['issues'])
                # Update affected metrics
                for metric in consistency_result['affected_metrics']:
                    if metrics_quality[metric] == 'ok':
                        metrics_quality[metric] = 'suspect'

            # Determine overall status and severity
            severity = 'low'
            status = 'passed'

            if any(issue['type'] in ['identical_metrics_per_horizon', 'mape_unreliable'] for issue in issues):
                severity = 'high'
                status = 'failed'
            elif any(issue['type'] in ['mape_high', 'few_unique_values'] for issue in issues):
                severity = 'medium'
                status = 'failed'
            elif len(issues) > 0:
                severity = 'low'
                status = 'failed'

            return {
                'status': status,
                'severity': severity,
                'issues': issues,
                'metrics_quality': metrics_quality,
                'message': f"Found {len(issues)} quality issues"
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error checking evaluation metrics: {e}',
                'severity': 'high'
            }

    def _check_smape_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check SMAPE metric quality and validity."""
        issues = []
        severity = 'low'

        if 'smape' not in df.columns:
            return {'issues': issues, 'severity': severity}

        smape_values = df['smape'].dropna()

        if len(smape_values) == 0:
            issues.append({
                'type': 'smape_missing',
                'description': 'No SMAPE values found'
            })
            severity = 'medium'
            return {'issues': issues, 'severity': severity}

        # Check range (SMAPE should be 0-2 for percentage, but can be higher)
        invalid_range = smape_values[(smape_values < 0) | (smape_values > 10)]
        if len(invalid_range) > 0:
            issues.append({
                'type': 'smape_invalid_range',
                'count': len(invalid_range),
                'description': f"{len(invalid_range)} SMAPE values outside valid range [0, 10]"
            })
            severity = 'high'

        # Check for unrealistic precision (all same decimal places)
        decimal_places = smape_values.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
        if len(decimal_places.unique()) == 1 and decimal_places.iloc[0] > 6:
            issues.append({
                'type': 'smape_suspicious_precision',
                'description': 'All SMAPE values have identical high precision (possible calculation error)'
            })
            severity = 'medium'

        # Check for NaN or infinite values
        nan_count = smape_values.isna().sum()
        inf_count = np.isinf(smape_values).sum()
        if nan_count > 0 or inf_count > 0:
            issues.append({
                'type': 'smape_invalid_values',
                'nan_count': nan_count,
                'inf_count': inf_count,
                'description': f"SMAPE contains {nan_count} NaN and {inf_count} infinite values"
            })
            severity = 'high'

        return {'issues': issues, 'severity': severity}

    def _check_swase_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check SWASE metric quality and validity."""
        issues = []
        severity = 'low'

        if 'swase' not in df.columns:
            return {'issues': issues, 'severity': severity}

        swase_values = df['swase'].dropna()

        if len(swase_values) == 0:
            issues.append({
                'type': 'swase_missing',
                'description': 'No SWASE values found'
            })
            severity = 'medium'
            return {'issues': issues, 'severity': severity}

        # Check range (SWASE should be non-negative)
        negative_values = swase_values[swase_values < 0]
        if len(negative_values) > 0:
            issues.append({
                'type': 'swase_negative',
                'count': len(negative_values),
                'description': f"{len(negative_values)} SWASE values are negative"
            })
            severity = 'high'

        # Check for extreme values (SWASE > 10 might indicate issues)
        extreme_values = swase_values[swase_values > 10]
        if len(extreme_values) > len(swase_values) * 0.1:  # More than 10%
            issues.append({
                'type': 'swase_extreme',
                'count': len(extreme_values),
                'description': f"{len(extreme_values)} SWASE values > 10 (extreme values)"
            })
            severity = 'medium'

        # Check for identical values across symbols (should vary with regime weighting)
        unique_swase = len(swase_values.unique())
        if unique_swase < len(swase_values) * 0.5:  # Less than 50% unique
            issues.append({
                'type': 'swase_low_variability',
                'unique_count': unique_swase,
                'total_count': len(swase_values),
                'description': f"Only {unique_swase}/{len(swase_values)} unique SWASE values (should vary with regime weighting)"
            })
            severity = 'medium'

        # Check for NaN or infinite values
        nan_count = swase_values.isna().sum()
        inf_count = np.isinf(swase_values).sum()
        if nan_count > 0 or inf_count > 0:
            issues.append({
                'type': 'swase_invalid_values',
                'nan_count': nan_count,
                'inf_count': inf_count,
                'description': f"SWASE contains {nan_count} NaN and {inf_count} infinite values"
            })
            severity = 'high'

        return {'issues': issues, 'severity': severity}

    def _check_cross_metric_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check consistency between related metrics."""
        issues = []
        affected_metrics = []

        # Check SMAPE vs MAPE relationship (SMAPE should be close to MAPE for reasonable ranges)
        if 'smape' in df.columns and 'mape' in df.columns:
            valid_rows = df[['smape', 'mape']].dropna()
            if len(valid_rows) > 0:
                # SMAPE and MAPE should be reasonably close for MAPE < 1
                reasonable_mape = valid_rows[valid_rows['mape'] < 1.0]
                if len(reasonable_mape) > 0:
                    ratio = reasonable_mape['smape'] / reasonable_mape['mape']
                    inconsistent = ratio[(ratio < 0.5) | (ratio > 2.0)]
                    if len(inconsistent) > len(reasonable_mape) * 0.2:  # More than 20%
                        issues.append({
                            'type': 'smape_mape_inconsistent',
                            'count': len(inconsistent),
                            'description': f"SMAPE/MAPE ratio inconsistent for {len(inconsistent)} symbols"
                        })
                        affected_metrics.extend(['smape', 'mape'])

        # Check SWASE vs SMAPE relationship (SWASE should be similar or higher than SMAPE)
        if 'swase' in df.columns and 'smape' in df.columns:
            valid_rows = df[['swase', 'smape']].dropna()
            if len(valid_rows) > 0:
                # SWASE should generally be >= SMAPE (due to shock weighting)
                inconsistent = valid_rows[valid_rows['swase'] < valid_rows['smape'] * 0.8]
                if len(inconsistent) > len(valid_rows) * 0.3:  # More than 30%
                    issues.append({
                        'type': 'swase_smape_inconsistent',
                        'count': len(inconsistent),
                        'description': f"SWASE unexpectedly lower than SMAPE for {len(inconsistent)} symbols"
                    })
                    affected_metrics.extend(['swase', 'smape'])

        # Check directional accuracy validity
        if 'directional_accuracy' in df.columns:
            dir_acc = df['directional_accuracy'].dropna()
            invalid_dir = dir_acc[(dir_acc < 0) | (dir_acc > 1)]
            if len(invalid_dir) > 0:
                issues.append({
                    'type': 'directional_accuracy_invalid',
                    'count': len(invalid_dir),
                    'description': f"{len(invalid_dir)} directional accuracy values outside [0,1] range"
                })
                affected_metrics.append('directional_accuracy')

        return {'issues': issues, 'affected_metrics': list(set(affected_metrics))}

    def _check_data_freshness(self) -> Dict[str, Any]:
        """Check if data is fresh and not stale."""
        try:
            # Check evaluation results timestamp
            eval_file = Path('data/metrics/evaluation_results_baseline_latest.csv')
            if eval_file.exists():
                file_age = datetime.now() - datetime.fromtimestamp(eval_file.stat().st_mtime)
                if file_age.total_seconds() > self.thresholds['max_staleness_hours'] * 3600:
                    return {
                        'status': 'failed',
                        'message': f'Evaluation results are {file_age.total_seconds()/3600:.1f} hours old',
                        'severity': 'medium'
                    }

            # Check forecast results
            results_dir = Path('results/hpo')
            if results_dir.exists():
                parquet_files = list(results_dir.glob('**/*.parquet'))
                if parquet_files:
                    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
                    file_age = datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)
                    if file_age.total_seconds() > self.thresholds['max_staleness_hours'] * 3600:
                        return {
                            'status': 'failed',
                            'message': f'Latest forecast results are {file_age.total_seconds()/3600:.1f} hours old',
                            'severity': 'medium'
                        }

            return {
                'status': 'passed',
                'message': 'Data freshness check passed'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error checking data freshness: {e}',
                'severity': 'medium'
            }

    def _check_pipeline_health(self) -> Dict[str, Any]:
        """Check if pipelines are running or stuck."""
        try:
            import psutil
            import os

            current_pid = os.getpid()
            python_processes = []

            # Get all Python processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        if proc.info['pid'] != current_pid:  # Exclude this quality check process
                            cmdline = proc.info.get('cmdline', [])
                            create_time = proc.info.get('create_time', 0)
                            runtime_hours = (time.time() - create_time) / 3600

                            python_processes.append({
                                'pid': proc.info['pid'],
                                'cmdline': ' '.join(cmdline) if cmdline else '',
                                'runtime_hours': runtime_hours,
                                'create_time': create_time
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Check for hanging processes
            hanging_processes = []
            long_running_processes = []

            for proc in python_processes:
                # Check for main.py processes that have been running too long
                if 'main.py' in proc['cmdline']:
                    if proc['runtime_hours'] > self.thresholds['max_pipeline_runtime'] / 3600:
                        hanging_processes.append({
                            'pid': proc['pid'],
                            'runtime_hours': round(proc['runtime_hours'], 1),
                            'cmdline': proc['cmdline'][:100] + '...' if len(proc['cmdline']) > 100 else proc['cmdline']
                        })
                    elif proc['runtime_hours'] > 1.0:  # Flag processes running > 1 hour
                        long_running_processes.append({
                            'pid': proc['pid'],
                            'runtime_hours': round(proc['runtime_hours'], 1),
                            'cmdline': proc['cmdline'][:100] + '...' if len(proc['cmdline']) > 100 else proc['cmdline']
                        })

            # Determine status and severity
            if hanging_processes:
                return {
                    'status': 'failed',
                    'message': f'Found {len(hanging_processes)} hanging pipeline processes exceeding {self.thresholds["max_pipeline_runtime"]/3600:.1f} hours',
                    'severity': 'critical',
                    'hanging_processes': hanging_processes,
                    'long_running_processes': long_running_processes
                }
            elif long_running_processes:
                return {
                    'status': 'warning',
                    'message': f'Found {len(long_running_processes)} long-running pipeline processes (>1 hour)',
                    'severity': 'medium',
                    'long_running_processes': long_running_processes
                }
            elif len(python_processes) > 3:  # More than 3 Python processes might indicate issues
                return {
                    'status': 'warning',
                    'message': f'{len(python_processes)} Python processes running - monitor for potential issues',
                    'severity': 'low',
                    'process_count': len(python_processes)
                }
            else:
                return {
                    'status': 'passed',
                    'message': 'Pipeline health check passed'
                }

        except ImportError:
            # Fallback to subprocess method if psutil not available
            return self._check_pipeline_health_fallback()
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error checking pipeline health: {e}',
                'severity': 'medium'
            }

    def _check_pipeline_health_fallback(self) -> Dict[str, Any]:
        """Fallback pipeline health check when psutil is not available."""
        try:
            # Check for running Python processes related to the system
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # More than just header
                    # Check if any have been running too long
                    # This is a simplified check - in production would parse more carefully
                    return {
                        'status': 'warning',
                        'message': f'{len(lines)-1} Python processes running - monitor for stuck pipelines',
                        'severity': 'low'
                    }

            return {
                'status': 'passed',
                'message': 'Pipeline health check passed'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error checking pipeline health: {e}',
                'severity': 'medium'
            }

    def _check_streamlit_status(self) -> Dict[str, Any]:
        """Check if Streamlit dashboard is running and accessible."""
        try:
            # Try to connect to Streamlit
            response = requests.get('http://localhost:8501', timeout=5)

            if response.status_code == 200:
                return {
                    'status': 'passed',
                    'message': 'Streamlit dashboard is running and accessible'
                }
            else:
                return {
                    'status': 'failed',
                    'message': f'Streamlit returned status {response.status_code}',
                    'severity': 'medium'
                }

        except requests.exceptions.RequestException:
            return {
                'status': 'failed',
                'message': 'Streamlit dashboard is not accessible',
                'severity': 'medium'
            }

    def _check_calculation_consistency(self) -> Dict[str, Any]:
        """Check for calculation consistency issues."""
        try:
            eval_file = Path('data/metrics/evaluation_results_baseline_latest.csv')
            if not eval_file.exists():
                return {'status': 'skipped', 'message': 'No evaluation file to check'}

            df = pd.read_csv(eval_file)

            issues = []

            # Check if MSE = MAE^2 (should be approximately true)
            df['mse_calculated'] = df['mae'] ** 2
            mse_diff = np.abs(df['mse'] - df['mse_calculated'])
            inconsistent_mse = (mse_diff > 0.001).sum()
            if inconsistent_mse > 0:
                issues.append(f"{inconsistent_mse} records have inconsistent MSE/MAE relationship")

            # Check for negative values where they shouldn't be
            for col in ['mae', 'rmse', 'mape']:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"{negative_count} negative {col.upper()} values found")

            # Check MAPE bounds (should be 0-1 for percentage, but can be >1)
            extreme_mape = (df['mape'] > 10).sum()
            if extreme_mape > 0:
                issues.append(f"{extreme_mape} extremely high MAPE values (>10)")

            if issues:
                return {
                    'status': 'failed',
                    'message': '; '.join(issues),
                    'severity': 'medium',
                    'details': issues
                }
            else:
                return {
                    'status': 'passed',
                    'message': 'Calculation consistency check passed'
                }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error checking calculation consistency: {e}',
                'severity': 'medium'
            }

    def _check_data_staleness(self) -> Dict[str, Any]:
        """Check for stale or test/debugging data."""
        try:
            # Check if evaluation file contains test data patterns
            eval_file = Path('data/metrics/evaluation_results_baseline_latest.csv')
            if eval_file.exists():
                df = pd.read_csv(eval_file)

                # Check for debugging patterns
                test_symbols = ['TEST_', 'DEBUG_', 'DUMMY_']
                test_count = 0
                for symbol in df['symbol'].unique():
                    if any(test in symbol.upper() for test in test_symbols):
                        test_count += 1

                if test_count > 0:
                    return {
                        'status': 'warning',
                        'message': f'Found {test_count} test/debug symbols in evaluation data',
                        'severity': 'low'
                    }

                # Check timestamp freshness
                if 'evaluation_timestamp' in df.columns:
                    timestamps = pd.to_datetime(df['evaluation_timestamp'])
                    age_hours = (datetime.now() - timestamps.max()).total_seconds() / 3600
                    if age_hours > self.thresholds['max_staleness_hours']:
                        return {
                            'status': 'failed',
                            'message': f'Evaluation data is {age_hours:.1f} hours old',
                            'severity': 'medium'
                        }

            return {
                'status': 'passed',
                'message': 'Data staleness check passed'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error checking data staleness: {e}',
                'severity': 'low'
            }

    def _log_check_result(self, check_type: str, result: Dict[str, Any]):
        """Log check result to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO quality_checks (check_type, status, details, severity)
                VALUES (?, ?, ?, ?)
            ''', (
                check_type,
                result.get('status', 'unknown'),
                json.dumps(result),
                result.get('severity', 'low')
            ))

    def _log_alert(self, alert_type: str, message: str, severity: str):
        """Log alert to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO alerts (alert_type, message, severity)
                VALUES (?, ?, ?)
            ''', (alert_type, message, severity))

        logger.warning(f"ALERT [{severity}]: {alert_type} - {message}")

    def _store_check_results(self, results: Dict[str, Any]):
        """Store comprehensive check results to database and JSON."""
        # Store key metrics
        failed_checks = sum(1 for check in results['checks'].values()
                          if check.get('status') == 'failed')

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO metrics_history (metric_name, value, check_type)
                VALUES (?, ?, ?)
            ''', ('failed_checks', failed_checks, 'summary'))

            conn.execute('''
                INSERT INTO metrics_history (metric_name, value, check_type)
                VALUES (?, ?, ?)
            ''', ('total_alerts', len(results['alerts']), 'summary'))

        # Also save full results to JSON file
        try:
            report_path = Path('data/metrics/quality_report_latest.json')
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Quality report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save quality report: {e}")

    def get_quality_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Generate a quality report for the specified time period.

        Args:
            hours_back: Hours to look back for the report

        Returns:
            Quality report dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        with sqlite3.connect(self.db_path) as conn:
            # Get recent checks
            checks_df = pd.read_sql_query('''
                SELECT * FROM quality_checks
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', conn, params=(cutoff_time.isoformat(),))

            # Get recent alerts
            alerts_df = pd.read_sql_query('''
                SELECT * FROM alerts
                WHERE timestamp >= ? AND resolved = FALSE
                ORDER BY timestamp DESC
            ''', conn, params=(cutoff_time.isoformat(),))

            # Get metrics summary
            metrics_df = pd.read_sql_query('''
                SELECT metric_name, AVG(value) as avg_value, COUNT(*) as count
                FROM metrics_history
                WHERE timestamp >= ?
                GROUP BY metric_name
            ''', conn, params=(cutoff_time.isoformat(),))

        return {
            'period_hours': hours_back,
            'generated_at': datetime.now().isoformat(),
            'checks_summary': {
                'total_checks': len(checks_df),
                'failed_checks': len(checks_df[checks_df['status'] == 'failed']),
                'error_checks': len(checks_df[checks_df['status'] == 'error'])
            },
            'active_alerts': len(alerts_df),
            'recent_alerts': alerts_df.head(10).to_dict('records'),
            'metrics_summary': metrics_df.to_dict('records'),
            'quality_score': self._calculate_quality_score(checks_df, alerts_df)
        }

    def _calculate_quality_score(self, checks_df: pd.DataFrame, alerts_df: pd.DataFrame) -> float:
        """Calculate overall quality score (0-100)."""
        if len(checks_df) == 0:
            return 0.0

        # Base score from check success rate
        success_rate = len(checks_df[checks_df['status'] == 'passed']) / len(checks_df)

        # Penalty for alerts
        alert_penalty = min(len(alerts_df) * 5, 30)  # Max 30 point penalty

        # Penalty for high severity issues
        severity_penalty = 0
        high_severity = checks_df[checks_df['severity'].isin(['high', 'critical'])]
        severity_penalty = len(high_severity) * 2

        score = (success_rate * 100) - alert_penalty - severity_penalty
        return max(0.0, min(100.0, score))

    def migrate_to_sqlite(self):
        """
        Migrate CSV-based evaluation results to SQLite for better performance.
        This addresses the user's concern about CSV storage.
        """
        try:
            # Create SQLite database for evaluation results
            eval_db_path = Path('data/evaluation_results.db')

            with sqlite3.connect(eval_db_path) as conn:
                # Create tables
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS evaluation_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        model_type TEXT,
                        target_horizon INTEGER,
                        experiment TEXT,
                        predictions_count INTEGER,
                        evaluation_timestamp TEXT,
                        mae REAL,
                        mse REAL,
                        rmse REAL,
                        mape REAL,
                        directional_accuracy REAL,
                        n_samples INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Migrate existing CSV data
                csv_file = Path('data/metrics/evaluation_results_baseline_latest.csv')
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    df.to_sql('evaluation_results', conn, if_exists='replace', index=False)

                    logger.info(f"Migrated {len(df)} evaluation records to SQLite")

                # Create indexes for performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol_horizon ON evaluation_results(symbol, target_horizon)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_model_type ON evaluation_results(model_type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluation_results(evaluation_timestamp)')

            return {
                'status': 'success',
                'message': f'Migrated evaluation data to SQLite database at {eval_db_path}',
                'records_migrated': len(df) if 'df' in locals() else 0
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to migrate to SQLite: {e}'
            }

    def run_metric_sanity_report(self) -> Dict[str, Any]:
        """
        Run the full metric sanity check, write JSON + Markdown reports,
        and return the JSON structure for further processing.
        """
        # Load evaluation results
        eval_file = Path('data/metrics/evaluation_results_baseline_latest.csv')
        if not eval_file.exists():
            raise FileNotFoundError(f"Evaluation results file not found: {eval_file}")

        df = pd.read_csv(eval_file)

        # Build core result structure
        base_check = self._check_evaluation_metrics_quality()

        result = {
            "run_metadata": self._build_run_metadata(df),
            "overall_status": {
                "status": base_check["status"],
                "severity": base_check["severity"],
                "issue_count": len(base_check.get("issues", [])),
                "summary": base_check["message"],
            },
            "metric_summaries": self._build_metric_summaries(df),
            "horizon_issues": self._build_horizon_issues(df),
            "symbol_examples": self._build_symbol_examples(df),
            "sanity_flags": self._build_sanity_flags(df),
        }

        # Create quality directory
        quality_dir = Path("results/quality")
        quality_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON
        json_path = quality_dir / "metric_sanity_latest.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # Write Markdown
        md_path = quality_dir / "metric_sanity_latest.md"
        markdown_text = self._render_metric_sanity_markdown(result)
        md_path.write_text(markdown_text, encoding="utf-8")

        return result

    def _build_run_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build run metadata for the sanity report."""
        now = datetime.now()
        return {
            "run_id": now.strftime("%Y-%m-%dT%H-%M-%SZ"),
            "evaluated_at": now.isoformat() + "Z",
            "source_file": "data/metrics/evaluation_results_baseline_latest.csv",
            "n_rows": int(len(df)),
            "symbols_covered": int(df["symbol"].nunique()) if "symbol" in df.columns else 0,
            "horizons_covered": sorted(df["target_horizon"].astype(str).unique().tolist())
            if "target_horizon" in df.columns else [],
        }

    def _build_metric_summaries(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build metric summaries with stats and issues."""
        summaries = {}

        metrics_to_check = ["mae", "mape", "smape", "swase", "directional_accuracy"]

        for metric in metrics_to_check:
            if metric not in df.columns:
                continue

            values = df[metric].dropna()
            if len(values) == 0:
                continue

            # Calculate stats
            stats = {
                "mean": round(float(values.mean()), 3),
                "std": round(float(values.std()), 3),
                "min": round(float(values.min()), 3),
                "max": round(float(values.max()), 3),
                "unique_values": int(values.nunique()),
                "issues": []
            }

            # Check for specific issues
            if metric == "mape":
                high_mape = (values >= 0.99).sum()
                if high_mape > 0:
                    stats["issues"].append(f"{high_mape} rows have MAPE >= 0.99 (likely stock splits or invalid denominators).")

            elif metric == "smape":
                nan_count = values.isna().sum()
                inf_count = np.isinf(values).sum()
                if nan_count > 0 or inf_count > 0:
                    stats["issues"].append(f"SMAPE has {nan_count} NaN and {inf_count} inf values.")

                suspicious = (values > 2.0).sum()
                if suspicious > 0:
                    stats["issues"].append(f"{suspicious} SMAPE values > 2.0 (suspicious).")

                if stats["unique_values"] < 10:
                    stats["issues"].append(f"Only {stats['unique_values']} unique SMAPE values across all symbols/horizons.")

            elif metric == "swase":
                nan_count = values.isna().sum()
                inf_count = np.isinf(values).sum()
                if nan_count > 0 or inf_count > 0:
                    stats["issues"].append(f"SWASE has {nan_count} NaN and {inf_count} inf values.")

                suspicious = (values > 3.0).sum()
                if suspicious > 0:
                    stats["issues"].append(f"{suspicious} SWASE values > 3.0 (suspicious).")

                if stats["unique_values"] < 10:
                    stats["issues"].append(f"Only {stats['unique_values']} unique SWASE values across all symbols/horizons.")

            summaries[metric] = stats

        return summaries

    def _build_horizon_issues(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify horizon-level issues like identical metrics."""
        issues = []

        if "target_horizon" not in df.columns:
            return issues

        for horizon in df["target_horizon"].unique():
            h_df = df[df["target_horizon"] == horizon]

            affected_metrics = []
            example_symbols = []

            # Check for identical metrics across symbols
            for metric in ["mape", "smape", "swase"]:
                if metric in h_df.columns:
                    unique_vals = h_df[metric].dropna().nunique()
                    if unique_vals == 1 and len(h_df) > 1:
                        affected_metrics.append(metric.upper())
                        if not example_symbols:
                            example_symbols = h_df["symbol"].head(3).tolist()

            if affected_metrics:
                issues.append({
                    "target_horizon": str(horizon),
                    "metrics": affected_metrics,
                    "description": f"All symbols share identical {', '.join(affected_metrics)} values; likely a per-horizon aggregation bug.",
                    "example_symbols": example_symbols
                })

        return issues

    def _build_symbol_examples(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build examples of problematic symbols."""
        examples = []

        if "symbol" not in df.columns or "target_horizon" not in df.columns:
            return examples

        # Find symbols with extreme values
        for _, row in df.iterrows():
            symbol = row.get("symbol")
            horizon = row.get("target_horizon")
            issues = []

            # Check MAPE
            mape = row.get("mape")
            if mape is not None and mape >= 0.99:
                issues.append(f"MAPE = {mape:.2f} (>= 0.99), likely division-by-zero or split.")

            # Check SWASE vs peers
            swase = row.get("swase")
            if swase is not None:
                peer_avg = df["swase"].mean()
                if swase > peer_avg * 2:
                    issues.append(f"SWASE = {swase:.2f} – far above peer average ({peer_avg:.2f}).")

            if issues:
                examples.append({
                    "symbol": symbol,
                    "target_horizon": str(horizon),
                    "issues": issues
                })

                # Limit to top examples
                if len(examples) >= 5:
                    break

        return examples

    def _build_sanity_flags(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Build boolean sanity flags."""
        flags = {
            "has_smape_column": "smape" in df.columns,
            "has_swase_column": "swase" in df.columns,
            "too_few_unique_smape_values": False,
            "too_few_unique_swase_values": False,
            "high_smape_share_ge_1_0": False,
            "high_swase_share_ge_3_0": False
        }

        if flags["has_smape_column"]:
            smape_values = df["smape"].dropna()
            flags["too_few_unique_smape_values"] = bool(len(smape_values.unique()) < 10)
            flags["high_smape_share_ge_1_0"] = bool((smape_values >= 1.0).mean() > 0.1)

        if flags["has_swase_column"]:
            swase_values = df["swase"].dropna()
            flags["too_few_unique_swase_values"] = bool(len(swase_values.unique()) < 10)
            flags["high_swase_share_ge_3_0"] = bool((swase_values >= 3.0).mean() > 0.1)

        return flags

    def _render_metric_sanity_markdown(self, result: Dict[str, Any]) -> str:
        """Render the sanity report as Markdown."""
        md = []

        # Header
        md.append("# Metric Sanity Report – Forecast Evaluation")
        md.append("")

        # Run metadata
        meta = result["run_metadata"]
        md.append(f"**Run ID:** {meta['run_id']}")
        md.append(f"**Evaluated at:** {meta['evaluated_at']}")
        md.append(f"**Source file:** `{meta['source_file']}`")
        md.append(f"**Rows:** {meta['n_rows']} – **Symbols:** {meta['symbols_covered']} – **Horizons:** {', '.join(meta['horizons_covered'])}")
        md.append("")

        # Overall status
        status = result["overall_status"]
        status_icon = "❌ FAILED" if status["status"] == "failed" else "✅ PASSED"
        severity_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(status["severity"], "⚪")
        md.append("---")
        md.append("")
        md.append("## Overall Status")
        md.append(f"- **Status:** {status_icon}")
        md.append(f"- **Severity:** {severity_icon} {status['severity'].title()}")
        md.append(f"- **Issue count:** {status['issue_count']}")
        md.append("")
        md.append("**Summary:**")
        md.append(f"{status['summary']}")
        md.append("")

        # Metric summaries
        md.append("---")
        md.append("")
        md.append("## Metric Summaries")
        md.append("")

        for metric_name, summary in result["metric_summaries"].items():
            md.append(f"### {metric_name.upper()}")
            md.append(f"- Mean: `{summary['mean']}`")
            md.append(f"- Std: `{summary['std']}`")
            md.append(f"- Min / Max: `{summary['min']}` / `{summary['max']}`")
            md.append(f"- Unique values: `{summary['unique_values']}`")

            if summary["issues"]:
                md.append("- Issues:")
                for issue in summary["issues"]:
                    md.append(f"  - {issue}")
            else:
                md.append("- Issues: _None detected_")

            md.append("")

        # Horizon issues
        if result["horizon_issues"]:
            md.append("---")
            md.append("")
            md.append("## Horizon-Level Issues")
            md.append("")

            for issue in result["horizon_issues"]:
                md.append(f"- **Horizon {issue['target_horizon']}**")
                md.append(f"  - Metrics affected: {', '.join(issue['metrics'])}")
                md.append(f"  - Description: {issue['description']}")
                md.append(f"  - Example symbols: {', '.join(issue['example_symbols'])}")
                md.append("")

        # Symbol examples
        if result["symbol_examples"]:
            md.append("---")
            md.append("")
            md.append("## Symbol Examples")
            md.append("")

            for example in result["symbol_examples"]:
                md.append(f"- **{example['symbol']} @ Horizon {example['target_horizon']}**")
                for issue in example["issues"]:
                    md.append(f"  - {issue}")
                md.append("")

        # Recommended follow-up
        md.append("---")
        md.append("")
        md.append("## Recommended Follow-Up")
        md.append("")
        md.append("1. **SMAPE / SWASE implementation**")
        md.append("   - Review the SMAPE formula and ensure the denominator uses `(|pred| + |actual| + eps)`.")
        md.append("   - Confirm SWASE uses scaled errors with correct shock weights.")
        md.append("2. **Per-horizon loops**")
        md.append("   - Verify that metrics are computed per `(symbol, horizon)` and not accidentally broadcast or re-used.")
        md.append("3. **Evaluation dataset**")
        md.append("   - Inspect raw rows where SMAPE > 2.0 or SWASE > 3.0.")
        md.append("   - Check for zero or tiny actuals, or missing/incorrect shock flags.")
        md.append("")
        md.append("---")

        return "\n".join(md)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Quality Assurance Agent')
    parser.add_argument('--check', action='store_true', help='Run one-time quality check')
    parser.add_argument('--monitor', action='store_true', help='Start continuous monitoring')
    parser.add_argument('--report', action='store_true', help='Generate quality report')
    parser.add_argument('--migrate', action='store_true', help='Migrate CSV data to SQLite')
    parser.add_argument('--sanity-report', action='store_true', help='Generate metric sanity report (JSON + Markdown)')
    parser.add_argument('--hours', type=int, default=24, help='Hours for report (default: 24)')

    args = parser.parse_args()

    agent = QualityAssuranceAgent()

    if args.check:
        results = agent.run_full_quality_check()
        print(json.dumps(results, indent=2))

    elif args.monitor:
        print("Starting continuous quality monitoring...")
        agent.start_monitoring()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            agent.stop_monitoring()
            print("Monitoring stopped.")

    elif args.report:
        report = agent.get_quality_report(args.hours)
        print(json.dumps(report, indent=2))

    elif args.migrate:
        result = agent.migrate_to_sqlite()
        print(json.dumps(result, indent=2))

    elif args.sanity_report:
        try:
            result = agent.run_metric_sanity_report()
            print("✅ Metric sanity report generated successfully!")
            print(f"📄 JSON report: results/quality/metric_sanity_latest.json")
            print(f"📝 Markdown report: results/quality/metric_sanity_latest.md")
            print(f"📊 Overall status: {result['overall_status']['status']} ({result['overall_status']['severity']} severity)")
            print(f"🔍 Issues found: {result['overall_status']['issue_count']}")
        except Exception as e:
            print(f"❌ Failed to generate metric sanity report: {e}")
            import traceback
            traceback.print_exc()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()