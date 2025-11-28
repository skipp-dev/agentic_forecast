"""
Drift Report Service

Provides comprehensive drift detection and reporting using statistical methods
and data quality analysis. Can integrate with Evidently or Deepchecks for
advanced drift detection.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime, timedelta
import json
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

class DriftReportService:
    """
    Service for generating comprehensive drift reports and data quality analysis.

    Provides:
    - Data drift detection (feature distributions)
    - Performance drift analysis
    - Statistical tests (PSI, KS, etc.)
    - HTML and JSON report generation
    """

    def __init__(self, reports_dir: str = "reports/drift"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        logger.info(f"DriftReportService initialized with reports dir: {reports_dir}")

    def generate_data_drift_report(self, reference_df: pd.DataFrame,
                                 current_df: pd.DataFrame,
                                 features: List[str] = None) -> Dict[str, Any]:
        """
        Generate data drift report comparing reference and current feature distributions.

        Args:
            reference_df: Reference/baseline data
            current_df: Current data to compare
            features: List of features to analyze (if None, uses all numeric columns)

        Returns:
            Dictionary containing drift analysis results
        """
        if features is None:
            # Use numeric columns as features
            features = reference_df.select_dtypes(include=[np.number]).columns.tolist()

        report = {
            "report_type": "data_drift",
            "timestamp": datetime.now().isoformat(),
            "features_analyzed": features,
            "drift_summary": {},
            "feature_details": {},
            "recommendations": []
        }

        total_drift_score = 0.0
        drifted_features = []

        for feature in features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                logger.warning(f"Feature {feature} not found in both datasets")
                continue

            ref_data = reference_df[feature].dropna()
            curr_data = current_df[feature].dropna()

            if len(ref_data) == 0 or len(curr_data) == 0:
                logger.warning(f"Insufficient data for feature {feature}")
                continue

            # Calculate drift metrics
            feature_report = self._analyze_feature_drift(ref_data, curr_data, feature)
            report["feature_details"][feature] = feature_report

            # Simple drift detection based on distribution change
            drift_score = feature_report.get("drift_score", 0.0)
            report["drift_summary"][feature] = {
                "drift_score": drift_score,
                "is_drifted": drift_score > 0.1  # Threshold for significant drift
            }

            total_drift_score += drift_score
            if drift_score > 0.1:
                drifted_features.append(feature)

        # Overall assessment
        avg_drift_score = total_drift_score / len(features) if features else 0.0
        report["overall_drift_score"] = avg_drift_score
        report["drifted_features"] = drifted_features
        report["drift_severity"] = self._classify_drift_severity(avg_drift_score, len(drifted_features))

        # Generate recommendations
        report["recommendations"] = self._generate_drift_recommendations(report)

        # Save report
        report_path = self._save_report(report, "data_drift")
        report["report_path"] = report_path

        logger.info(f"Data drift report generated: {len(drifted_features)} drifted features, severity: {report['drift_severity']}")
        return report

    def generate_performance_drift_report(self, reference_metrics: Dict[str, List[float]],
                                        current_metrics: Dict[str, List[float]],
                                        horizons: List[int] = None) -> Dict[str, Any]:
        """
        Generate performance drift report comparing reference and current model metrics.

        Args:
            reference_metrics: Historical metrics (e.g., {'mape': [0.05, 0.04, ...]})
            current_metrics: Current metrics
            horizons: Forecast horizons to analyze

        Returns:
            Dictionary containing performance drift analysis
        """
        if horizons is None:
            horizons = [1, 3, 5, 10, 20]

        report = {
            "report_type": "performance_drift",
            "timestamp": datetime.now().isoformat(),
            "horizons_analyzed": horizons,
            "metric_analysis": {},
            "drift_summary": {},
            "recommendations": []
        }

        total_performance_change = 0.0
        degraded_horizons = []

        for horizon in horizons:
            horizon_key = f"{horizon}d"
            report["metric_analysis"][horizon_key] = {}

            for metric_name in reference_metrics.keys():
                if metric_name not in current_metrics:
                    continue

                ref_values = np.array(reference_metrics[metric_name])
                curr_values = np.array(current_metrics[metric_name])

                if len(ref_values) == 0 or len(curr_values) == 0:
                    continue

                # Calculate performance change
                ref_mean = np.mean(ref_values)
                curr_mean = np.mean(curr_values)

                if ref_mean == 0:
                    change_pct = 0.0
                else:
                    change_pct = (curr_mean - ref_mean) / abs(ref_mean)

                # Statistical significance test
                try:
                    _, p_value = stats.ttest_ind(ref_values, curr_values)
                    is_significant = p_value < 0.05
                except:
                    is_significant = False
                    p_value = 1.0

                report["metric_analysis"][horizon_key][metric_name] = {
                    "reference_mean": ref_mean,
                    "current_mean": curr_mean,
                    "change_percentage": change_pct,
                    "p_value": p_value,
                    "is_significant": is_significant,
                    "performance_degraded": change_pct > 0.1  # For error metrics, increase is bad
                }

                # For MAPE/MAE, positive change indicates degradation
                if metric_name.lower() in ['mape', 'mae', 'rmse']:
                    total_performance_change += abs(change_pct)
                    if change_pct > 0.1:  # Significant degradation
                        degraded_horizons.append(horizon_key)

        # Overall assessment
        avg_performance_change = total_performance_change / len(horizons) if horizons else 0.0
        report["overall_performance_change"] = avg_performance_change
        report["degraded_horizons"] = degraded_horizons
        report["performance_severity"] = self._classify_performance_severity(avg_performance_change, len(degraded_horizons))

        # Generate recommendations
        report["recommendations"] = self._generate_performance_recommendations(report)

        # Save report
        report_path = self._save_report(report, "performance_drift")
        report["report_path"] = report_path

        logger.info(f"Performance drift report generated: {len(degraded_horizons)} degraded horizons, severity: {report['performance_severity']}")
        return report

    def _analyze_feature_drift(self, ref_data: pd.Series, curr_data: pd.Series, feature_name: str) -> Dict[str, Any]:
        """Analyze drift for a single feature"""
        analysis = {
            "feature_name": feature_name,
            "reference_stats": {},
            "current_stats": {},
            "drift_metrics": {}
        }

        # Basic statistics
        for label, data in [("reference", ref_data), ("current", curr_data)]:
            analysis[f"{label}_stats"] = {
                "mean": float(data.mean()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "skew": float(data.skew()),
                "kurtosis": float(data.kurtosis()),
                "n_samples": len(data)
            }

        # Population Stability Index (PSI)
        psi_score = self._calculate_psi(ref_data, curr_data)
        analysis["drift_metrics"]["psi"] = psi_score

        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_p_value = stats.ks_2samp(ref_data, curr_data)
            analysis["drift_metrics"]["ks_statistic"] = ks_stat
            analysis["drift_metrics"]["ks_p_value"] = ks_p_value
        except Exception as e:
            analysis["drift_metrics"]["ks_error"] = str(e)

        # Simple drift score (combination of PSI and KS)
        drift_score = psi_score
        if "ks_statistic" in analysis["drift_metrics"]:
            drift_score += analysis["drift_metrics"]["ks_statistic"]

        analysis["drift_score"] = drift_score

        return analysis

    def _calculate_psi(self, ref_data: pd.Series, curr_data: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins
            combined = pd.concat([ref_data, curr_data])
            bin_edges = pd.qcut(combined, q=bins, duplicates='drop', retbins=True)[1]

            # Calculate distributions
            ref_hist, _ = np.histogram(ref_data, bins=bin_edges)
            curr_hist, _ = np.histogram(curr_data, bins=bin_edges)

            # Convert to proportions
            ref_prop = ref_hist / len(ref_data)
            curr_prop = curr_hist / len(curr_data)

            # Avoid division by zero
            ref_prop = np.where(ref_prop == 0, 1e-10, ref_prop)
            curr_prop = np.where(curr_prop == 0, 1e-10, curr_prop)

            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))

            return float(psi)

        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0

    def _classify_drift_severity(self, avg_drift_score: float, num_drifted_features: int) -> str:
        """Classify overall drift severity"""
        if avg_drift_score > 0.25 or num_drifted_features > 5:
            return "severe"
        elif avg_drift_score > 0.15 or num_drifted_features > 2:
            return "moderate"
        elif avg_drift_score > 0.05 or num_drifted_features > 0:
            return "mild"
        else:
            return "none"

    def _classify_performance_severity(self, avg_change: float, num_degraded: int) -> str:
        """Classify performance degradation severity"""
        if avg_change > 0.3 or num_degraded > 3:
            return "severe"
        elif avg_change > 0.2 or num_degraded > 1:
            return "moderate"
        elif avg_change > 0.1 or num_degraded > 0:
            return "mild"
        else:
            return "none"

    def _generate_drift_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on drift analysis"""
        recommendations = []

        severity = report.get("drift_severity", "none")
        drifted_features = report.get("drifted_features", [])

        if severity == "severe":
            recommendations.append("URGENT: High data drift detected. Consider immediate model retraining.")
            recommendations.append("Investigate root cause of drift in features: " + ", ".join(drifted_features[:3]))
        elif severity == "moderate":
            recommendations.append("Moderate data drift detected. Schedule model retraining within 1-2 days.")
            if drifted_features:
                recommendations.append("Monitor features: " + ", ".join(drifted_features[:3]))
        elif severity == "mild":
            recommendations.append("Mild data drift detected. Continue monitoring and consider retraining in next cycle.")
        else:
            recommendations.append("No significant data drift detected. Continue normal operations.")

        return recommendations

    def _generate_performance_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance analysis"""
        recommendations = []

        severity = report.get("performance_severity", "none")
        degraded_horizons = report.get("degraded_horizons", [])

        if severity == "severe":
            recommendations.append("URGENT: Severe performance degradation detected. Immediate model retraining required.")
            recommendations.append("Consider switching to different model family (e.g., TFT → NHITS).")
        elif severity == "moderate":
            recommendations.append("Moderate performance degradation detected. Schedule HPO and retraining.")
            if degraded_horizons:
                recommendations.append("Focus on degraded horizons: " + ", ".join(degraded_horizons))
        elif severity == "mild":
            recommendations.append("Mild performance degradation detected. Monitor closely and retrain if trend continues.")
        else:
            recommendations.append("Model performance stable. Continue normal operations.")

        return recommendations

    def _save_report(self, report: Dict[str, Any], report_type: str) -> str:
        """Save report to disk in both JSON and HTML formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{report_type}_report_{timestamp}"

        # Save JSON
        json_path = os.path.join(self.reports_dir, f"{base_name}.json")
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Save HTML (simplified version)
        html_path = os.path.join(self.reports_dir, f"{base_name}.html")
        self._generate_html_report(report, html_path)

        return json_path

    def _generate_html_report(self, report: Dict[str, Any], html_path: str):
        """Generate a simple HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Drift Report - {report.get('report_type', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 5px; margin: 5px 0; }}
                .severe {{ color: red; }}
                .moderate {{ color: orange; }}
                .mild {{ color: yellow; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.get('report_type', 'Unknown').replace('_', ' ').title()} Report</h1>
                <p>Generated: {report.get('timestamp', 'Unknown')}</p>
            </div>

            <div class="section">
                <h2>Summary</h2>
                <div class="metric">Severity: <span class="{report.get('drift_severity', report.get('performance_severity', 'none'))}">{report.get('drift_severity', report.get('performance_severity', 'none')).upper()}</span></div>
                <div class="metric">Drifted Features: {len(report.get('drifted_features', report.get('degraded_horizons', [])))}</div>
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {"".join(f"<li>{rec}</li>" for rec in report.get('recommendations', []))}
                </ul>
            </div>
        </body>
        </html>
        """

        with open(html_path, 'w') as f:
            f.write(html_content)
