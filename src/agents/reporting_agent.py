from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from html import escape
import markdown
import numpy as np
from langsmith import traceable
from src.utils.llm_utils import extract_json_from_llm_output
from src.guardrails.status import compute_guardrail_status
from src.reporting.postprocess import normalize_system_report
from src.reporting.validators import validate_report_consistency, ReportConsistencyError

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

@dataclass
class ReportingInput:
    analytics_summary: Dict[str, Any]
    hpo_plan: Dict[str, Any]
    research_insights: Dict[str, Any]
    guardrail_status: Dict[str, Any]
    run_metadata: Dict[str, Any]

@dataclass
class ReportSection:
    title: str
    audience: str   # "quants" | "ops" | "management" | "mixed"
    body_markdown: str

@dataclass
class SystemReport:
    # === New schema (matches current LLM prompt) ===
    executive_summary: str
    sections: List[ReportSection] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    key_opportunities: List[str] = field(default_factory=list)
    actions_for_quants: List[str] = field(default_factory=list)
    actions_for_ops: List[str] = field(default_factory=list)

    # === Legacy fields (kept for compatibility; usually empty) ===
    performance_overview: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    optimization_recommendations: Dict[str, Any] = field(default_factory=dict)
    research_insights: Dict[str, Any] = field(default_factory=dict)
    operational_notes: Dict[str, Any] = field(default_factory=dict)
    priority_actions: List[str] = field(default_factory=list)

@dataclass
class FullReport:
    report_id: str
    timestamp: str
    run_type: str
    metrics_overview: Dict[str, Any]
    system_report: SystemReport

def render_html_report(report: FullReport) -> str:
    """
    Render a FullReport dataclass (report_id, timestamp, run_type,
    metrics_overview, system_report) into an HTML string.
    """

    metrics = report.metrics_overview or {}
    sys_report = report.system_report

    # Convenience helpers
    def _fmt_pct(x, default="–"):
        try:
            return f"{float(x) * 100:.2f}%"
        except Exception:
            return default

    def _fmt_float(x, default="–", digits=4):
        try:
            return f"{float(x):.{digits}f}"
        except Exception:
            return default

    # Top-level metrics
    total_symbols = metrics.get("total_symbols", "–")
    models_trained = metrics.get("models_trained", "–")
    models_promoted = metrics.get("models_promoted", "–")
    avg_mape = _fmt_pct(metrics.get("avg_mape", None))
    median_mape = _fmt_pct(metrics.get("median_mape", None))
    num_anomalies = metrics.get("num_anomalies", "–")

    guardrails = metrics.get("guardrails", {}) or {}
    guard_total = guardrails.get("total_checks", "–")
    guard_passed = guardrails.get("passed", "–")
    guard_warn = guardrails.get("warnings", "–")
    guard_crit = guardrails.get("critical", "–")

    risk_events: List[dict] = metrics.get("risk_events", []) or []

    # Build Risk Events HTML
    risk_events_html_parts: List[str] = []
    for ev in risk_events:
        ev_type = escape(str(ev.get("type", "unknown")))
        reason = escape(str(ev.get("reason", "")))
        vol = ev.get("portfolio_vol_annualized", None)
        lim = ev.get("limit_annualized", None)

        vol_str = _fmt_pct(vol) if vol is not None else "–"
        lim_str = _fmt_pct(lim) if lim is not None else "–"

        extra_kv = []
        for k, v in ev.items():
            if k in {"type", "reason", "portfolio_vol_annualized", "limit_annualized"}:
                continue
            extra_kv.append(f"{escape(str(k))}: {escape(str(v))}")
        extra_html = "<br>".join(extra_kv) if extra_kv else ""

        block = f"""
        <div class="risk-event">
            <strong>Type:</strong> {ev_type}<br>
            <strong>Reason:</strong> {reason or "–"}<br>
            <strong>Annualized Volatility:</strong> {vol_str}<br>
            <strong>Limit:</strong> {lim_str}<br>
            {extra_html}
        </div>
        """
        risk_events_html_parts.append(block)

    risk_events_html = "\n".join(risk_events_html_parts) if risk_events_html_parts else """
        <p class="muted">No risk events recorded for this run.</p>
    """

    # Build sections HTML (NOW with markdown rendering)
    sections_html_parts: List[str] = []
    for section in sys_report.sections:
        title = escape(section.title)
        audience = escape(section.audience or "mixed")

        # Render Markdown to HTML safely
        raw_markdown = section.body_markdown or ""
        body_html = markdown.markdown(
            raw_markdown,
            extensions=["extra", "sane_lists"]
        )

        section_html = f"""
        <div class="section-box">
            <h3>{title}</h3>
            <p class="audience-tag"><em>Audience: {audience}</em></p>
            <div class="section-body">
                {body_html}
            </div>
        </div>
        """
        sections_html_parts.append(section_html)

    sections_html = "\n".join(sections_html_parts) if sections_html_parts else """
        <p class="muted">No detailed sections provided.</p>
    """

    # Key risks / opportunities / actions
    def _build_list_cards(items: List[str], css_class: str) -> str:
        if not items:
            return '<p class="muted">None recorded.</p>'
        return "\n".join(
            f'<div class="{css_class}">{escape(item)}</div>'
            for item in items
        )

    key_risks_html = _build_list_cards(sys_report.key_risks, "risk-item")
    key_opps_html = _build_list_cards(sys_report.key_opportunities, "opp-item")
    actions_quants_html = _build_list_cards(sys_report.actions_for_quants, "action-item")
    actions_ops_html = _build_list_cards(sys_report.actions_for_ops, "action-item")

    # Executive summary (still plain text)
    exec_summary = escape(sys_report.executive_summary or "Executive summary missing.")

    # Basic metadata
    report_id = escape(report.report_id)
    timestamp = escape(report.timestamp)
    run_type = escape(report.run_type)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>System Forecasting Report – {report_id}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 10px;
            margin-bottom: 5px;
        }}
        .timestamp {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .run-type-badge {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.8em;
            margin-left: 10px;
        }}
        .executive-summary {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        .metric-card {{
            background: #ffffff;
            border: 1px solid #ddd;
            padding: 10px 12px;
            border-radius: 5px;
        }}
        .metric-label {{
            font-size: 0.85em;
            color: #6c757d;
        }}
        .metric-value {{
            font-size: 1.1em;
            font-weight: 600;
            margin-top: 2px;
        }}
        .section-box {{
            background: #fff;
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .section-body {{
            margin-top: 10px;
        }}
        .section-body ul {{
            margin-top: 5px;
            margin-bottom: 5px;
        }}
        .section-body li {{
            margin-bottom: 3px;
        }}
        .audience-tag {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .risk-item {{
            background: #fff3cd;
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
            border-left: 4px solid #ffc107;
        }}
        .opp-item {{
            background: #d4edda;
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
            border-left: 4px solid #28a745;
        }}
        .action-item {{
            background: #d1ecf1;
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
            border-left: 4px solid #17a2b8;
        }}
        .risk-event {{
            background: #f8d7da;
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
            border-left: 4px solid #dc3545;
        }}
        .muted {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .guardrail-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .guardrail-table th,
        .guardrail-table td {{
            border: 1px solid #ddd;
            padding: 6px 8px;
            text-align: left;
        }}
        .guardrail-table th {{
            background: #f1f3f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>
            System Forecasting Report
            <span class="run-type-badge">{run_type}</span>
        </h1>
        <p class="timestamp">
            Report ID: {report_id} · Generated: {timestamp}
        </p>

        <h2>Executive Summary</h2>
        <div class="executive-summary">
            <p>{exec_summary}</p>
        </div>

        <h2>Metrics Overview</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Symbols</div>
                <div class="metric-value">{total_symbols}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Models Trained</div>
                <div class="metric-value">{models_trained}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Models Promoted</div>
                <div class="metric-value">{models_promoted}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg MAPE</div>
                <div class="metric-value">{avg_mape}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Median MAPE</div>
                <div class="metric-value">{median_mape}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Anomalies Detected</div>
                <div class="metric-value">{num_anomalies}</div>
            </div>
        </div>

        <h2>Guardrails & Health</h2>
        <table class="guardrail-table">
            <tr>
                <th>Total Checks</th>
                <th>Passed</th>
                <th>Warnings</th>
                <th>Critical</th>
            </tr>
            <tr>
                <td>{guard_total}</td>
                <td>{guard_passed}</td>
                <td>{guard_warn}</td>
                <td>{guard_crit}</td>
            </tr>
        </table>

        <h2>Risk Events</h2>
        {risk_events_html}

        <h2>Detailed Sections</h2>
        {sections_html}

        <h2>Key Risks</h2>
        {key_risks_html}

        <h2>Key Opportunities</h2>
        {key_opps_html}

        <h2>Actions for Quants</h2>
        {actions_quants_html}

        <h2>Actions for Ops</h2>
        {actions_ops_html}
    </div>
</body>
</html>
"""
    return html

class LLMReportingAgent:
    """
    End-to-end reporting agent that generates, stores, and distributes human-readable reports.

    Features:
    - Generates comprehensive reports from system outputs
    - Stores raw JSON and rendered formats (Markdown/HTML)
    - Sends email notifications
    - Provides web dashboard integration
    - Includes Prometheus metrics for monitoring
    """

    def __init__(self, llm_client=None, settings=None):
        # Use the new LLM factory if no client provided
        if llm_client is None:
            from src.llm.llm_agent_factory import create_llm_agent
            # This will create the agent using the factory, but we need the LLM client
            # Let's use the factory's message building instead
            from src.llm.llm_agent_factory import get_llm_agent_factory
            self.factory = get_llm_agent_factory()
            from src.llm.llm_factory import create_reporting_agent_llm
            self.llm = create_reporting_agent_llm()
        else:
            self.llm = llm_client
            self.factory = None

        self.settings = settings or {}

        # Storage configuration
        self.reports_dir = Path(self.settings.get('reports_dir', 'results/reports'))
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Email configuration
        self.email_enabled = self.settings.get('email_enabled', False)
        self.email_recipients = self.settings.get('email_recipients', [])

        # Web dashboard configuration
        self.dashboard_enabled = self.settings.get('dashboard_enabled', False)
        self.dashboard_dir = Path(self.settings.get('dashboard_dir', 'web_dashboard'))

        # Initialize services
        self._init_services()

        # Store last report for continuous learning
        self._last_report = None

    def _init_services(self):
        """Initialize optional services with error handling."""
        # Initialize email service
        try:
            from src.services.email_service import get_email_service
            self.email_service = get_email_service()
            self.email_available = True
        except ImportError:
            self.email_service = None
            self.email_available = False
            print("Warning: Email service not available")

        # Initialize metrics service
        try:
            from src.services.metrics_service import get_metrics_service
            self.metrics_service = get_metrics_service()
            self.metrics_available = True
        except ImportError:
            self.metrics_service = None
            self.metrics_available = False
            print("Warning: Metrics service not available")

    def _build_metrics_overview(self, inputs: ReportingInput) -> Dict[str, Any]:
        """
        Construct the metrics_overview object PURELY from Python-side metrics.
        This ensures the LLM cannot hallucinate numbers.
        """
        # Extract from analytics_summary
        analytics = inputs.analytics_summary or {}
        # Extract from guardrail_status
        guardrails = inputs.guardrail_status or {}
        # Extract from hpo_plan
        hpo = inputs.hpo_plan or {}
        
        # Try to get evaluation metrics if available (from _enhance_input_with_evaluation_metrics)
        eval_metrics = analytics.get('evaluation_metrics', {})
        eval_results = eval_metrics.get('evaluation_results', {})
        model_comparison = eval_metrics.get('model_comparison', {})
        
        # Calculate promoted models count
        promoted_count = len(model_comparison.get('promotions', [])) if model_comparison else 0
        
        # Handle critical_issues which might be a list of issues or a count
        critical_val = guardrails.get('critical_issues', 0)
        if isinstance(critical_val, list):
            critical_count = len(critical_val)
        else:
            critical_count = int(critical_val)

        # Build the overview
        metrics_overview = {
            "total_symbols": eval_results.get('total_models', 0),
            "models_trained": eval_results.get('total_models', 0),
            "models_promoted": promoted_count,
            "avg_mape": eval_results.get('avg_mape', 0.0),
            "median_mape": eval_results.get('median_mape', 0.0),
            "num_anomalies": analytics.get('anomalies_detected', 0),
            
            "hpo": {
                "run_type": inputs.run_metadata.get('run_type', 'WEEKEND_HPO'),
                "total_trials": hpo.get('total_trials', 0),
                "families": hpo.get('per_family_search_spaces', {})
            },
            
            "guardrails": {
                "status": compute_guardrail_status(
                    total_checks=guardrails.get('total_checks', 0),
                    passed=guardrails.get('passed_checks', 0),
                    warnings=guardrails.get('warnings', 0),
                    critical=critical_count,
                    engine_errors=0  # We assume 0 for now unless we track it
                ),
                "total_checks": guardrails.get('total_checks', 0),
                "passed": guardrails.get('passed_checks', 0),
                "warnings": guardrails.get('warnings', 0),
                "critical": critical_count
            },
            
            "risk_events": guardrails.get('risk_events', []) or [],
            
            # Add model comparison for LLM context
            "model_comparison": model_comparison
        }
        
        # If we have a blocked portfolio event in guardrails, ensure it's in risk_events
        if guardrails.get('portfolio_blocked', False):
            # Check if already in risk_events
            has_block = any(e.get('type') == 'portfolio_blocked' for e in metrics_overview['risk_events'])
            if not has_block:
                metrics_overview['risk_events'].append({
                    "type": "portfolio_blocked",
                    "reason": guardrails.get('block_reason', "volatility_limit"),
                    "portfolio_vol_annualized": guardrails.get('portfolio_vol', 0.0),
                    "limit_annualized": guardrails.get('vol_limit', 0.0)
                })
                
        return metrics_overview

    def _enhance_input_with_evaluation_metrics(self, report_input: ReportingInput) -> ReportingInput:
        """
        Enhance the reporting input with evaluation metrics from the enhanced metrics system.
        """
        enhanced_analytics = dict(report_input.analytics_summary)
        enhanced_hpo = dict(report_input.hpo_plan)
        enhanced_research = dict(report_input.research_insights)

        try:
            # Load evaluation results from the metrics system
            evaluation_metrics = self._load_evaluation_metrics()

            if evaluation_metrics:
                # Enhance analytics summary with evaluation metrics
                enhanced_analytics.update({
                    'evaluation_metrics': evaluation_metrics,
                    'performance_summary': self._summarize_evaluation_metrics(evaluation_metrics)
                })

                # Add evaluation insights to research
                enhanced_research.update({
                    'evaluation_insights': self._extract_evaluation_insights(evaluation_metrics),
                    'model_comparison': evaluation_metrics.get('model_comparison', {}),
                    'risk_assessment': evaluation_metrics.get('risk_assessment', {})
                })

        except Exception as e:
            logger.warning(f"Failed to enhance input with evaluation metrics: {e}")

        return ReportingInput(
            analytics_summary=enhanced_analytics,
            hpo_plan=enhanced_hpo,
            research_insights=enhanced_research,
            guardrail_status=report_input.guardrail_status,
            run_metadata=report_input.run_metadata
        )

    def _load_evaluation_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Load evaluation metrics from the enhanced metrics system and metrics exporter.
        """
        try:
            # Try to load from metrics exporter first
            try:
                from src.services.metrics_exporter import _load_json_safely
            except ImportError:
                 # Fallback if not found in src
                 def _load_json_safely(path): return {}

            # Load quality and forecast reports
            quality_report = _load_json_safely("data/metrics/quality_report_latest.json")
            forecast_output = _load_json_safely("data/metrics/forecast_agent_output_latest.json")

            # Load evaluation results (CSV)
            # We look for all evaluation_results_*_latest.csv files to support HPO runs
            import pandas as pd
            import glob
            
            eval_files = glob.glob("data/metrics/evaluation_results_*_latest.csv")
            combined_results = []
            
            for f in eval_files:
                try:
                    df = pd.read_csv(f)
                    # Extract model family from filename if not in columns
                    # filename format: evaluation_results_{family}_latest.csv
                    if 'model_family' not in df.columns:
                        family = os.path.basename(f).replace("evaluation_results_", "").replace("_latest.csv", "")
                        # Handle 'baseline' vs 'BaselineLinear' mapping if needed
                        if family == 'baseline':
                            family = 'BaselineLinear'
                        df['model_family'] = family
                    
                    combined_results.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {f}: {e}")

            metrics_data = {
                'quality_report': quality_report,
                'forecast_output': forecast_output
            }

            if combined_results:
                full_df = pd.concat(combined_results, ignore_index=True)
                
                # Calculate summary metrics
                metrics_data['evaluation_results'] = {
                    'total_models': len(full_df),
                    'avg_mape': full_df['mape'].mean() if 'mape' in full_df.columns else 0,
                    'median_mape': full_df['mape'].median() if 'mape' in full_df.columns else 0,
                    'best_symbol': full_df.loc[full_df['mape'].idxmin()]['symbol'] if 'mape' in full_df.columns and not full_df.empty else 'N/A',
                    'worst_symbol': full_df.loc[full_df['mape'].idxmax()]['symbol'] if 'mape' in full_df.columns and not full_df.empty else 'N/A'
                }
                
                # Perform model comparison (Champion Selection)
                metrics_data['model_comparison'] = self._compare_models(full_df)
                
                # Add comprehensive metrics (mock or real)
                try:
                    # Calculate comprehensive metrics for top models
                    comprehensive_metrics = {}
                    if 'symbol' in full_df.columns and 'mape' in full_df.columns:
                        for symbol in full_df['symbol'].unique()[:5]:  # Top 5 symbols
                            symbol_data = full_df[full_df['symbol'] == symbol]
                            if not symbol_data.empty:
                                # Use the best performing model for this symbol
                                best_model = symbol_data.loc[symbol_data['mape'].idxmin()]
                                # Mock actual/predicted values for demonstration
                                y_true = np.array([100, 105, 102, 108, 106])  # Mock price data
                                y_pred = np.array([99, 106, 101, 109, 105])  # Mock predictions

                                from src.evaluation.enhanced_metrics import EnhancedFinancialMetrics
                                evaluator = EnhancedFinancialMetrics()
                                metrics = evaluator.calculate_comprehensive_metrics(y_true, y_pred)

                                comprehensive_metrics[symbol] = metrics

                    metrics_data['comprehensive_metrics'] = comprehensive_metrics

                except Exception as e:
                    logger.warning(f"Failed to calculate comprehensive metrics: {e}")

            return metrics_data

        except Exception as e:
            logger.warning(f"Failed to load evaluation metrics: {e}")
            return None

    def _compare_models(self, df: Any) -> Dict[str, Any]:
        """
        Compare BaselineLinear against Deep Models (AutoNHITS, AutoTFT, DeepAR).
        Logic:
        1. If BaselineLinear is winning (lowest SMAPE), stick with it.
        2. If Deep Model beats BaselineLinear by > 5% margin, promote it.
        3. If margin is small (< 5%), prefer BaselineLinear.
        """
        comparison = {
            "leaderboard": {},
            "promotions": [],
            "baseline_wins": 0,
            "challenger_wins": 0
        }
        
        if df.empty or 'model_family' not in df.columns or 'smape' not in df.columns:
            return comparison

        # Group by symbol and find best model for each
        for symbol, group in df.groupby('symbol'):
            # Find baseline performance
            baseline_row = group[group['model_family'] == 'BaselineLinear']
            if baseline_row.empty:
                # If no baseline, just take the best model
                best_model = group.loc[group['smape'].idxmin()]
                comparison['leaderboard'][symbol] = best_model['model_family']
                continue
                
            baseline_smape = baseline_row.iloc[0]['smape']
            
            # Find best challenger
            challengers = group[group['model_family'] != 'BaselineLinear']
            if challengers.empty:
                comparison['leaderboard'][symbol] = 'BaselineLinear'
                comparison['baseline_wins'] += 1
                continue
                
            best_challenger = challengers.loc[challengers['smape'].idxmin()]
            challenger_smape = best_challenger['smape']
            challenger_family = best_challenger['model_family']
            
            # Apply 5% margin rule
            # Improvement is (Baseline - Challenger) / Baseline
            improvement = (baseline_smape - challenger_smape) / baseline_smape
            
            if improvement > 0.05:
                # Challenger wins significantly
                comparison['leaderboard'][symbol] = challenger_family
                comparison['promotions'].append({
                    "symbol": symbol,
                    "from": "BaselineLinear",
                    "to": challenger_family,
                    "improvement": improvement
                })
                comparison['challenger_wins'] += 1
            else:
                # Baseline wins or challenger margin too small
                comparison['leaderboard'][symbol] = 'BaselineLinear'
                comparison['baseline_wins'] += 1
                
        return comparison

    def _summarize_evaluation_metrics(self, evaluation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of evaluation metrics for the report.
        """
        summary = {
            'total_models_evaluated': 0,
            'avg_mape': 0,
            'best_performing_symbol': 'N/A',
            'worst_performing_symbol': 'N/A',
            'risk_assessment': {}
        }

        try:
            eval_results = evaluation_metrics.get('evaluation_results', {})
            if eval_results:
                # This would be populated from actual CSV data
                summary.update({
                    'total_models_evaluated': eval_results.get('total_models', 0),
                    'avg_mape': eval_results.get('avg_mape', 0),
                    'best_performing_symbol': eval_results.get('best_symbol', 'N/A'),
                    'worst_performing_symbol': eval_results.get('worst_symbol', 'N/A')
                })

            # Add quality report summary
            quality = evaluation_metrics.get('quality_report', {})
            if quality:
                summary['quality_score'] = quality.get('overall_score', 0)
                summary['issues_count'] = quality.get('total_issues', 0)

        except Exception as e:
            logger.warning(f"Failed to summarize evaluation metrics: {e}")

        return summary

    def _extract_evaluation_insights(self, evaluation_metrics: Dict[str, Any]) -> List[str]:
        """
        Extract key insights from evaluation metrics.
        """
        insights = []

        try:
            comprehensive = evaluation_metrics.get('comprehensive_metrics', {})

            for symbol, metrics in comprehensive.items():
                directional_acc = metrics.get('directional_metrics', {}).get('directional_accuracy', 0)
                if directional_acc > 0.6:
                    insights.append(f"{symbol} shows strong directional accuracy ({directional_acc:.1%})")
                elif directional_acc < 0.4:
                    insights.append(f"{symbol} has poor directional accuracy ({directional_acc:.1%}) - needs attention")

                # Add trading insights
                trading = metrics.get('trading_metrics', {})
                if trading.get('win_rate', 0) > 0.6:
                    insights.append(f"{symbol} demonstrates profitable trading strategy (win rate: {trading['win_rate']:.1%})")

        except Exception as e:
            logger.warning(f"Failed to extract evaluation insights: {e}")

        if not insights:
            insights = ["Evaluation metrics analysis completed - no significant insights identified"]

        return insights

    def _sanitize_for_json(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._sanitize_for_json(obj.tolist())
        else:
            return obj

    @traceable(
        name="reporting_agent_generate_report",
        tags=["reporting", "llm", "synthesis"],
        metadata={"role": "reporting_agent"}
    )
    def generate_report(self, report_input: ReportingInput) -> FullReport:
        """
        Generate a comprehensive system report from aggregated inputs.
        This call is traced to LangSmith.
        """
        from src.configs.llm_prompts import build_reporting_agent_user_prompt

        # Build user prompt using the factory approach
        from src.configs.llm_prompts import PROMPTS
        base_system_prompt = PROMPTS.get("reporting_agent", "You are a helpful reporting assistant.")
        
        # Inject HARD RULES for consistency
        system_prompt = base_system_prompt + """
        
        CRITICAL INSTRUCTIONS:
        1. DO NOT invent numbers. If a metric is 0 in the input, it MUST be 0 in the report.
        2. If 'anomalies_detected' is 0, you MUST say "No anomalies detected".
        3. If 'total_trials' is 0, state clearly that no HPO trials were run.
        4. Your output MUST be consistent with the provided JSON metrics.
        """

        # Enhance the input with evaluation metrics
        enhanced_input = self._enhance_input_with_evaluation_metrics(report_input)
        
        # Build metrics overview (Python-owned truth)
        metrics_overview = self._build_metrics_overview(enhanced_input)

        # Sanitize all inputs to ensure JSON serializability (handle numpy types)
        sanitized_metrics = self._sanitize_for_json(metrics_overview)
        sanitized_analytics = self._sanitize_for_json(enhanced_input.analytics_summary)
        sanitized_hpo = self._sanitize_for_json(enhanced_input.hpo_plan)
        sanitized_research = self._sanitize_for_json(enhanced_input.research_insights)
        sanitized_guardrail = self._sanitize_for_json(enhanced_input.guardrail_status)
        sanitized_metadata = self._sanitize_for_json(enhanced_input.run_metadata)

        user_prompt = build_reporting_agent_user_prompt(
            metrics_overview=sanitized_metrics,
            analytics_summary=sanitized_analytics,
            hpo_plan=sanitized_hpo,
            research_insights=sanitized_research,
            guardrail_status=sanitized_guardrail,
            run_metadata=sanitized_metadata
        )

        logger.info("Calling LLM for report generation (LangSmith tracing enabled)")

        # Record Prometheus metrics
        self._record_llm_call_start()

        try:
            raw = self.llm.complete(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.2,
                max_tokens=2000,
            )

            self._record_llm_call_success()
            logger.info(f"Raw LLM response (first 500 chars): {raw[:500]}")

        except Exception as e:
            self._record_llm_call_error()
            logger.error(f"LLM call failed: {e}")
            raise

        try:
            json_str = extract_json_from_llm_output(raw)
            data = json.loads(json_str)
            logger.info("Successfully parsed LLM response as JSON")
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}. Raw response: {raw}")
            # Fallback: everything goes into executive_summary
            data = {
                "executive_summary": raw,
                "sections": [],
                "key_risks": ["Unable to parse structured data"],
                "key_opportunities": [],
                "actions_for_quants": [],
                "actions_for_ops": []
            }

        # Filter only known top-level keys (from SystemReport)
        valid_keys = SystemReport.__annotations__.keys()
        filtered: Dict[str, Any] = {k: v for k, v in data.items() if k in valid_keys}

        required_keys = set(valid_keys)
        missing_keys = required_keys - set(filtered.keys())
        if missing_keys:
            logger.warning(f"LLM response missing keys: {missing_keys}. Filling with defaults.")

        # --- New-schema defaults ---
        if "executive_summary" not in filtered:
            filtered["executive_summary"] = "Executive summary missing."

        filtered.setdefault("sections", [])
        filtered.setdefault("key_risks", [])
        filtered.setdefault("key_opportunities", [])
        filtered.setdefault("actions_for_quants", [])
        filtered.setdefault("actions_for_ops", [])

        # --- Legacy fields defaults (kept empty unless LLM ever adds them) ---
        filtered.setdefault("performance_overview", {})
        filtered.setdefault("risk_assessment", {})
        filtered.setdefault("optimization_recommendations", {})
        filtered.setdefault("research_insights", {})
        filtered.setdefault("operational_notes", {})
        filtered.setdefault("priority_actions", [])

        # Normalize sections into ReportSection instances
        sections_raw = filtered.get("sections", [])
        normalized_sections: List[ReportSection] = []
        for s in sections_raw:
            if not isinstance(s, dict):
                continue
            title = s.get("title") or "Untitled"
            audience = s.get("audience") or "mixed"
            body = s.get("body_markdown") or ""
            normalized_sections.append(ReportSection(title=title,
                                                     audience=audience,
                                                     body_markdown=body))
        filtered["sections"] = normalized_sections

        system_report = SystemReport(**filtered)
        
        # Create FullReport
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"report_{timestamp}"
        
        # Validate consistency before returning
        # We render a temporary HTML to check against
        temp_html = render_html_report(FullReport(
            report_id=report_id,
            timestamp=timestamp,
            run_type=enhanced_input.run_metadata.get('run_type', 'unknown'),
            metrics_overview=metrics_overview,
            system_report=system_report
        ))
        
        try:
            validate_report_consistency(temp_html, metrics_overview)
        except ReportConsistencyError as e:
            logger.error(f"Report consistency check failed: {e}")
            # We could choose to fail hard, or just log it. 
            # For now, let's log it and maybe append a warning to the executive summary.
            system_report.executive_summary += f"\n\n[AUTOMATED WARNING: {str(e)}]"

        full_report = FullReport(
            report_id=report_id,
            timestamp=timestamp,
            run_type=enhanced_input.run_metadata.get('run_type', 'unknown'),
            metrics_overview=metrics_overview,
            system_report=system_report
        )

        return full_report

    def generate_and_store_report(self, report_input: ReportingInput) -> Dict[str, Any]:
        """
        End-to-end report generation: create, store, and distribute.

        Returns:
            Dict with report metadata and file paths
        """
        # Generate the report
        full_report = self.generate_report(report_input)

        # Store raw JSON
        json_path = self._store_json_report(full_report)

        # Generate and store Markdown
        markdown_path = self._store_markdown_report(full_report)

        # Generate and store HTML
        html_path = self._store_html_report(full_report)

        # Send email if enabled
        if self.email_enabled and self.email_recipients:
            self._send_email_report(full_report, markdown_path)

        # Update dashboard if enabled
        if self.dashboard_enabled:
            self._update_dashboard(full_report, html_path)

        metadata = {
            'report_id': full_report.report_id,
            'timestamp': full_report.timestamp,
            'json_path': str(json_path),
            'markdown_path': str(markdown_path),
            'html_path': str(html_path),
            'email_sent': self.email_enabled and bool(self.email_recipients),
            'dashboard_updated': self.dashboard_enabled
        }

        # Store report for continuous learning
        # Convert SystemReport to dict, handling nested dataclasses
        report_dict = full_report.system_report.__dict__.copy()
        if full_report.system_report.sections:
            report_dict['sections'] = [s.__dict__ for s in full_report.system_report.sections]
        self._last_report = report_dict

        # Record metrics if available
        if self.metrics_available and self.metrics_service:
            try:
                self.metrics_service.record_report_generation(
                    report_type='comprehensive',
                    success=True,
                    duration=0,  # Could track actual duration
                    size_bytes=len(json.dumps(report_dict))
                )
            except Exception as e:
                logger.warning(f"Failed to record report metrics: {e}")

        logger.info(f"Report {full_report.report_id} generated and stored successfully")
        return metadata

    def _store_json_report(self, report: FullReport) -> Path:
        """Store raw JSON report."""
        json_path = self.reports_dir / f"{report.report_id}.json"

        # Convert dataclass to dict for JSON serialization
        # Handle ReportSection objects
        sections_dict = [
            {'title': s.title, 'audience': s.audience, 'body_markdown': s.body_markdown}
            for s in report.system_report.sections
        ]

        report_dict = {
            'report_id': report.report_id,
            'timestamp': report.timestamp,
            'run_type': report.run_type,
            'metrics_overview': report.metrics_overview,
            'system_report': {
                'executive_summary': report.system_report.executive_summary,
                'sections': sections_dict,
                'key_risks': report.system_report.key_risks,
                'key_opportunities': report.system_report.key_opportunities,
                'actions_for_quants': report.system_report.actions_for_quants,
                'actions_for_ops': report.system_report.actions_for_ops,
                # Legacy fields
                'performance_overview': report.system_report.performance_overview,
                'risk_assessment': report.system_report.risk_assessment,
                'optimization_recommendations': report.system_report.optimization_recommendations,
                'research_insights': report.system_report.research_insights,
                'operational_notes': report.system_report.operational_notes,
                'priority_actions': report.system_report.priority_actions
            }
        }

        with open(json_path, 'w') as f:
            json.dump(report_dict, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Stored JSON report at {json_path}")
        return json_path

    def _store_markdown_report(self, report: FullReport) -> Path:
        """Generate and store Markdown report."""
        markdown_path = self.reports_dir / f"{report.report_id}.md"
        markdown_content = self.generate_markdown_report(report.system_report, report.timestamp)

        with open(markdown_path, 'w') as f:
            f.write(markdown_content)

        logger.info(f"Stored Markdown report at {markdown_path}")
        return markdown_path

    def _store_html_report(self, report: FullReport) -> Path:
        """Generate and store HTML report."""
        html_path = self.reports_dir / f"{report.report_id}.html"
        html_content = render_html_report(report)

        # Post-process to enforce ground truth
        # We need to reconstruct the guardrail status string from the metrics
        # or pass it down. The metrics_overview has it.
        guard_status = report.metrics_overview.get("guardrails", {}).get("status", "unknown")
        
        html_content = normalize_system_report(
            report_html=html_content,
            metrics=report.metrics_overview,
            guardrail_status=guard_status
        )

        with open(html_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Stored HTML report at {html_path}")
        return html_path

    def _send_email_report(self, report: FullReport, markdown_path: Path):
        """Send email with report."""
        try:
            from src.services.email_service import EmailService

            email_service = EmailService()
            subject = f"Daily Forecast Platform Report - {report.timestamp}"

            # Read markdown content and generate HTML
            with open(markdown_path, 'r') as f:
                markdown_content = f.read()
            html_content = render_html_report(report)

            # Send to all recipients
            for recipient in self.email_recipients:
                self.email_service.send_email_report(
                    subject=subject,
                    html_content=html_content,
                    markdown_content=markdown_content,
                    recipients=[recipient]
                )

            logger.info(f"Sent email report to {len(self.email_recipients)} recipients")

        except ImportError:
            logger.warning("Email service not available, skipping email delivery")
        except Exception as e:
            logger.error(f"Failed to send email report: {e}")

    def _update_dashboard(self, report: FullReport, html_path: Path):
        """Update web dashboard with latest report."""
        try:
            dashboard_index = self.dashboard_dir / "index.html"
            dashboard_index.parent.mkdir(parents=True, exist_ok=True)

            # Create a simple dashboard index
            dashboard_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Agentic Forecast Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .report-link {{ margin: 10px 0; }}
                </style>
            </head>
            <body>
                <h1>Agentic Forecast Platform Dashboard</h1>
                <p><strong>Last Updated:</strong> {report.timestamp}</p>
                <h2>Latest Reports</h2>
                <div class="report-link">
                    <a href="{html_path.name}">Latest Report ({report.timestamp})</a>
                </div>
                <h2>System Status</h2>
                <p>Executive Summary: {report.system_report.executive_summary[:200]}...</p>
            </body>
            </html>
            """

            with open(dashboard_index, 'w') as f:
                f.write(dashboard_html)

            logger.info(f"Updated dashboard at {dashboard_index}")

        except Exception as e:
            logger.error(f"Failed to update dashboard: {e}")

    def _record_llm_call_start(self):
        """Record Prometheus metrics for LLM call start."""
        # Not supported by current MetricsService
        pass

    def _record_llm_call_success(self):
        """Record Prometheus metrics for successful LLM call."""
        if self.metrics_service:
            try:
                self.metrics_service.record_llm_call(
                    agent_name='reporting_agent',
                    model='gpt-4o', # Assumption
                    tokens_used=0, # Unknown
                    cost=0.0,
                    success=True,
                    duration=0.0
                )
            except Exception:
                pass

    def _record_llm_call_error(self):
        """Record Prometheus metrics for failed LLM call."""
        if self.metrics_service:
            try:
                self.metrics_service.record_llm_call(
                    agent_name='reporting_agent',
                    model='gpt-4o',
                    tokens_used=0,
                    cost=0.0,
                    success=False,
                    duration=0.0
                )
            except Exception:
                pass

    def generate_markdown_report(self, report: SystemReport, timestamp: Optional[str] = None) -> str:
        """
        Convert the structured report to a formatted Markdown document.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = []

        # Title and Executive Summary
        md.append("# System Forecasting Report")
        md.append(f"**Generated:** {timestamp}")
        md.append("")

        md.append("## Executive Summary")
        md.append(report.executive_summary)
        md.append("")

        # New Schema Sections
        if report.sections:
            for section in report.sections:
                md.append(f"## {section.title}")
                md.append(f"*(Audience: {section.audience})*")
                md.append(section.body_markdown)
                md.append("")

        if report.key_risks:
            md.append("## Key Risks")
            for risk in report.key_risks:
                md.append(f"- {risk}")
            md.append("")

        if report.key_opportunities:
            md.append("## Key Opportunities")
            for opp in report.key_opportunities:
                md.append(f"- {opp}")
            md.append("")

        if report.actions_for_quants:
            md.append("## Actions for Quants")
            for action in report.actions_for_quants:
                md.append(f"- {action}")
            md.append("")

        if report.actions_for_ops:
            md.append("## Actions for Ops")
            for action in report.actions_for_ops:
                md.append(f"- {action}")
            md.append("")

        # Legacy Fields (Fallback)
        if report.performance_overview:
            md.append("## Performance Overview (Legacy)")
            perf = report.performance_overview
            md.append(f"**Global Metrics:** {perf.get('global_metrics', 'N/A')}")
            md.append(f"**Symbol Performance:** {perf.get('symbol_performance', 'N/A')}")
            md.append("")

        if report.risk_assessment:
            md.append("## Risk Assessment (Legacy)")
            risk_dict = report.risk_assessment
            md.append("**Current Risks:**")
            for risk_item in risk_dict.get('current_risks', []):
                md.append(f"- {risk_item}")
            md.append("")

        if report.priority_actions:
            md.append("## Priority Actions (Legacy)")
            for action in report.priority_actions:
                if isinstance(action, dict):
                    md.append(f"### {action.get('action', 'Unknown Action')}")
                    md.append(f"- **Priority:** {action.get('priority', 'Unknown')}")
                else:
                    md.append(f"- {action}")
            md.append("")

        return "\n".join(md)

    # generate_html_report is now replaced by render_html_report function
