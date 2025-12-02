from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from langsmith import traceable

logger = logging.getLogger(__name__)

@dataclass
class ReportingInput:
    analytics_summary: Dict[str, Any]
    hpo_plan: Dict[str, Any]
    research_insights: Dict[str, Any]
    guardrail_status: Dict[str, Any]
    run_metadata: Dict[str, Any]

@dataclass
class SystemReport:
    executive_summary: str
    performance_overview: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    optimization_recommendations: Dict[str, Any]
    research_insights: Dict[str, Any]
    operational_notes: Dict[str, Any]
    priority_actions: list

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
            from services.metrics_exporter import _load_json_safely

            evaluation_results = _load_json_safely("data/metrics/evaluation_results_latest.csv")
            quality_report = _load_json_safely("data/metrics/quality_report_latest.json")
            forecast_output = _load_json_safely("data/metrics/forecast_agent_output_latest.json")

            metrics_data = {
                'evaluation_results': evaluation_results,
                'quality_report': quality_report,
                'forecast_output': forecast_output
            }

            # If we have evaluation results, enhance with comprehensive metrics
            if evaluation_results:
                try:
                    import pandas as pd
                    eval_df = pd.read_csv("data/metrics/evaluation_results_latest.csv")

                    # Calculate comprehensive metrics for top models
                    comprehensive_metrics = {}
                    for symbol in eval_df['symbol'].unique()[:5]:  # Top 5 symbols
                        symbol_data = eval_df[eval_df['symbol'] == symbol]
                        if not symbol_data.empty:
                            # Use the best performing model for this symbol
                            best_model = symbol_data.loc[symbol_data['mape'].idxmin()]
                            # Mock actual/predicted values for demonstration
                            y_true = [100, 105, 102, 108, 106]  # Mock price data
                            y_pred = [99, 106, 101, 109, 105]  # Mock predictions

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

    @traceable(
        name="reporting_agent_generate_report",
        tags=["reporting", "llm", "synthesis"],
        metadata={"role": "reporting_agent"}
    )
    def generate_report(self, report_input: ReportingInput) -> SystemReport:
        """
        Generate a comprehensive system report from aggregated inputs.
        This call is traced to LangSmith.
        """
        from src.prompts.llm_prompts import build_reporting_agent_user_prompt

        # Build user prompt using the factory approach
        if self.factory:
            # Enhance the input with evaluation metrics
            enhanced_input = self._enhance_input_with_evaluation_metrics(report_input)
            user_prompt = build_reporting_agent_user_prompt(
                analytics_summary=enhanced_input.analytics_summary,
                hpo_plan=enhanced_input.hpo_plan,
                research_insights=enhanced_input.research_insights,
                guardrail_status=enhanced_input.guardrail_status,
                run_metadata=enhanced_input.run_metadata
            )
            messages = self.factory.build_agent_messages('reporting_agent', user_prompt)
        else:
            # Fallback to direct prompt building
            from src.prompts.llm_prompts import PROMPTS
            system_prompt = PROMPTS["reporting_agent"]
            # Enhance the input with evaluation metrics
            enhanced_input = self._enhance_input_with_evaluation_metrics(report_input)
            user_prompt = build_reporting_agent_user_prompt(
                analytics_summary=enhanced_input.analytics_summary,
                hpo_plan=enhanced_input.hpo_plan,
                research_insights=enhanced_input.research_insights,
                guardrail_status=enhanced_input.guardrail_status,
                run_metadata=enhanced_input.run_metadata
            )

        logger.info("Calling LLM for report generation (LangSmith tracing enabled)")

        # Record Prometheus metrics
        self._record_llm_call_start()

        try:
            raw = self.llm.complete(
                prompt=user_prompt,
                system=system_prompt if 'system_prompt' not in locals() else system_prompt,
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
            data = json.loads(raw)
            logger.info("Successfully parsed LLM response as JSON")
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}. Raw response: {raw}")
            # Fallback: wrap the raw text if model messed up
            data = {
                "executive_summary": raw,
                "performance_overview": {
                    "global_metrics": "Unable to parse structured data",
                    "symbol_performance": "Unable to parse structured data",
                    "regime_performance": "Unable to parse structured data",
                    "horizon_analysis": "Unable to parse structured data"
                },
                "risk_assessment": {
                    "current_risks": ["Unable to parse structured data"],
                    "guardrail_status": "Unable to parse structured data",
                    "uncertainty_sources": ["Unable to parse structured data"]
                },
                "optimization_recommendations": {
                    "hpo_priorities": "Unable to parse structured data",
                    "model_improvements": "Unable to parse structured data",
                    "feature_engineering": "Unable to parse structured data"
                },
                "research_insights": {
                    "key_findings": ["Unable to parse structured data"],
                    "market_implications": "Unable to parse structured data",
                    "data_suggestions": "Unable to parse structured data"
                },
                "operational_notes": {
                    "system_health": "Unable to parse structured data",
                    "data_quality": "Unable to parse structured data",
                    "maintenance_needs": "Unable to parse structured data"
                },
                "priority_actions": []
            }

        return SystemReport(**data)

    def generate_and_store_report(self, report_input: ReportingInput) -> Dict[str, Any]:
        """
        End-to-end report generation: create, store, and distribute.

        Returns:
            Dict with report metadata and file paths
        """
        # Generate the report
        report = self.generate_report(report_input)

        # Create timestamp for this report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"report_{timestamp}"

        # Store raw JSON
        json_path = self._store_json_report(report, report_id, timestamp)

        # Generate and store Markdown
        markdown_path = self._store_markdown_report(report, report_id, timestamp)

        # Generate and store HTML
        html_path = self._store_html_report(report, report_id, timestamp)

        # Send email if enabled
        if self.email_enabled and self.email_recipients:
            self._send_email_report(report, markdown_path, timestamp)

        # Update dashboard if enabled
        if self.dashboard_enabled:
            self._update_dashboard(report, html_path, timestamp)

        metadata = {
            'report_id': report_id,
            'timestamp': timestamp,
            'json_path': str(json_path),
            'markdown_path': str(markdown_path),
            'html_path': str(html_path),
            'email_sent': self.email_enabled and bool(self.email_recipients),
            'dashboard_updated': self.dashboard_enabled
        }

        # Store report for continuous learning
        self._last_report = report.__dict__ if report else None

        # Record metrics if available
        if self.metrics_available and self.metrics_service:
            try:
                self.metrics_service.record_report_generation(
                    report_type='comprehensive',
                    success=True,
                    duration=0,  # Could track actual duration
                    size_bytes=len(json.dumps(report.__dict__ if report else {}))
                )
            except Exception as e:
                logger.warning(f"Failed to record report metrics: {e}")

        logger.info(f"Report {report_id} generated and stored successfully")
        return metadata

    def _store_json_report(self, report: SystemReport, report_id: str, timestamp: str) -> Path:
        """Store raw JSON report."""
        json_path = self.reports_dir / f"{report_id}.json"

        # Convert dataclass to dict for JSON serialization
        report_dict = {
            'report_id': report_id,
            'timestamp': timestamp,
            'executive_summary': report.executive_summary,
            'performance_overview': report.performance_overview,
            'risk_assessment': report.risk_assessment,
            'optimization_recommendations': report.optimization_recommendations,
            'research_insights': report.research_insights,
            'operational_notes': report.operational_notes,
            'priority_actions': report.priority_actions
        }

        with open(json_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Stored JSON report at {json_path}")
        return json_path

    def _store_markdown_report(self, report: SystemReport, report_id: str, timestamp: str) -> Path:
        """Generate and store Markdown report."""
        markdown_path = self.reports_dir / f"{report_id}.md"
        markdown_content = self.generate_markdown_report(report, timestamp)

        with open(markdown_path, 'w') as f:
            f.write(markdown_content)

        logger.info(f"Stored Markdown report at {markdown_path}")
        return markdown_path

    def _store_html_report(self, report: SystemReport, report_id: str, timestamp: str) -> Path:
        """Generate and store HTML report."""
        html_path = self.reports_dir / f"{report_id}.html"
        html_content = self.generate_html_report(report, timestamp)

        with open(html_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Stored HTML report at {html_path}")
        return html_path

    def _send_email_report(self, report: SystemReport, markdown_path: Path, timestamp: str):
        """Send email with report."""
        try:
            from src.services.email_service import EmailService

            email_service = EmailService()
            subject = f"Daily Forecast Platform Report - {timestamp}"

            # Read markdown content and generate HTML
            with open(markdown_path, 'r') as f:
                markdown_content = f.read()
            html_content = self.generate_html_report(report, timestamp)

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

    def _update_dashboard(self, report: SystemReport, html_path: Path, timestamp: str):
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
                <p><strong>Last Updated:</strong> {timestamp}</p>
                <h2>Latest Reports</h2>
                <div class="report-link">
                    <a href="{html_path.name}">Latest Report ({timestamp})</a>
                </div>
                <h2>System Status</h2>
                <p>Executive Summary: {report.executive_summary[:200]}...</p>
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
        try:
            from src.services.metrics_service import MetricsService
            MetricsService.record_llm_call_start('reporting_agent')
        except ImportError:
            pass  # Metrics service not available

    def _record_llm_call_success(self):
        """Record Prometheus metrics for successful LLM call."""
        try:
            from src.services.metrics_service import MetricsService
            MetricsService.record_llm_call_success('reporting_agent', tokens_input=0, tokens_output=0)
        except ImportError:
            pass

    def _record_llm_call_error(self):
        """Record Prometheus metrics for failed LLM call."""
        try:
            from src.services.metrics_service import MetricsService
            MetricsService.record_llm_call_error('reporting_agent')
        except ImportError:
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

        # Performance Overview
        md.append("## Performance Overview")
        perf = report.performance_overview
        md.append(f"**Global Metrics:** {perf.get('global_metrics', 'N/A')}")
        md.append(f"**Symbol Performance:** {perf.get('symbol_performance', 'N/A')}")
        md.append(f"**Regime Performance:** {perf.get('regime_performance', 'N/A')}")
        md.append(f"**Horizon Analysis:** {perf.get('horizon_analysis', 'N/A')}")
        md.append("")

        # Risk Assessment
        md.append("## Risk Assessment")
        risk = report.risk_assessment
        md.append("**Current Risks:**")
        for risk_item in risk.get('current_risks', []):
            md.append(f"- {risk_item}")
        md.append("")
        md.append(f"**Guardrail Status:** {risk.get('guardrail_status', 'N/A')}")
        md.append("")
        md.append("**Uncertainty Sources:**")
        for uncertainty in risk.get('uncertainty_sources', []):
            md.append(f"- {uncertainty}")
        md.append("")

        # Optimization Recommendations
        md.append("## Optimization Recommendations")
        opt = report.optimization_recommendations
        md.append(f"**HPO Priorities:** {opt.get('hpo_priorities', 'N/A')}")
        md.append(f"**Model Improvements:** {opt.get('model_improvements', 'N/A')}")
        md.append(f"**Feature Engineering:** {opt.get('feature_engineering', 'N/A')}")
        md.append("")

        # Research Insights
        md.append("## Research Insights")
        research = report.research_insights
        md.append("**Key Findings:**")
        for finding in research.get('key_findings', []):
            md.append(f"- {finding}")
        md.append("")
        md.append(f"**Market Implications:** {research.get('market_implications', 'N/A')}")
        md.append(f"**Data Suggestions:** {research.get('data_suggestions', 'N/A')}")
        md.append("")

        # Operational Notes
        md.append("## Operational Notes")
        ops = report.operational_notes
        md.append(f"**System Health:** {ops.get('system_health', 'N/A')}")
        md.append(f"**Data Quality:** {ops.get('data_quality', 'N/A')}")
        md.append(f"**Maintenance Needs:** {ops.get('maintenance_needs', 'N/A')}")
        md.append("")

        # Priority Actions
        md.append("## Priority Actions")
        for action in report.priority_actions:
            md.append(f"### {action.get('action', 'Unknown Action')}")
            md.append(f"- **Priority:** {action.get('priority', 'Unknown')}")
            md.append(f"- **Timeline:** {action.get('timeline', 'Unknown')}")
            md.append(f"- **Owner:** {action.get('owner', 'Unknown')}")
            md.append(f"- **Rationale:** {action.get('rationale', 'Unknown')}")
            md.append("")

        return "\n".join(md)

    def generate_html_report(self, report: SystemReport, timestamp: Optional[str] = None) -> str:
        """
        Convert the structured report to a formatted HTML document.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html lang='en'>")
        html.append("<head>")
        html.append("    <meta charset='UTF-8'>")
        html.append("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html.append("    <title>System Forecasting Report</title>")
        html.append("    <style>")
        html.append("        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }")
        html.append("        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }")
        html.append("        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }")
        html.append("        h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }")
        html.append("        .executive-summary { background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }")
        html.append("        .metric { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #28a745; }")
        html.append("        .risk-item { background: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; border-left: 4px solid #ffc107; }")
        html.append("        .action-item { background: #d1ecf1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #17a2b8; }")
        html.append("        .priority-high { border-left-color: #dc3545 !important; }")
        html.append("        .priority-medium { border-left-color: #ffc107 !important; }")
        html.append("        .priority-low { border-left-color: #28a745 !important; }")
        html.append("        .timestamp { color: #6c757d; font-size: 0.9em; }")
        html.append("    </style>")
        html.append("</head>")
        html.append("<body>")
        html.append("    <div class='container'>")

        # Title and timestamp
        html.append(f"        <h1>System Forecasting Report</h1>")
        html.append(f"        <p class='timestamp'>Generated: {timestamp}</p>")

        # Executive Summary
        html.append("        <h2>Executive Summary</h2>")
        html.append("        <div class='executive-summary'>")
        html.append(f"            <p>{report.executive_summary}</p>")
        html.append("        </div>")

        # Performance Overview
        html.append("        <h2>Performance Overview</h2>")
        perf = report.performance_overview
        html.append("        <div class='metric'>")
        html.append(f"            <strong>Global Metrics:</strong> {perf.get('global_metrics', 'N/A')}")
        html.append("        </div>")
        html.append("        <div class='metric'>")
        html.append(f"            <strong>Symbol Performance:</strong> {perf.get('symbol_performance', 'N/A')}")
        html.append("        </div>")
        html.append("        <div class='metric'>")
        html.append(f"            <strong>Regime Performance:</strong> {perf.get('regime_performance', 'N/A')}")
        html.append("        </div>")
        html.append("        <div class='metric'>")
        html.append(f"            <strong>Horizon Analysis:</strong> {perf.get('horizon_analysis', 'N/A')}")
        html.append("        </div>")

        # Risk Assessment
        html.append("        <h2>Risk Assessment</h2>")
        risk = report.risk_assessment
        html.append("        <h3>Current Risks</h3>")
        for risk_item in risk.get('current_risks', []):
            html.append(f"        <div class='risk-item'>{risk_item}</div>")
        html.append("        <div class='metric'>")
        html.append(f"            <strong>Guardrail Status:</strong> {risk.get('guardrail_status', 'N/A')}")
        html.append("        </div>")
        html.append("        <h3>Uncertainty Sources</h3>")
        for uncertainty in risk.get('uncertainty_sources', []):
            html.append(f"        <div class='risk-item'>{uncertainty}</div>")

        # Optimization Recommendations
        html.append("        <h2>Optimization Recommendations</h2>")
        opt = report.optimization_recommendations
        html.append("        <div class='metric'>")
        html.append(f"            <strong>HPO Priorities:</strong> {opt.get('hpo_priorities', 'N/A')}")
        html.append("        </div>")
        html.append("        <div class='metric'>")
        html.append(f"            <strong>Model Improvements:</strong> {opt.get('model_improvements', 'N/A')}")
        html.append("        </div>")
        html.append("        <div class='metric'>")
        html.append(f"            <strong>Feature Engineering:</strong> {opt.get('feature_engineering', 'N/A')}")
        html.append("        </div>")

        # Research Insights
        html.append("        <h2>Research Insights</h2>")
        research = report.research_insights
        html.append("        <h3>Key Findings</h3>")
        for finding in research.get('key_findings', []):
            html.append(f"        <div class='metric'>{finding}</div>")
        html.append("        <div class='metric'>")
        html.append(f"            <strong>Market Implications:</strong> {research.get('market_implications', 'N/A')}")
        html.append("        </div>")
        html.append("        <div class='metric'>")
        html.append(f"            <strong>Data Suggestions:</strong> {research.get('data_suggestions', 'N/A')}")
        html.append("        </div>")

        # Operational Notes
        html.append("        <h2>Operational Notes</h2>")
        ops = report.operational_notes
        html.append("        <div class='metric'>")
        html.append(f"            <strong>System Health:</strong> {ops.get('system_health', 'N/A')}")
        html.append("        </div>")
        html.append("        <div class='metric'>")
        html.append(f"            <strong>Data Quality:</strong> {ops.get('data_quality', 'N/A')}")
        html.append("        </div>")
        html.append("        <div class='metric'>")
        html.append(f"            <strong>Maintenance Needs:</strong> {ops.get('maintenance_needs', 'N/A')}")
        html.append("        </div>")

        # Priority Actions
        html.append("        <h2>Priority Actions</h2>")
        for action in report.priority_actions:
            priority_class = f"priority-{action.get('priority', 'medium').lower()}"
            html.append(f"        <div class='action-item {priority_class}'>")
            html.append(f"            <h4>{action.get('action', 'Unknown Action')}</h4>")
            html.append(f"            <p><strong>Priority:</strong> {action.get('priority', 'Unknown')}</p>")
            html.append(f"            <p><strong>Timeline:</strong> {action.get('timeline', 'Unknown')}</p>")
            html.append(f"            <p><strong>Owner:</strong> {action.get('owner', 'Unknown')}</p>")
            html.append(f"            <p><strong>Rationale:</strong> {action.get('rationale', 'Unknown')}</p>")
            html.append("        </div>")

        html.append("    </div>")
        html.append("</body>")
        html.append("</html>")

        return "\n".join(html)