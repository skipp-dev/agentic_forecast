from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from langsmith import traceable
from src.utils.llm_utils import extract_json_from_llm_output

logger = logging.getLogger(__name__)

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
            try:
                from src.services.metrics_exporter import _load_json_safely
            except ImportError:
                 # Fallback if not found in src
                 def _load_json_safely(path): return {}

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
                    # Check if file exists before reading
                    if os.path.exists("data/metrics/evaluation_results_latest.csv"):
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
        from src.configs.llm_prompts import build_reporting_agent_user_prompt

        # Build user prompt using the factory approach
        from src.configs.llm_prompts import PROMPTS
        system_prompt = PROMPTS.get("reporting_agent", "You are a helpful reporting assistant.")

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
            # messages = self.factory.build_agent_messages('reporting_agent', user_prompt) # Not used by complete()
        else:
            # Fallback to direct prompt building
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
            return SystemReport(
                executive_summary=raw,
                sections=[],
                key_risks=["Unable to parse structured data"],
                key_opportunities=[],
                actions_for_quants=[],
                actions_for_ops=[],
                performance_overview={"error": "Unable to parse structured data"},
                risk_assessment={"error": "Unable to parse structured data"},
                optimization_recommendations={"error": "Unable to parse structured data"},
                research_insights={"error": "Unable to parse structured data"},
                operational_notes={"error": "Unable to parse structured data"},
                priority_actions=[],
            )

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

        return SystemReport(**filtered)

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
        # Handle ReportSection objects
        sections_dict = [
            {'title': s.title, 'audience': s.audience, 'body_markdown': s.body_markdown}
            for s in report.sections
        ]

        report_dict = {
            'report_id': report_id,
            'timestamp': timestamp,
            'executive_summary': report.executive_summary,
            'sections': sections_dict,
            'key_risks': report.key_risks,
            'key_opportunities': report.key_opportunities,
            'actions_for_quants': report.actions_for_quants,
            'actions_for_ops': report.actions_for_ops,
            # Legacy fields
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
            risk = report.risk_assessment
            md.append("**Current Risks:**")
            for risk_item in risk.get('current_risks', []):
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
        html.append("        .section-box { background: #fff; border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }")
        html.append("        .risk-item { background: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; border-left: 4px solid #ffc107; }")
        html.append("        .opp-item { background: #d4edda; padding: 10px; margin: 5px 0; border-radius: 3px; border-left: 4px solid #28a745; }")
        html.append("        .action-item { background: #d1ecf1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #17a2b8; }")
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

        # New Schema Sections
        if report.sections:
            for section in report.sections:
                html.append(f"        <h2>{section.title}</h2>")
                html.append(f"        <p><em>Audience: {section.audience}</em></p>")
                html.append(f"        <div class='section-box'>")
                # Simple markdown to html conversion (very basic)
                body_html = section.body_markdown.replace('\n', '<br>')
                html.append(f"            {body_html}")
                html.append(f"        </div>")

        if report.key_risks:
            html.append("        <h2>Key Risks</h2>")
            for risk in report.key_risks:
                html.append(f"        <div class='risk-item'>{risk}</div>")

        if report.key_opportunities:
            html.append("        <h2>Key Opportunities</h2>")
            for opp in report.key_opportunities:
                html.append(f"        <div class='opp-item'>{opp}</div>")

        if report.actions_for_quants:
            html.append("        <h2>Actions for Quants</h2>")
            for action in report.actions_for_quants:
                html.append(f"        <div class='action-item'>{action}</div>")

        if report.actions_for_ops:
            html.append("        <h2>Actions for Ops</h2>")
            for action in report.actions_for_ops:
                html.append(f"        <div class='action-item'>{action}</div>")

        # Legacy Fields (Fallback)
        if report.performance_overview:
            html.append("        <h2>Performance Overview (Legacy)</h2>")
            perf = report.performance_overview
            html.append(f"        <div class='section-box'>Global Metrics: {perf.get('global_metrics', 'N/A')}</div>")

        html.append("    </div>")
        html.append("</body>")
        html.append("</html>")

        return "\n".join(html)
