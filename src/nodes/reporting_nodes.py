import pandas as pd
from ..core.state import PipelineGraphState
import os
from typing import Dict, Any, List
from ..agents.reporting_agent import LLMReportingAgent, ReportingInput

def generate_report_node(state: PipelineGraphState) -> PipelineGraphState:
    """
    Generates a comprehensive report using the LLMReportingAgent.
    Creates structured reports with analytics, HPO insights, and recommendations.
    """
    print("--- Node: Generate LLM-Powered Comprehensive Report ---")

    try:
        # Initialize the LLM reporting agent
        config = state.get('config', {})
        reporting_config = config.get('reporting', {})
        reporting_agent = LLMReportingAgent(settings=reporting_config)

        # Extract data from state for the reporting agent
        analytics_summary = extract_analytics_summary(state)
        hpo_plan = extract_hpo_plan(state)
        research_insights = extract_research_insights(state)
        guardrail_status = extract_guardrail_status(state)
        run_metadata = extract_run_metadata(state)

        # Create reporting input
        report_input = ReportingInput(
            analytics_summary=analytics_summary,
            hpo_plan=hpo_plan,
            research_insights=research_insights,
            guardrail_status=guardrail_status,
            run_metadata=run_metadata
        )

        # Generate and store the comprehensive report
        metadata = reporting_agent.generate_and_store_report(report_input)

        print("[OK] LLM-powered comprehensive report generated and stored")
        print(f"   [JSON] JSON: {metadata.get('json_path', 'N/A')}")
        print(f"   [MD] Markdown: {metadata.get('markdown_path', 'N/A')}")
        print(f"   [HTML] HTML: {metadata.get('html_path', 'N/A')}")
        print(f"   [EMAIL] Email sent: {metadata.get('email_sent', False)}")
        print(f"   [DASH] Dashboard updated: {metadata.get('dashboard_updated', False)}")

        # Store report metadata in state
        state['report_metadata'] = metadata
        state['report_generated'] = True

    except Exception as e:
        print(f"[ERROR] Error generating LLM report: {e}")
        # Fallback to basic CSV report
        print("[FALLBACK] Falling back to basic CSV report...")
        generate_fallback_csv_report(state)
        state['report_generated'] = False

    return state

def extract_analytics_summary(state: PipelineGraphState) -> Dict[str, Any]:
    """Extract analytics summary from state."""
    analytics_results = state.get('analytics_results', {})
    
    # Calculate aggregate metrics
    total_symbols = len(state.get('symbols', []))
    mapes = []
    volatilities = []
    
    for symbol, metrics in analytics_results.items():
        if 'mape' in metrics and metrics['mape'] is not None:
            mapes.append(metrics['mape'])
        if 'volatility' in metrics:
            volatilities.append(metrics['volatility'])
            
    avg_performance = sum(mapes) / len(mapes) if mapes else 0
    avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0
    
    # Identify top performers (lowest MAPE)
    perf_list = []
    for symbol, metrics in analytics_results.items():
        if 'mape' in metrics and metrics['mape'] is not None:
            perf_list.append({'symbol': symbol, 'mape': metrics['mape']})
    
    perf_list.sort(key=lambda x: x['mape'])
    top_performers = [item['symbol'] for item in perf_list[:3]]
    
    # Identify high risk (placeholder logic)
    high_risk_symbols = [] 

    summary = {
        'total_symbols': total_symbols,
        'avg_performance': avg_performance,
        'top_performers': top_performers,
        'risk_metrics': {
            'avg_volatility': avg_volatility,
            'high_risk_symbols': high_risk_symbols
        }
    }

    return summary

def extract_hpo_plan(state: PipelineGraphState) -> Dict[str, Any]:
    """Extract HPO plan from state."""
    hpo_results = state.get('hpo_results', {})
    
    plan = {
        'hpo_completed': bool(hpo_results),
        'symbols_optimized': list(hpo_results.keys()) if hpo_results else [],
        'total_optimizations': 0, 
        'best_improvements': [],
        'next_priorities': []
    }
    
    return plan

def extract_research_insights(state: PipelineGraphState) -> Dict[str, Any]:
    """Extract research insights from state."""
    return {
        'feature_importance': {},
        'anomalies_detected': 0,
        'key_findings': [],
        'market_implications': 'N/A',
        'data_suggestions': []
    }

def extract_guardrail_status(state: PipelineGraphState) -> Dict[str, Any]:
    """Extract guardrail status from state."""
    return {
        'total_checks': 0,
        'passed_checks': 0,
        'failed_checks': 0,
        'warnings': 0,
        'critical_issues': [],
        'can_proceed': True
    }

def extract_run_metadata(state: PipelineGraphState) -> Dict[str, Any]:
    """Extract run metadata from state."""
    return {
        'run_type': state.get('run_type', 'DAILY'),
        'symbols_processed': len(state.get('symbols', [])),
        'timestamp': pd.Timestamp.now().isoformat(),
        'execution_time': 'N/A',
        'models_trained': len(state.get('best_models', {})),
        'features_generated': len(state.get('features', {})),
        'forecasts_generated': len(state.get('forecasts', {})),
        'errors_encountered': len(state.get('errors', [])),
        'actions_executed': 0
    }

def generate_fallback_csv_report(state: PipelineGraphState) -> None:
    """Fallback CSV report generation."""
    analytics_results = state.get('analytics_results', {})
    data = []
    for symbol, metrics in analytics_results.items():
        metrics['symbol'] = symbol
        data.append(metrics)
        
    if data:
        df = pd.DataFrame(data)
        report_path = "results/reports"
        os.makedirs(report_path, exist_ok=True)
        report_filename = os.path.join(report_path, "fallback_report.csv")
        df.to_csv(report_filename, index=False)
        print(f"[OK] Fallback CSV report saved to {report_filename}")
