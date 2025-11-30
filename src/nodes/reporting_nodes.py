import pandas as pd
from ..graphs.state import GraphState
import os
from typing import Dict, Any
from agents.llm_reporting_agent import LLMReportingAgent, ReportingInput


def generate_report_node(state: GraphState) -> GraphState:
    """
    Generates a comprehensive report using the LLMReportingAgent.
    Creates structured reports with analytics, HPO insights, and recommendations.
    """
    print("--- Node: Generate LLM-Powered Comprehensive Report ---")

    try:
        # Initialize the LLM reporting agent
        reporting_agent = LLMReportingAgent()

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

        print("✅ LLM-powered comprehensive report generated and stored")
        print(f"   📄 JSON: {metadata.get('json_path', 'N/A')}")
        print(f"   📝 Markdown: {metadata.get('markdown_path', 'N/A')}")
        print(f"   🌐 HTML: {metadata.get('html_path', 'N/A')}")
        print(f"   📧 Email sent: {metadata.get('email_sent', False)}")
        print(f"   🖥️ Dashboard updated: {metadata.get('dashboard_updated', False)}")

        # Store report metadata in state for downstream use
        state['report_metadata'] = metadata
        state['report_generated'] = True

        # Extract priority actions for continuous learning
        if hasattr(reporting_agent, '_last_report'):
            priority_actions = reporting_agent._last_report.get('priority_actions', [])
            state['priority_actions'] = priority_actions

            # Log priority actions for continuous learning
            if priority_actions:
                print("🎯 Priority Actions Identified:")
                for action in priority_actions[:3]:  # Show top 3
                    print(f"   • {action.get('action', 'Unknown')} (Priority: {action.get('priority', 'Unknown')})")

    except Exception as e:
        print(f"❌ Error generating LLM report: {e}")
        # Fallback to basic CSV report
        print("🔄 Falling back to basic CSV report...")
        generate_fallback_csv_report(state)

    return state


def extract_analytics_summary(state: GraphState) -> Dict[str, Any]:
    """Extract analytics summary from state."""
    analytics_df = state.get('analytics_summary', pd.DataFrame())

    if isinstance(analytics_df, pd.DataFrame) and not analytics_df.empty:
        # Convert DataFrame to summary dict
        summary = {
            'total_symbols': len(analytics_df),
            'avg_performance': analytics_df.get('performance_score', pd.Series()).mean() if 'performance_score' in analytics_df.columns else 0,
            'top_performers': analytics_df.nlargest(3, 'performance_score')['symbol'].tolist() if 'performance_score' in analytics_df.columns and 'symbol' in analytics_df.columns else [],
            'risk_metrics': {
                'avg_volatility': analytics_df.get('volatility', pd.Series()).mean() if 'volatility' in analytics_df.columns else 0,
                'high_risk_symbols': analytics_df[analytics_df.get('volatility', 0) > 0.3]['symbol'].tolist() if 'volatility' in analytics_df.columns and 'symbol' in analytics_df.columns else []
            }
        }
    else:
        summary = {
            'total_symbols': len(state.get('symbols', [])),
            'avg_performance': 0,
            'top_performers': [],
            'risk_metrics': {'avg_volatility': 0, 'high_risk_symbols': []}
        }

    return summary


def extract_hpo_plan(state: GraphState) -> Dict[str, Any]:
    """Extract HPO plan from state."""
    hpo_results = state.get('hpo_results', {})
    hpo_decision = state.get('hpo_decision', {})

    plan = {
        'hpo_completed': bool(hpo_results),
        'symbols_optimized': list(hpo_results.keys()) if hpo_results else [],
        'total_optimizations': sum(len(results) for results in hpo_results.values()) if hpo_results else 0,
        'best_improvements': [],
        'next_priorities': hpo_decision.get('next_priorities', [])
    }

    # Extract best improvements
    for symbol, results in hpo_results.items():
        for model_family, result in results.items():
            if result and hasattr(result, 'best_val_mape'):
                improvement = {
                    'symbol': symbol,
                    'model_family': model_family,
                    'best_mape': result.best_val_mape,
                    'improvement_pct': result.improvement_pct if hasattr(result, 'improvement_pct') else 0
                }
                plan['best_improvements'].append(improvement)

    # Sort by improvement percentage
    plan['best_improvements'].sort(key=lambda x: x.get('improvement_pct', 0), reverse=True)

    return plan


def extract_research_insights(state: GraphState) -> Dict[str, Any]:
    """Extract research insights from state."""
    shap_results = state.get('shap_results', {})
    anomalies = state.get('anomalies', {})

    insights = {
        'feature_importance': {},
        'anomalies_detected': len(anomalies),
        'key_findings': [],
        'market_implications': 'Analysis completed with SHAP insights and anomaly detection',
        'data_suggestions': []
    }

    # Extract SHAP insights
    if shap_results:
        for symbol, shap_data in shap_results.items():
            if isinstance(shap_data, dict) and 'feature_importance' in shap_data:
                insights['feature_importance'][symbol] = shap_data['feature_importance']

        insights['key_findings'].append(f"SHAP analysis completed for {len(shap_results)} symbols")

    # Add anomaly insights
    if anomalies:
        insights['key_findings'].append(f"Detected {len(anomalies)} anomalous patterns requiring attention")

    return insights


def extract_guardrail_status(state: GraphState) -> Dict[str, Any]:
    """Extract guardrail status from state."""
    guardrail_log = state.get('guardrail_log', [])
    risk_kpis = state.get('risk_kpis', pd.DataFrame())

    status = {
        'total_checks': len(guardrail_log),
        'passed_checks': sum(1 for log in guardrail_log if log.get('status') == 'passed'),
        'failed_checks': sum(1 for log in guardrail_log if log.get('status') == 'failed'),
        'warnings': sum(1 for log in guardrail_log if log.get('status') == 'warning'),
        'critical_issues': [log for log in guardrail_log if log.get('severity') == 'critical'],
        'can_proceed': all(log.get('status') != 'failed' for log in guardrail_log)
    }

    # Add risk KPI summary
    if isinstance(risk_kpis, pd.DataFrame) and not risk_kpis.empty:
        status['risk_summary'] = {
            'avg_var': risk_kpis.get('var_95', pd.Series()).mean() if 'var_95' in risk_kpis.columns else 0,
            'max_drawdown': risk_kpis.get('max_drawdown', pd.Series()).max() if 'max_drawdown' in risk_kpis.columns else 0,
            'high_risk_symbols': risk_kpis[risk_kpis.get('var_95', 0) > 0.05]['symbol'].tolist() if 'var_95' in risk_kpis.columns and 'symbol' in risk_kpis.columns else []
        }

    return status


def extract_run_metadata(state: GraphState) -> Dict[str, Any]:
    """Extract run metadata from state."""
    return {
        'run_type': state.get('run_type', 'DAILY'),
        'symbols_processed': len(state.get('symbols', [])),
        'timestamp': pd.Timestamp.now().isoformat(),
        'execution_time': 'N/A',  # Could be calculated if start time is tracked
        'models_trained': len(state.get('best_models', {})),
        'features_generated': len(state.get('features', {})),
        'forecasts_generated': len(state.get('forecasts', {})),
        'errors_encountered': len(state.get('errors', [])),
        'actions_executed': len(state.get('executed_actions', []))
    }


def generate_fallback_csv_report(state: GraphState) -> None:
    """Fallback CSV report generation when LLM reporting fails."""
    hpo_results = state.get('hpo_results', {})
    all_performance_data = []

    for symbol, results in hpo_results.items():
        for model_family, result in results.items():
            if result:
                performance_data = {
                    'symbol': symbol,
                    'model_family': model_family,
                    'mape': result.best_val_mape,
                    'mae': result.best_val_mae,
                    'model_id': result.best_model_id,
                    'artifact_path': result.artifact_info.artifact_uri
                }
                all_performance_data.append(performance_data)

    if all_performance_data:
        report_df = pd.DataFrame(all_performance_data)

        # Save report to disk
        report_path = "results/reports"
        os.makedirs(report_path, exist_ok=True)
        report_filename = os.path.join(report_path, "fallback_model_evaluation_report.csv")

        report_df.to_csv(report_filename, index=False)

        print(f"✅ Fallback CSV performance report saved to {report_filename}")

        # For display, show the top 5 models by MAPE
        print("\nTop 5 Performing Models (by MAPE):")
        print(report_df.sort_values(by='mape').head(5))
    else:
        print("⚠️ No performance data available to generate a report.")


def apply_continuous_learning_feedback(priority_actions: list, state: GraphState) -> Dict[str, Any]:
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
                state['drift_detected'] = True  # Trigger retraining loop

            elif 'hpo' in action_type or 'optimization' in action_type:
                feedback['actions_triggered'].append(f"High-priority HPO triggered: {action_type}")
                feedback['decisions_updated'].append({'type': 'hpo', 'symbol': action.get('owner', 'all')})
                state['hpo_triggered'] = True  # Trigger HPO loop

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

    return feedback
