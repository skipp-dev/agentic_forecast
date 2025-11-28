#!/usr/bin/env python3
"""
Demo script showing the complete metric sanity reporting pipeline.

This demonstrates the end-to-end flow:
1. QualityAgent generates sanity reports (JSON + Markdown)
2. LangGraph node calls ReportingLLM for explanations
3. MCP tool provides interactive access
"""

import json
from pathlib import Path
from agents.quality_agent import QualityAssuranceAgent
from metric_sanity_explainer import metric_sanity_explainer_node, GraphState
from mcp_tools import handle_metric_sanity_explainer


def demo_pipeline():
    """Run the complete metric sanity pipeline demo."""
    print("🚀 Starting Metric Sanity Reporting Pipeline Demo")
    print("=" * 60)

    # Step 1: Generate sanity reports with QualityAgent
    print("\n📊 Step 1: Generating metric sanity reports...")
    agent = QualityAssuranceAgent()

    try:
        result = agent.run_metric_sanity_report()
        print("✅ JSON and Markdown reports generated!")
        print(f"   📄 JSON: results/quality/metric_sanity_latest.json")
        print(f"   📝 MD: results/quality/metric_sanity_latest.md")
        print(f"   📊 Status: {result['overall_status']['status']} ({result['overall_status']['severity']})")
        print(f"   🔍 Issues: {result['overall_status']['issue_count']}")

    except Exception as e:
        print(f"❌ Failed to generate reports: {e}")
        return

    # Step 2: LangGraph node processes the report
    print("\n🤖 Step 2: LangGraph node calling ReportingLLM...")

    state = GraphState(
        run_id=result['run_metadata']['run_id'],
        metric_sanity_report_path="results/quality/metric_sanity_latest.json",
        metric_sanity_summary_path="results/quality/metric_sanity_summary.json"
    )

    result_state = metric_sanity_explainer_node(state)
    print("✅ LLM analysis complete!")
    print(f"   📄 Summary: results/quality/metric_sanity_summary.json")
    print(f"   📊 Status: {result_state.get('metric_sanity_status')}")
    print(f"   🔍 Issues: {result_state.get('metric_sanity_issue_count')}")

    # Step 3: MCP tool for interactive access
    print("\n🔧 Step 3: MCP tool demonstration...")

    # Test structured mode
    structured_result = handle_metric_sanity_explainer({"mode": "structured"})
    print("✅ Structured mode result:")
    summary = structured_result["content"]
    print(f"   📊 Status: {summary.get('status_summary', 'N/A')[:60]}...")
    print(f"   🔍 Findings: {len(summary.get('key_findings', []))} items")
    print(f"   🎯 Actions: {len(summary.get('recommended_actions', []))} items")

    # Test executive mode
    executive_result = handle_metric_sanity_explainer({"mode": "executive"})
    print("✅ Executive mode result:")
    markdown = executive_result["content"].get("markdown", "")
    print(f"   📝 Length: {len(markdown)} characters")
    print(f"   📋 Preview: {markdown[:100]}...")

    print("\n" + "=" * 60)
    print("🎉 Pipeline demo complete!")
    print("\n📋 Generated files:")
    print("   • results/quality/metric_sanity_latest.json")
    print("   • results/quality/metric_sanity_latest.md")
    print("   • results/quality/metric_sanity_summary.json")
    print("\n🔄 In production, this would be:")
    print("   • Called automatically after evaluation")
    print("   • Feed into Grafana dashboards")
    print("   • Trigger alerts for critical issues")
    print("   • Provide LLM explanations for stakeholders")


if __name__ == "__main__":
    demo_pipeline()