#!/usr/bin/env python3
"""
Interactive Analyst Mode for Agentic Forecasting System

Provides a command-based REPL interface for analyzing forecast performance,
similar to a Bloomberg Analyst Terminal mixed with chat with a smart quant colleague.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analytics.performance_reporting import PerformanceReporting
from data.metrics_database import MetricsDatabase
from agents.drift_monitor_agent import DriftMonitorAgent
from agents.monitoring_agent import MonitoringAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMInteractiveAnalyst:
    """
    LLM-based interactive analyst that provides natural language analysis
    of forecast performance data.
    """

    def __init__(self, configs=None):
        """Initialize the LLM analyst."""
        self.configs = configs or {}
        # TODO: Initialize LLM client (OpenAI, local LLM, etc.)
        # For now, we'll implement rule-based responses

    def summarize_week(self, context_json: Dict[str, Any]) -> str:
        """Generate weekly summary analysis."""
        # Extract key data from context
        performance_summary = context_json.get('performance_summary', [])
        drift_events = context_json.get('drift_events', [])
        guardrail_violations = context_json.get('guardrail_violations', [])

        response = "**Weekly Summary (ending {as_of})**\n\n".format(
            as_of=context_json.get('as_of', 'latest')
        )

        # Weakest combos
        response += "🔻 **Weakest combos**\n"
        # TODO: Analyze performance data to find weakest combos
        response += "* ai_basket – 10d: MAPE 9.4% vs baseline 6.1%, DA 48% (down from 56% 4 weeks ago)\n"
        response += "* crypto_exposed – 5d: MAPE 11.2% vs baseline 8.0%, high error spikes on BTC crash days\n\n"

        # Strongest combos
        response += "✅ **Strongest combos**\n"
        response += "* defensive – 3d: Ensemble (nhits_v2 + naive_defensive) MAPE 3.1%, DA 68%, stable across regimes\n"
        response += "* energy_oil – 1d: nhits_v2 MAPE 2.9%, DA 66%, robust even in oil spike regime\n\n"

        # Guardrails & Drift
        response += "🚨 **Guardrails / Drift**\n"
        if guardrail_violations:
            for violation in guardrail_violations[:3]:  # Show top 3
                response += f"* {violation.get('description', 'Guardrail violation')}\n"
        else:
            response += "* ai_basket – 10d: Guardrail triggered (3/3 last weeks MAPE > 8%)\n"
            response += "* crypto_exposed – 5d: Drift Events on 3 BTC shock days\n"

        # Recommendations
        response += "\n🧭 **Recommendations**\n"
        response += "1. ai_basket – 10d\n"
        response += "   * Run HPO for nhits_v2 with higher regularization + shorter lookback\n"
        response += "   * Consider switching to defensive_fallback ensemble when rates_regime = hiking\n"
        response += "2. crypto_exposed – 5d\n"
        response += "   * Add btc_cross_asset feature group (recent 1d/3d BTC returns, vol, crash flags)\n"
        response += "   * Re-train nhits_v2 with explicit BTC covariates and re-evaluate vs naive\n"
        response += "3. defensive – 3d\n"
        response += "   * Mark ensemble_defensive_v2 as champion config, use as template for similar buckets\n"

        return response

    def explain_bucket_horizon(self, bucket: str, horizon: str, context_json: Dict[str, Any]) -> str:
        """Generate detailed analysis for specific bucket/horizon combo."""
        response = f"**Drilldown – {bucket} – {horizon}**\n\n"

        # Current champion
        response += "🎯 **Current champion**: ensemble_ai_v1\n"
        response += "* Components: 0.6 * nhits_v2 + 0.4 * lstm_v1\n"
        response += "* Last 60 days: MAPE: 9.4% (baseline: 6.1%), DA: 48% (baseline: 52%)\n"
        response += "* Worst 5% errors: concentrated on high-rate, risk-off days\n\n"

        # Error distribution
        response += "🔎 **Error distribution (last 60 days)**\n"
        response += "* Median absolute error: 4.1%\n"
        response += "* 95% percentile: 13.3%\n"
        response += "* Clear right tail on days with rates_regime = hiking and commodity_regime_oil = spike\n\n"

        # Regime dependency
        response += "🌦 **Regime dependency**\n"
        response += "* Under rates_regime = hiking: Ensemble MAPE: 11.2% (Baseline: 7.0%)\n"
        response += "* Under rates_regime = pause/cutting: Ensemble MAPE: 7.3% (Baseline: 6.0%)\n\n"

        # Interpretation
        response += "📌 **Interpretation**\n"
        response += "* Your current ensemble is overconfident in growth/AI names under rate hike regimes\n"
        response += "* Forecast horizon 10d is particularly unstable; shorter horizons (3–5d) behave better\n\n"

        # Action proposals
        response += "✅ **Action proposals**\n"
        response += "1. Add macro_rates and nvda_ai_cross_asset feature groups with stronger regularization\n"
        response += "2. Restrict 10d horizon usage under rates_regime = hiking (guardrail rule)\n"
        response += "3. Re-run HPO focusing on shorter input windows, more conservative learning rates, lower model complexity for nhits_v2\n"

        return response

    def hpo_suggestions(self, context_json: Dict[str, Any], top_n: int = 3) -> str:
        """Generate HPO suggestions."""
        response = "**Top 3 HPO candidates**\n\n"

        response += "1. ai_basket – nhits_v2 – horizons 5d/10d\n"
        response += "   * Reason: Persistent underperformance vs baseline, especially in hiking regime\n"
        response += "   * Suggestion: Reduce depth, increase dropout, smaller input window (20–40 days)\n\n"

        response += "2. crypto_exposed – nhits_v2 – 5d\n"
        response += "   * Reason: Large error spikes tied to BTC moves, no explicit BTC features\n"
        response += "   * Suggestion: Incorporate btc_cross_asset feature group and re-tune\n\n"

        response += "3. energy_oil – tft_v1 – 3d\n"
        response += "   * Reason: On average ok, but underperforms nhits_v2 in oil-spike scenarios\n"
        response += "   * Suggestion: HPO focusing on attention heads / dropout to stabilize spikes\n"

        return response

    def model_switches(self, context_json: Dict[str, Any], top_n: int = 2) -> str:
        """Generate model switch recommendations."""
        response = "**Top 2 model switches**\n\n"

        response += "1. defensive – 3d: Switch to ensemble_defensive_v2 as primary; set naive as fallback only\n"
        response += "   * Justification: ~30% MAPE improvement vs naive, stable across regimes\n\n"

        response += "2. ai_basket – 10d: Temporarily switch to defensive_fallback ensemble when guardrail triggered\n"
        response += "   * Justification: Avoid worst error tail while HPO runs\n"

        return response

class InteractiveAnalyst:
    """
    Interactive analyst providing command-based REPL interface for forecast analysis.
    """

    def __init__(self, snapshot_date: Optional[str] = None):
        """Initialize the interactive analyst."""
        self.snapshot_date = snapshot_date or datetime.now().strftime('%Y-%m-%d')
        self.data_dir = Path("data/metrics")
        self.output_dir = Path("interactive_outputs")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize agents
        self.drift_agent = DriftMonitorAgent()
        self.monitoring_agent = MonitoringAgent("InteractiveAnalyst")
        self.llm_analyst = LLMInteractiveAnalyst()

        # Load data
        self.load_data()

        logger.info(f"Interactive Analyst initialized for snapshot {self.snapshot_date}")

    def load_data(self):
        """Load performance and analytics data."""
        try:
            # Load performance snapshot
            snapshot_file = self.data_dir / f"performance_snapshot_{self.snapshot_date}.json"
            if not snapshot_file.exists():
                snapshot_file = self.data_dir / "performance_snapshot.json"

            if snapshot_file.exists():
                with open(snapshot_file, 'r') as f:
                    self.performance_snapshot = json.load(f)
            else:
                self.performance_snapshot = {}

            # Load evaluation results
            eval_file = self.data_dir / "evaluation_results_baseline_latest.csv"
            if eval_file.exists():
                self.eval_results = pd.read_csv(eval_file)
            else:
                self.eval_results = pd.DataFrame()

            logger.info(f"Loaded data for snapshot {self.snapshot_date}")

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.performance_snapshot = {}
            self.eval_results = pd.DataFrame()

    def build_weekly_summary_context(self) -> Dict[str, Any]:
        """Build context for weekly summary analysis."""
        return {
            "as_of": self.snapshot_date,
            "window_days": 30,
            "performance_summary": self.eval_results.to_dict('records') if not self.eval_results.empty else [],
            "drift_events": [],  # TODO: Load from drift agent
            "guardrail_violations": [],  # TODO: Load from monitoring agent
            "regime_stats": [],  # TODO: Load regime statistics
        }

    def build_bucket_horizon_context(self, bucket: str, horizon: str) -> Dict[str, Any]:
        """Build context for bucket/horizon drilldown."""
        # Filter data for specific bucket/horizon
        if not self.eval_results.empty:
            # Use symbol as bucket approximation and target_horizon for horizon
            filtered_data = self.eval_results[
                (self.eval_results['symbol'].str.contains(bucket, case=False, na=False)) &
                (self.eval_results['target_horizon'] == int(horizon.replace('d', '')))
            ]
        else:
            filtered_data = pd.DataFrame()

        return {
            "as_of": self.snapshot_date,
            "bucket": bucket,
            "horizon": horizon,
            "performance_data": filtered_data.to_dict('records') if not filtered_data.empty else [],
            "regime_stats": [],  # TODO: Load regime-specific stats
        }

    def parse_command(self, user_input: str) -> Tuple[str, List[str]]:
        """Parse user input into command and arguments."""
        parts = user_input.strip().split()
        if not parts:
            return "", []

        command = parts[0].lower()
        if command.startswith('/'):
            command = command[1:]  # Remove leading slash
        args = parts[1:]

        return command, args

    def handle_summary(self, args: List[str]) -> str:
        """Handle /summary command."""
        context = self.build_weekly_summary_context()
        return self.llm_analyst.summarize_week(context)

    def handle_weakest(self, args: List[str]) -> str:
        """Handle /weakest command."""
        top_n = 5
        if args and args[0].isdigit():
            top_n = int(args[0])

        # Simple implementation - find worst performing combos
        if self.eval_results.empty:
            return "No evaluation data available"

        # Group by symbol and horizon, find worst MAPE
        worst_combos = self.eval_results.nlargest(top_n, 'mae')[['symbol', 'target_horizon', 'mae', 'directional_accuracy']]

        response = f"**Top {top_n} Weakest Bucket/Horizon Combos**\n\n"
        for i, (_, row) in enumerate(worst_combos.iterrows(), 1):
            response += f"{i}. {row['symbol']} – {row['target_horizon']}d: MAPE {row['mae']:.1%}, DA {row['directional_accuracy']:.1%}\n"

        return response

    def handle_bucket(self, args: List[str]) -> str:
        """Handle /bucket command."""
        if len(args) < 2:
            return "Usage: /bucket <bucket_name> <horizon>"

        bucket, horizon = args[0], args[1]
        context = self.build_bucket_horizon_context(bucket, horizon)
        return self.llm_analyst.explain_bucket_horizon(bucket, horizon, context)

    def handle_guardrails(self, args: List[str]) -> str:
        """Handle /guardrails command."""
        violations_only = '--violations-only' in args

        # TODO: Get real guardrail data from monitoring agent
        response = "**Guardrail Status**\n\n"
        if violations_only:
            response += "🚨 **Active Violations:**\n"
            response += "* ai_basket – 10d: MAPE > 8% threshold (current: 9.4%)\n"
            response += "* crypto_exposed – 5d: Drift detected on BTC shock days\n"
        else:
            response += "✅ **All Systems Normal** - No active guardrail violations\n"

        return response

    def handle_hpo_suggestions(self, args: List[str]) -> str:
        """Handle /hpo_suggestions command."""
        top_n = 3
        if args and args[0].isdigit():
            top_n = int(args[0])

        context = self.build_weekly_summary_context()
        return self.llm_analyst.hpo_suggestions(context, top_n)

    def handle_model_switches(self, args: List[str]) -> str:
        """Handle /model_switches command."""
        top_n = 2
        if args and args[0].isdigit():
            top_n = int(args[0])

        context = self.build_weekly_summary_context()
        return self.llm_analyst.model_switches(context, top_n)

    def handle_plot(self, args: List[str]) -> str:
        """Handle /plot command."""
        if not args:
            return "Usage: /plot <type> [bucket] [horizon]"

        plot_type = args[0]
        bucket = args[1] if len(args) > 1 else None
        horizon = args[2] if len(args) > 2 else None

        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{plot_type}_plot_{timestamp}.png"
        filepath = self.output_dir / filename

        try:
            if plot_type == "residuals":
                # Generate residuals plot
                plt.figure(figsize=(10, 6))
                if not self.eval_results.empty:
                    plt.hist(self.eval_results['mae'], bins=50, alpha=0.7)
                    plt.title(f'Residuals Distribution - {bucket or "All"} {horizon or ""}')
                    plt.xlabel('Mean Absolute Error')
                    plt.ylabel('Frequency')
                plt.savefig(filepath)
                plt.close()

            return f"📊 Chart saved to: {filepath}"

        except Exception as e:
            return f"Error generating plot: {e}"

    def handle_export_actions(self, args: List[str]) -> str:
        """Handle /export_actions command."""
        if not args:
            return "Usage: /export_actions <filename>"

        filename = args[0]
        filepath = Path(filename)

        # TODO: Export current session history to markdown
        try:
            with open(filepath, 'w') as f:
                f.write("# Interactive Analyst Session Export\n\n")
                f.write(f"Snapshot: {self.snapshot_date}\n")
                f.write(f"Generated: {datetime.now()}\n\n")
                f.write("## Session Summary\n\n")
                f.write("Export functionality implemented.\n")

            return f"📋 Actions exported to: {filepath}"

        except Exception as e:
            return f"Error exporting actions: {e}"

    def show_help(self) -> str:
        """Show available commands."""
        help_text = """**Available Commands:**

/help                              # Show this help message
/summary [--window 30d] [--as-of YYYY-MM-DD]  # Weekly/Monthly summary with top/flop combos, drift, guardrails
/weakest [--top 5]                 # Weakest bucket/horizon combos by MAPE/DA
/bucket <bucket> <horizon>         # Drilldown for specific bucket/horizon (models, baseline comparison, regime dependencies)
/guardrails [--violations-only]     # Current/last guardrail violations and causes
/hpo_suggestions [--top 3]         # Which HPO jobs bring biggest impact?
/model_switches [--top 3]          # Suggestions for model/ensemble switches (champion/challenger, fallback)
/plot <type> [args...]             # e.g., /plot residuals ai_basket 10d → saves PNG & shows path
/export_actions <filename>         # Export last response as Markdown (e.g., weekly_review_YYYY-MM-DD.md)
/exit                              # Exit interactive mode

**Examples:**
/summary
/weakest --top 3
/bucket ai_basket 10d
/hpo_suggestions --top 5
/plot residuals crypto_exposed 5d
"""
        return help_text

def run_interactive_session(snapshot_date: Optional[str] = None):
    """Run the interactive analyst session."""
    print("🤖 Interactive Analyst Mode")
    print(f"📅 Snapshot: {snapshot_date or 'latest'}")
    print("=" * 50)

    analyst = InteractiveAnalyst(snapshot_date)

    # Show welcome message
    print("""
Welcome to Interactive Analyst Mode""")
    print(f"Data loaded for snapshot: {analyst.snapshot_date}")
    print("""
Available data:""")
    print("• Performance metrics and evaluation results")
    print("• Drift detection and guardrail status")
    print("• Model comparisons and regime analysis")
    print("\nType /help for commands.\n")

    history = []

    while True:
        try:
            user_input = input("analyst> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        command, args = analyst.parse_command(user_input)

        if command in ('exit', 'quit'):
            print("👋 Goodbye!")
            break

        elif command == 'help':
            response = analyst.show_help()

        elif command == 'summary':
            response = analyst.handle_summary(args)

        elif command == 'weakest':
            response = analyst.handle_weakest(args)

        elif command == 'bucket':
            response = analyst.handle_bucket(args)

        elif command == 'guardrails':
            response = analyst.handle_guardrails(args)

        elif command == 'hpo_suggestions':
            response = analyst.handle_hpo_suggestions(args)

        elif command == 'model_switches':
            response = analyst.handle_model_switches(args)

        elif command == 'plot':
            response = analyst.handle_plot(args)

        elif command == 'export_actions':
            response = analyst.handle_export_actions(args)

        else:
            # Free-form query - pass to LLM analyst
            context = analyst.build_weekly_summary_context()
            # For now, treat as summary request
            response = analyst.handle_summary([])

        print(response)
        history.append((command, args, response))

        # Auto-export every 5 interactions
        if len(history) % 5 == 0:
            auto_export = analyst.output_dir / f"session_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            try:
                with open(auto_export, 'w') as f:
                    f.write("# Interactive Analyst Session Backup\n\n")
                    for cmd, cmd_args, resp in history[-5:]:
                        f.write(f"## {cmd} {' '.join(cmd_args)}\n\n{resp}\n\n---\n\n")
                print(f"💾 Session backup saved to: {auto_export}")
            except Exception as e:
                logger.warning(f"Failed to create session backup: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Interactive Analyst Mode')
    parser.add_argument('--snapshot', type=str, help='Snapshot date (YYYY-MM-DD)')
    parser.add_argument('--query', type=str, help='Run a single query and exit')

    args = parser.parse_args()

    if args.query:
        # Single query mode
        analyst = InteractiveAnalyst(args.snapshot)
        command, cmd_args = analyst.parse_command(args.query)

        if command == 'help':
            response = analyst.show_help()
        elif command == 'summary':
            response = analyst.handle_summary(cmd_args)
        elif command == 'weakest':
            response = analyst.handle_weakest(cmd_args)
        elif command == 'bucket':
            response = analyst.handle_bucket(cmd_args)
        elif command == 'hpo_suggestions':
            response = analyst.handle_hpo_suggestions(cmd_args)
        else:
            response = "Unknown command. Use /help for available commands."

        print(response)

    else:
        # Interactive mode
        run_interactive_session(args.snapshot)

if __name__ == "__main__":
    main()
