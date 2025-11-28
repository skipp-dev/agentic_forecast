#!/usr/bin/env python3
"""
Forecast Visualization Dashboard

Streamlit-based dashboard for visualizing forecast performance and analytics.
Provides interactive charts, tables, and real-time insights.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analytics.performance_reporting import PerformanceReporting
from data.metrics_database import MetricsDatabase

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Agentic Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .success-metric {
        color: #28a745;
        font-weight: bold;
    }
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
    }
    .danger-metric {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ForecastDashboard:
    """
    Streamlit dashboard for forecast visualization and analytics.
    """

    def __init__(self):
        """Initialize the dashboard."""
        self.metrics_db = MetricsDatabase()
        self.performance_reporter = PerformanceReporting()
        self.data_dir = Path("data/metrics")

        # Load data
        self.load_data()

        logger.info("Forecast Dashboard initialized")

    def load_data(self):
        """Load performance and forecast data."""
        try:
            # Load evaluation results
            eval_file = self.data_dir / "evaluation_results_baseline_latest.csv"
            if eval_file.exists():
                self.eval_results = pd.read_csv(eval_file)
                logger.info(f"Loaded {len(self.eval_results)} evaluation results")
            else:
                self.eval_results = pd.DataFrame()
                logger.warning("No evaluation results found")

            # Load performance snapshot
            snapshot_file = self.data_dir / "performance_snapshot.json"
            if snapshot_file.exists():
                with open(snapshot_file, 'r') as f:
                    self.performance_snapshot = json.load(f)
                logger.info("Loaded performance snapshot")
            else:
                self.performance_snapshot = {}
                logger.warning("No performance snapshot found")

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.eval_results = pd.DataFrame()
            self.performance_snapshot = {}

    def create_overview_metrics(self):
        """Create overview metrics cards."""
        if self.eval_results.empty:
            st.error("No evaluation data available")
            return

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_da = self.eval_results['directional_accuracy'].mean()
            st.metric(
                "Average Directional Accuracy",
                f"{avg_da:.1%}",
                help="Percentage of correct directional predictions"
            )

        with col2:
            avg_mae = self.eval_results['mae'].mean()
            st.metric(
                "Average MAE",
                f"{avg_mae:.4f}",
                help="Mean Absolute Error across all models"
            )

        with col3:
            total_evaluations = len(self.eval_results)
            st.metric(
                "Total Evaluations",
                f"{total_evaluations:,}",
                help="Number of model evaluations performed"
            )

        with col4:
            unique_symbols = self.eval_results['symbol'].nunique()
            st.metric(
                "Symbols Analyzed",
                f"{unique_symbols:,}",
                help="Number of unique symbols with forecasts"
            )

    def create_performance_charts(self):
        """Create performance visualization charts."""
        if self.eval_results.empty:
            st.error("No data available for charts")
            return

        st.subheader("📊 Model Performance Analytics")

        # Create tabs for different chart types
        tab1, tab2, tab3 = st.tabs(["Model Comparison", "Symbol Performance", "Forecast Horizons"])

        with tab1:
            self.create_model_comparison_chart()

        with tab2:
            self.create_symbol_performance_chart()

        with tab3:
            self.create_horizon_performance_chart()

    def create_model_comparison_chart(self):
        """Create model comparison chart."""
        # Group by model type and calculate averages
        model_perf = self.eval_results.groupby('model_type').agg({
            'directional_accuracy': 'mean',
            'mae': 'mean',
            'mape': 'mean'
        }).reset_index()

        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Directional Accuracy by Model', 'MAE by Model', 'MAPE by Model'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )

        # Directional Accuracy
        fig.add_trace(
            go.Bar(
                x=model_perf['model_type'],
                y=model_perf['directional_accuracy'],
                name='Directional Accuracy',
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        # MAE
        fig.add_trace(
            go.Bar(
                x=model_perf['model_type'],
                y=model_perf['mae'],
                name='MAE',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )

        # MAPE
        fig.add_trace(
            go.Bar(
                x=model_perf['model_type'],
                y=model_perf['mape'],
                name='MAPE',
                marker_color='lightgreen'
            ),
            row=1, col=3
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def create_symbol_performance_chart(self):
        """Create symbol performance chart."""
        # Get top 20 symbols by directional accuracy
        symbol_perf = self.eval_results.groupby('symbol')['directional_accuracy'].mean()
        top_symbols = symbol_perf.nlargest(20).reset_index()

        fig = px.bar(
            top_symbols,
            x='symbol',
            y='directional_accuracy',
            title='Top 20 Symbols by Directional Accuracy',
            labels={'directional_accuracy': 'Directional Accuracy', 'symbol': 'Symbol'},
            color='directional_accuracy',
            color_continuous_scale='Blues'
        )

        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Add MAE comparison
        symbol_mae = self.eval_results.groupby('symbol')['mae'].mean()
        top_symbols_mae = symbol_mae.nsmallest(20).reset_index()  # Lower MAE is better

        fig2 = px.bar(
            top_symbols_mae,
            x='symbol',
            y='mae',
            title='Top 20 Symbols by Lowest MAE',
            labels={'mae': 'Mean Absolute Error', 'symbol': 'Symbol'},
            color='mae',
            color_continuous_scale='Reds_r'  # Reverse scale so lower values are darker
        )

        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    def create_horizon_performance_chart(self):
        """Create forecast horizon performance chart."""
        horizon_perf = self.eval_results.groupby('target_horizon').agg({
            'directional_accuracy': 'mean',
            'mae': 'mean',
            'mape': 'mean'
        }).reset_index()

        # Create line chart
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=horizon_perf['target_horizon'],
            y=horizon_perf['directional_accuracy'],
            mode='lines+markers',
            name='Directional Accuracy',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))

        fig.add_trace(go.Scatter(
            x=horizon_perf['target_horizon'],
            y=horizon_perf['mae'],
            mode='lines+markers',
            name='MAE',
            line=dict(color='red', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))

        fig.update_layout(
            title='Performance by Forecast Horizon',
            xaxis=dict(title='Forecast Horizon (Days)'),
            yaxis=dict(title='Directional Accuracy', side='left'),
            yaxis2=dict(title='Mean Absolute Error', side='right', overlaying='y'),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def create_stock_outlook_section(self):
        """Create stock outlook analysis section."""
        st.subheader("🎯 Stock Outlook Analysis")

        # Filters
        col1, col2 = st.columns(2)

        with col1:
            selected_horizon = st.selectbox(
                "Forecast Horizon",
                options=sorted(self.eval_results['target_horizon'].unique()),
                index=0
            )

        with col2:
            top_n = st.slider("Number of Top Stocks", min_value=5, max_value=50, value=10)

        # Filter data
        horizon_data = self.eval_results[self.eval_results['target_horizon'] == selected_horizon].copy()

        if horizon_data.empty:
            st.warning(f"No data available for {selected_horizon}-day horizon")
            return

        # Calculate combined score
        horizon_data['combined_score'] = horizon_data['directional_accuracy'] - (horizon_data['mae'] * 100)

        # Get top stocks
        top_stocks = horizon_data.nlargest(top_n, 'combined_score')

        # Display table
        st.dataframe(
            top_stocks[['symbol', 'model_type', 'directional_accuracy', 'mae', 'combined_score']]
            .round(4)
            .rename(columns={
                'symbol': 'Symbol',
                'model_type': 'Model Type',
                'directional_accuracy': 'Directional Accuracy',
                'mae': 'MAE',
                'combined_score': 'Combined Score'
            }),
            use_container_width=True
        )

        # Create outlook chart
        fig = px.scatter(
            top_stocks,
            x='mae',
            y='directional_accuracy',
            size='combined_score',
            color='model_type',
            hover_name='symbol',
            title=f'Top {top_n} Stocks Outlook - {selected_horizon} Day Horizon',
            labels={
                'mae': 'Mean Absolute Error',
                'directional_accuracy': 'Directional Accuracy',
                'model_type': 'Model Type'
            }
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    def create_data_export_section(self):
        """Create data export section."""
        st.subheader("📥 Data Export")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Export Performance Summary CSV"):
                self.export_performance_summary()
                st.success("Performance summary exported to data/metrics/")

        with col2:
            if st.button("📈 Export Top Stocks Analysis CSV"):
                self.export_top_stocks_analysis()
                st.success("Top stocks analysis exported to data/metrics/")

    def export_performance_summary(self):
        """Export performance summary to CSV."""
        if self.eval_results.empty:
            st.error("No data to export")
            return

        summary = self.eval_results.groupby(['model_type', 'target_horizon']).agg({
            'directional_accuracy': ['mean', 'std', 'count'],
            'mae': ['mean', 'std'],
            'mape': ['mean', 'std']
        }).round(4)

        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_summary_{timestamp}.csv"
        filepath = self.data_dir / filename

        summary.to_csv(filepath)
        st.info(f"Exported to: {filepath}")

    def export_top_stocks_analysis(self):
        """Export top stocks analysis to CSV."""
        if self.eval_results.empty:
            st.error("No data to export")
            return

        # Get top stocks for each horizon
        horizons = sorted(self.eval_results['target_horizon'].unique())
        top_stocks_all = []

        for horizon in horizons:
            horizon_data = self.eval_results[self.eval_results['target_horizon'] == horizon].copy()
            horizon_data['combined_score'] = horizon_data['directional_accuracy'] - (horizon_data['mae'] * 100)
            top_stocks = horizon_data.nlargest(10, 'combined_score')
            top_stocks['horizon'] = horizon
            top_stocks_all.append(top_stocks)

        if top_stocks_all:
            combined_df = pd.concat(top_stocks_all, ignore_index=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"top_stocks_analysis_{timestamp}.csv"
            filepath = self.data_dir / filename

            combined_df.to_csv(filepath, index=False)
            st.info(f"Exported to: {filepath}")

    def run_dashboard(self):
        """Run the Streamlit dashboard."""
        st.markdown('<h1 class="main-header">📊 Agentic Forecast Dashboard</h1>', unsafe_allow_html=True)

        # Sidebar
        st.sidebar.title("🎛️ Dashboard Controls")

        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            self.load_data()
            st.sidebar.success("Data refreshed!")

        # Navigation
        page = st.sidebar.radio(
            "Navigate to:",
            ["Overview", "Performance Analysis", "Stock Outlook", "Data Export"]
        )

        # Main content
        if page == "Overview":
            st.header("📈 Overview")
            self.create_overview_metrics()

            st.subheader("🔍 Quick Insights")
            if not self.eval_results.empty:
                # Best performing model
                best_model = self.eval_results.loc[self.eval_results['directional_accuracy'].idxmax()]
                st.info(f"🏆 **Best Model:** {best_model['model_type']} for {best_model['symbol']} "
                       f"(DA: {best_model['directional_accuracy']:.1%})")

                # Most analyzed symbol
                symbol_counts = self.eval_results['symbol'].value_counts()
                most_analyzed = symbol_counts.index[0]
                st.info(f"📊 **Most Analyzed Symbol:** {most_analyzed} "
                       f"({symbol_counts[most_analyzed]} evaluations)")

        elif page == "Performance Analysis":
            self.create_performance_charts()

        elif page == "Stock Outlook":
            self.create_stock_outlook_section()

        elif page == "Data Export":
            self.create_data_export_section()

        # Footer
        st.markdown("---")
        st.markdown("*Dashboard last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")


def main():
    """Main entry point for the dashboard."""
    dashboard = ForecastDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()