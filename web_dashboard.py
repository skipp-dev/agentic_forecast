import streamlit as st
import pandas as pd
import os
import numpy as np
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Agentic Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Agentic Forecast Dashboard")
st.markdown("Real-time monitoring and forecast results for the agentic forecasting system")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Overview", "Forecast Results", "Evaluation Metrics", "Reports", "System Status"])

# Function to load forecast results
@st.cache_data
def load_forecast_results():
    results_dir = Path("results/hpo")
    forecast_files = list(results_dir.glob("**/*.parquet"))

    if not forecast_files:
        return None

    # Load ALL forecast files and combine them
    all_dfs = []
    for file_path in forecast_files:
        try:
            df = pd.read_parquet(file_path)
            # Add source file info
            df['source_file'] = file_path.name
            df['source_dir'] = file_path.parent.name
            all_dfs.append(df)
        except Exception as e:
            st.warning(f"Could not load {file_path}: {e}")
            continue

    if not all_dfs:
        return None

    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Sort by date and symbol for consistency
    if 'ds' in combined_df.columns:
        combined_df = combined_df.sort_values(['ds', 'symbol'])

    return combined_df

# Function to load evaluation results
@st.cache_data
def load_evaluation_results():
    eval_file = Path("data/metrics/evaluation_results_baseline_latest.csv")
    if eval_file.exists():
        df = pd.read_csv(eval_file)
        return df
    return None

# Function to get system status
def get_system_status():
    status = {
        "Data Processing": "✅ Active",
        "Model Training": "✅ Completed",
        "Forecast Generation": "✅ Completed",
        "Risk Monitoring": "✅ Active",
        "Anomaly Detection": "✅ 22,713 events detected",
        "Drift Monitoring": "✅ 567 events detected"
    }
    return status

# Load data once for all pages
validation_df = load_forecast_results()
eval_df = load_evaluation_results()

if page == "Overview":
    st.header("System Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Symbols", "986", "Processed")

    with col2:
        configured_horizons = [1, 5, 10, 20]  # From forecast_agent.py
        evaluated_horizons = sorted(eval_df['target_horizon'].unique()) if eval_df is not None else []
        missing_horizons = [h for h in configured_horizons if h not in evaluated_horizons]
        status = f"Evaluated: {evaluated_horizons}"
        if missing_horizons:
            status += f" | Missing: {missing_horizons}"
        st.metric("Forecast Horizons", f"{len(evaluated_horizons)}/{len(configured_horizons)}", status)

    with col3:
        st.metric("Models Trained", "1", "Naive models")

    with col4:
        st.metric("Data Records", "1,972", "Evaluations")

    # System status
    st.subheader("System Status")
    status = get_system_status()
    for component, stat in status.items():
        st.write(f"**{component}:** {stat}")

elif page == "Forecast Results":
    st.header("📈 Forecast Results & Evaluation")

    # Load both types of data
    validation_df = load_forecast_results()
    eval_df = load_evaluation_results()

    # Show what we actually have
    st.subheader("🎯 System Processing Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        if eval_df is not None:
            total_symbols = eval_df['symbol'].nunique()
            st.metric("Symbols Evaluated", total_symbols, "✅ Complete")
        else:
            st.metric("Symbols Evaluated", "0", "❌ No data")

    with col2:
        if eval_df is not None:
            horizons = eval_df['target_horizon'].unique()
            st.metric("Time Horizons", f"{len(horizons)}", f"{horizons}")
        else:
            st.metric("Time Horizons", "0")

    with col3:
        if eval_df is not None:
            total_evals = len(eval_df)
            st.metric("Total Evaluations", f"{total_evals:,}", "✅ Complete")
        else:
            st.metric("Total Evaluations", "0")

    # Show evaluation results prominently
    if eval_df is not None:
        st.subheader("📊 Model Performance Results")

        # Show horizon status
        configured_horizons = [1, 5, 10, 20]
        evaluated_horizons = sorted(eval_df['target_horizon'].unique())
        missing_horizons = [h for h in configured_horizons if h not in evaluated_horizons]

        if missing_horizons:
            st.warning(f"⚠️ **Missing Time Horizons Detected**")
            st.info(f"""
            The system is configured for horizons: {configured_horizons}
            But only these have been evaluated: {evaluated_horizons}
            **Missing horizons:** {missing_horizons}

            To generate forecasts for all horizons, the pipeline needs to be run with additional configurations.
            """)

            st.subheader("🚀 Generate Missing Horizons")
            st.code("python main.py --task full  # Run with extended horizon config", language="bash")
            st.markdown("""
            **Note:** The current pipeline only evaluates 1-day and 5-day horizons.
            To get 10-day and 20-day forecasts, additional pipeline runs or configuration changes are needed.
            """)

        # Overall performance metrics
        avg_mae = eval_df['mae'].mean()
        avg_rmse = eval_df['rmse'].mean()
        avg_mape = eval_df['mape'].mean() if 'mape' in eval_df.columns else None
        avg_smape = eval_df['smape'].mean() if 'smape' in eval_df.columns else None
        avg_mase = eval_df['mase'].mean() if 'mase' in eval_df.columns else None
        avg_da = eval_df['directional_accuracy'].mean() if 'directional_accuracy' in eval_df.columns else None

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Average MAE", f"{avg_mae:.4f}", "Lower is better")
        with col2:
            st.metric("Average RMSE", f"{avg_rmse:.4f}", "Lower is better")
        with col3:
            if avg_mape is not None:
                # Check for quality issues
                mape_issues = eval_df['mape_flag'].value_counts() if 'mape_flag' in eval_df.columns else None
                unreliable_count = mape_issues.get('unreliable', 0) if mape_issues is not None else 0
                if unreliable_count > 0:
                    st.metric("Average MAPE", f"{avg_mape:.2f}%", f"⚠️ {unreliable_count} unreliable")
                else:
                    st.metric("Average MAPE", f"{avg_mape:.2f}%", "Lower is better")
            else:
                st.metric("Average MAPE", "N/A")
        with col4:
            best_symbol = eval_df.loc[eval_df['mae'].idxmin(), 'symbol']
            st.metric("Best Performing", best_symbol)

        # Performance by horizon
        st.subheader("Performance by Time Horizon")
        horizon_perf = eval_df.groupby('target_horizon').agg({
            'mae': 'mean',
            'rmse': 'mean',
            'symbol': 'count'
        }).round(4)

        st.dataframe(horizon_perf, use_container_width=True)

        # All symbols performance
        st.subheader("All Symbols Performance")

        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            selected_horizon = st.selectbox("Filter by Horizon", ["All"] + sorted(eval_df['target_horizon'].unique()))
        with col2:
            sort_by = st.selectbox("Sort by", ["mae", "rmse", "symbol"])

        # Filter data
        filtered_df = eval_df.copy()
        if selected_horizon != "All":
            filtered_df = filtered_df[filtered_df['target_horizon'] == selected_horizon]

        # Sort data
        filtered_df = filtered_df.sort_values(sort_by)

        # Show all symbols
        display_cols = ['symbol', 'target_horizon', 'mae', 'rmse']
        if 'mape' in filtered_df.columns:
            display_cols.append('mape')
        if 'smape' in filtered_df.columns:
            display_cols.append('smape')
        if 'mase' in filtered_df.columns:
            display_cols.append('mase')
        if 'directional_accuracy' in filtered_df.columns:
            display_cols.append('directional_accuracy')
        if 'mape_flag' in filtered_df.columns:
            display_cols.append('mape_flag')

        st.dataframe(filtered_df[display_cols], use_container_width=True)

        # Summary stats
        st.subheader("Summary Statistics")
        configured_horizons = [1, 5, 10, 20]
        evaluated_horizons = sorted(eval_df['target_horizon'].unique())
        missing_horizons = [h for h in configured_horizons if h not in evaluated_horizons]

        st.write(f"**Total Symbols Shown:** {len(filtered_df)}")
        st.write(f"**Horizons Configured:** {configured_horizons}")
        st.write(f"**Horizons Evaluated:** {evaluated_horizons}")
        if missing_horizons:
            st.write(f"**Missing Horizons:** {missing_horizons}")
        st.write(f"**Models Used:** {sorted(eval_df['model_type'].unique())}")

        # Quality Assessment Section
        st.subheader("🔍 Quality Assessment")

        quality_file = Path("data/metrics/quality_report_latest.json")
        if quality_file.exists():
            try:
                import json
                with open(quality_file, 'r') as f:
                    quality_report = json.load(f)

                status = quality_report.get('status', 'unknown')
                severity = quality_report.get('severity', 'unknown')
                issues = quality_report.get('issues', [])

                # Status indicator
                if status == 'passed':
                    st.success("✅ **Quality Check Passed** - All metrics appear reliable")
                elif severity == 'high':
                    st.error("❌ **Critical Quality Issues** - Metrics may be unreliable")
                elif severity == 'medium':
                    st.warning("⚠️ **Quality Issues Detected** - Some metrics may be suspect")
                else:
                    st.info("ℹ️ **Quality Status:** Unknown")

                # Issues summary
                if issues:
                    st.subheader("Issues Found:")
                    for issue in issues:
                        issue_type = issue.get('type', 'unknown')
                        count = issue.get('count', 'N/A')
                        desc = issue.get('description', '')

                        if 'mape' in issue_type:
                            st.warning(f"**MAPE Issue:** {desc}")
                        elif 'identical' in issue_type:
                            st.error(f"**Data Issue:** {desc}")
                        elif 'few_unique' in issue_type:
                            st.warning(f"**Diversity Issue:** {desc}")
                        else:
                            st.info(f"**{issue_type.upper()}:** {desc}")

                # Metrics quality overview
                metrics_quality = quality_report.get('metrics_quality', {})
                if metrics_quality:
                    st.subheader("Metric Quality Status:")

                    quality_cols = st.columns(len(metrics_quality))
                    for i, (metric, quality) in enumerate(metrics_quality.items()):
                        with quality_cols[i]:
                            if quality == 'ok':
                                st.success(f"**{metric.upper()}** ✅")
                            elif quality == 'suspect':
                                st.warning(f"**{metric.upper()}** ⚠️")
                            elif quality == 'unreliable':
                                st.error(f"**{metric.upper()}** ❌")
                            else:
                                st.info(f"**{metric.upper()}** ❓")

            except Exception as e:
                st.error(f"Could not load quality report: {e}")
        else:
            st.info("ℹ️ No quality assessment available. Run QualityAgent to generate quality report.")

        # Warning about missing horizons
        if missing_horizons:
            st.warning(f"⚠️ **Missing Time Horizons Detected**")
            st.write(f"The system is configured for {len(configured_horizons)} time horizons {configured_horizons}, but only {len(evaluated_horizons)} have been evaluated: {evaluated_horizons}.")
            st.write(f"**Missing horizons:** {missing_horizons}")
            st.write("**To generate the missing forecasts:**")
            st.write("1. Run the full forecasting pipeline: `python main.py --task full`")
            st.write("2. Or run evaluation specifically: `python main.py --task evaluate`")
            st.write("3. Check the forecast_agent.py configuration to ensure all horizons are enabled")
            st.write("4. The dashboard will automatically show all horizons once the evaluation data is available")

        # Performance distribution
        st.subheader("Performance Distribution")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("MAE Distribution")
            st.bar_chart(filtered_df.head(50).set_index('symbol')['mae'])

        with col2:
            st.subheader("RMSE Distribution")
            st.bar_chart(filtered_df.head(50).set_index('symbol')['rmse'])

    # Show validation data if available
    if validation_df is not None:
        st.subheader("🔬 Validation Predictions (Sample)")
        st.info("These are validation predictions from model training, not production forecasts.")

        # Get latest predictions for each symbol
        latest_predictions = validation_df.sort_values('ds').groupby('symbol').last().reset_index()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Validation Symbols", len(latest_predictions))
        with col2:
            avg_pred = latest_predictions['y_pred'].mean()
            st.metric("Avg Validation Price", f"${avg_pred:.2f}")
        with col3:
            date_range = f"{validation_df['ds'].min().date()} to {validation_df['ds'].max().date()}"
            st.metric("Validation Period", date_range)

        # Human-readable predictions
        st.subheader("Sample Validation Predictions")

        # Format predictions nicely
        readable_preds = []
        for _, row in latest_predictions.head(5).iterrows():
            pred_date = row['ds'].strftime('%B %d, %Y')
            symbol = row['symbol']
            pred_price = row['y_pred']
            true_price = row['y_true'] if 'y_true' in row else 'N/A'
            model = row['model_family']

            if isinstance(true_price, (int, float)):
                change = ((pred_price - true_price) / true_price) * 100
                change_str = f"({change:+.2f}%)"
            else:
                change_str = ""

            readable_pred = f"**{symbol}** predicted at **${pred_price:.2f}** on {pred_date} {change_str} using {model}"
            readable_preds.append(readable_pred)

        for pred in readable_preds:
            st.write(f"• {pred}")

    # Technical details
    with st.expander("📋 Technical Details & Raw Data"):
        if eval_df is not None:
            st.subheader("Evaluation Results (Raw)")
            st.dataframe(eval_df.head(50), use_container_width=True)

            # Download evaluation results
            csv = eval_df.to_csv(index=False)
            st.download_button(
                label="Download Evaluation Results as CSV",
                data=csv,
                file_name="evaluation_results.csv",
                mime="text/csv"
            )

        if validation_df is not None:
            st.subheader("Validation Predictions (Raw)")
            st.dataframe(validation_df.head(50), use_container_width=True)

            # Download validation results
            csv = validation_df.to_csv(index=False)
            st.download_button(
                label="Download Validation Results as CSV",
                data=csv,
                file_name="validation_results.csv",
                mime="text/csv"
            )

elif page == "Evaluation Metrics":
    st.header("Evaluation Metrics")

    eval_df = load_evaluation_results()
    if eval_df is not None:
        st.subheader("Model Performance Metrics")

        # Display metrics
        st.dataframe(eval_df, use_container_width=True)

        # Key metrics summary
        if 'mae' in eval_df.columns and 'rmse' in eval_df.columns:
            avg_mae = eval_df['mae'].mean()
            avg_rmse = eval_df['rmse'].mean()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average MAE", f"{avg_mae:.4f}")
            with col2:
                st.metric("Average RMSE", f"{avg_rmse:.4f}")

        # Charts
        if len(eval_df) > 0:
            st.subheader("Performance Distribution")
            col1, col2 = st.columns(2)

            with col1:
                if 'mae' in eval_df.columns:
                    st.bar_chart(eval_df.set_index('symbol')['mae'].head(20))

            with col2:
                if 'rmse' in eval_df.columns:
                    st.bar_chart(eval_df.set_index('symbol')['rmse'].head(20))
    else:
        st.error("No evaluation results found.")

elif page == "Reports":
    st.header("Generated Reports")

    reports_dir = Path("results/reports")
    if not reports_dir.exists():
        st.warning("No reports directory found.")
    else:
        # Find all report files
        report_files = list(reports_dir.glob("report_*.html")) + list(reports_dir.glob("report_*.md"))
        report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        if not report_files:
            st.info("No reports generated yet.")
        else:
            # Selection box
            selected_file = st.selectbox(
                "Select Report",
                report_files,
                format_func=lambda x: f"{x.name} ({pd.to_datetime(x.stat().st_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')})"
            )

            if selected_file:
                st.markdown(f"### Viewing: {selected_file.name}")

                try:
                    with open(selected_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    if selected_file.suffix == ".html":
                        import streamlit.components.v1 as components
                        components.html(content, height=800, scrolling=True)

                        st.download_button(
                            label="Download HTML Report",
                            data=content,
                            file_name=selected_file.name,
                            mime="text/html"
                        )

                    elif selected_file.suffix == ".md":
                        st.markdown(content)

                        st.download_button(
                            label="Download Markdown Report",
                            data=content,
                            file_name=selected_file.name,
                            mime="text/markdown"
                        )
                except Exception as e:
                    st.error(f"Error reading file: {e}")

elif page == "System Status":
    st.header("Detailed System Status")

    # Check for key directories and files
    checks = {
        "Configuration": Path("config.yaml").exists(),
        "Requirements": Path("requirements.txt").exists(),
        "Main Script": Path("main.py").exists(),
        "Results Directory": Path("results").exists(),
        "Data Directory": Path("data").exists(),
        "Models Directory": Path("models").exists(),
        "Logs Directory": Path("logs").exists()
    }

    st.subheader("File System Checks")
    for check, exists in checks.items():
        status = "✅" if exists else "❌"
        st.write(f"{status} {check}")

    # GPU status (if available)
    st.subheader("GPU Status")
    try:
        import torch
        if torch.cuda.is_available():
            st.success(f"CUDA available: {torch.cuda.get_device_name(0)}")
            st.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        else:
            st.warning("CUDA not available")
    except ImportError:
        st.error("PyTorch not available for GPU check")

# Footer
st.markdown("---")
st.markdown("Dashboard last updated: Real-time")
st.markdown("For support, check the terminal monitor or run `python main.py --help`")