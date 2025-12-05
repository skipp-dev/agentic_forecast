import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Agentic Forecast Backtest Dashboard", layout="wide")

st.title("Agentic Forecast Backtest Dashboard")

# Sidebar for configuration
st.sidebar.header("Configuration")
results_dir = st.sidebar.text_input("Results Directory", "backtest_results")

if not os.path.exists(results_dir):
    st.warning(f"Directory {results_dir} not found. Please run a backtest first.")
    st.stop()

# Load data
try:
    forecasts_path = os.path.join(results_dir, "forecasts.csv")
    performance_path = os.path.join(results_dir, "performance.csv")
    
    if os.path.exists(forecasts_path):
        forecasts_df = pd.read_csv(forecasts_path)
        forecasts_df['date'] = pd.to_datetime(forecasts_df['date'])
        if 'forecast_date' in forecasts_df.columns:
            forecasts_df['forecast_date'] = pd.to_datetime(forecasts_df['forecast_date'])
    else:
        forecasts_df = pd.DataFrame()
        
    if os.path.exists(performance_path):
        performance_df = pd.read_csv(performance_path)
        performance_df['date'] = pd.to_datetime(performance_df['date'])
    else:
        performance_df = pd.DataFrame()
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if forecasts_df.empty and performance_df.empty:
    st.warning("No data found in results directory.")
    st.stop()

# Overview
st.header("Overview")
col1, col2, col3 = st.columns(3)
if not forecasts_df.empty:
    col1.metric("Total Days", forecasts_df['date'].nunique())
    col2.metric("Symbols", forecasts_df['symbol'].nunique())
    col3.metric("Models", forecasts_df['model'].nunique())

# Performance Over Time
st.header("Performance Over Time")
if not performance_df.empty:
    metric = st.selectbox("Select Metric", ['mape', 'mae', 'rmse'], index=0)
    if metric in performance_df.columns:
        fig_perf = px.line(performance_df, x='date', y=metric, color='model_family', facet_col='symbol',
                           title=f"{metric.upper()} over Time by Model")
        st.plotly_chart(fig_perf, use_container_width=True)
    else:
        st.info(f"Metric {metric} not available in performance data.")

# Forecast Visualization
st.header("Forecast Visualization")
if not forecasts_df.empty:
    selected_symbol = st.selectbox("Select Symbol", forecasts_df['symbol'].unique())

    symbol_forecasts = forecasts_df[forecasts_df['symbol'] == selected_symbol]

    # Plot using forecast_date if available, else date
    x_col = 'forecast_date' if 'forecast_date' in symbol_forecasts.columns else 'date'
    
    fig_forecast = px.line(symbol_forecasts, x=x_col, y='forecast', color='model',
                           title=f"Forecasts for {selected_symbol}")

    st.plotly_chart(fig_forecast, use_container_width=True)

# Data Tables
st.header("Detailed Data")
with st.expander("Forecasts Data"):
    st.dataframe(forecasts_df)
with st.expander("Performance Data"):
    st.dataframe(performance_df)
