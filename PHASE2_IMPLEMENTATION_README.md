# Agentic Forecasting System - Phase 2 Implementation

This document outlines the implementation of Phase 2 enhancements to the agentic forecasting system, including individual component runners, macro economic data integration, regime detection, and automated scheduling.

## Overview

Phase 2 builds upon the Phase 1 foundation with advanced features for production deployment:

- **Individual Component Runners**: Modular scripts for independent pipeline component execution
- **Macro Economic Data**: Integration of economic indicators and market regime analysis
- **Regime-Aware Strategies**: Dynamic strategy selection based on market conditions
- **Automated Scheduling**: Production-ready daily pipeline automation
- **Performance Monitoring**: Comprehensive performance analysis and optimization

## Architecture

### Component Structure

```
scripts/
├── run_data_ingestion.py          # Independent data ingestion
├── run_feature_engineering.py     # Independent feature engineering
├── run_model_training.py          # Independent model training/forecasting
├── run_monitoring.py              # Independent monitoring
├── run_phase2_pipeline.py         # Phase 2 macro/regime/strategy pipeline
├── performance_monitoring.py      # Performance analysis and optimization
├── daily_pipeline.ps1             # Automated daily pipeline (PowerShell)
└── setup_task_scheduler.ps1       # Windows Task Scheduler setup

agents/
├── macro_data_agent.py            # Macro economic data collection
├── regime_detection_agent.py      # Market regime detection
└── strategy_selection_agent.py    # Regime-aware strategy selection
```

## Individual Component Runners

Each pipeline component can now be run independently with proper error handling and logging.

### Data Ingestion Runner

```bash
# Run data ingestion for all symbols
python scripts/run_data_ingestion.py

# Run with custom date range
python scripts/run_data_ingestion.py --start-date 2023-01-01 --end-date 2024-01-01

# Run for specific symbols
python scripts/run_data_ingestion.py --symbols AAPL MSFT GOOGL

# Save output for next component
python scripts/run_data_ingestion.py --output data/raw_data.pkl
```

### Feature Engineering Runner

```bash
# Process features from saved data
python scripts/run_feature_engineering.py --input data/raw_data.pkl --output data/features.pkl
```

### Model Training Runner

```bash
# Train models and generate forecasts
python scripts/run_model_training.py --input data/features.pkl --output models/forecasts.pkl
```

### Monitoring Runner

```bash
# Run monitoring and alerting
python scripts/run_monitoring.py --input data/raw_data.pkl --output data/monitoring_results.pkl
```

## Phase 2 Components

### Macro Economic Data Agent

Collects and processes macro economic indicators:

- **Interest Rates**: Federal Funds Rate, Treasury yields
- **Commodities**: Oil, Gold, Copper prices
- **Economic Indicators**: Unemployment, Payrolls, GDP, CPI (via FRED API)

```python
from agents.macro_data_agent import MacroDataAgent

agent = MacroDataAgent(config)
macro_data = agent.get_macro_data('2020-01-01', '2024-01-01')
```

### Regime Detection Agent

Identifies market regimes based on macro conditions:

- **Rate Regimes**: Easing, Neutral, Tightening
- **Labor Regimes**: Expansion, Stagnation, Contraction
- **Commodity Regimes**: Bull, Bear, Sideways
- **Seasonal Regimes**: Winter, Spring, Summer, Fall
- **Clustered Regimes**: Unsupervised regime detection

```python
from agents.regime_detection_agent import RegimeDetectionAgent

agent = RegimeDetectionAgent(config)
regime_analysis = agent.get_regime_summary(macro_features)
```

### Strategy Selection Agent

Creates regime-aware trading strategies:

- **Regime Affinity**: Strategies optimized for specific market conditions
- **Historical Performance**: Backtested performance analysis
- **Risk Management**: Dynamic risk-adjusted strategy allocation
- **Cross-Asset Features**: BTC and tech sector correlations

```python
from agents.strategy_selection_agent import StrategySelectionAgent

agent = StrategySelectionAgent(config)
recommendations = agent.get_strategy_recommendations(current_regimes, risk_tolerance='medium')
```

## Automated Daily Pipeline

### Windows Task Scheduler Setup

1. **Setup the scheduled task**:
```powershell
# Run as Administrator
powershell scripts\setup_task_scheduler.ps1
```

2. **Customize schedule** (optional):
```powershell
# Run at 8:00 AM instead of 6:00 AM
powershell scripts\setup_task_scheduler.ps1 -TaskTime "08:00"
```

3. **Uninstall the task**:
```powershell
powershell scripts\setup_task_scheduler.ps1 -Uninstall
```

### Daily Pipeline Execution

The `daily_pipeline.ps1` script executes the complete pipeline:

1. Data ingestion from configured sources
2. Feature engineering with GPU acceleration
3. Model training and forecasting
4. Monitoring and drift detection
5. Phase 2 macro/regime/strategy analysis
6. Report generation and cleanup

### Manual Pipeline Execution

```powershell
# Run the complete daily pipeline manually
powershell scripts\daily_pipeline.ps1

# Skip Phase 2 components
powershell scripts\daily_pipeline.ps1 -SkipPhase2

# Use specific Python path
powershell scripts\daily_pipeline.ps1 -PythonPath "C:\Python39\python.exe"
```

## Performance Monitoring and Optimization

### Performance Analysis

```bash
# Run complete performance analysis
python scripts/performance_monitoring.py --run-analysis

# Generate performance report
python scripts/performance_monitoring.py --output reports/performance_20241201.html
```

### Key Metrics Monitored

- **Execution Times**: Component-level timing analysis
- **Resource Usage**: CPU and memory consumption
- **Bottlenecks**: Identification of performance bottlenecks
- **Optimization Recommendations**: Automated improvement suggestions

### Performance Report Features

- HTML reports with interactive charts
- Bottleneck analysis and prioritization
- Resource usage visualization
- Actionable optimization recommendations

## Configuration

### Phase 2 Configuration

Add to `config.yaml`:

```yaml
# Macro Economic Data
macro:
  fred_api_key: "your_fred_api_key"
  indicators: [fed_funds, treasury_10y, oil, gold, unemployment, gdp]
  historical_years: 3

# Regime Detection
regime:
  clustering:
    enabled: true
    n_clusters: 4

# Strategy Selection
strategy:
  risk_tolerance: "medium"
  max_strategies: 5

# Performance Monitoring
performance:
  enabled: true
  alerts:
    max_execution_time: 3600
    max_memory_usage: 8

# Automation
automation:
  enabled: true
  schedule_time: "06:00"
  execution_order: [data_ingestion, feature_engineering, model_training, monitoring, phase2_pipeline, reporting]
```

## Error Handling and Logging

### Logging Configuration

All components use structured logging with:
- File and console output
- Configurable log levels
- Component-specific log files
- Automatic log rotation

### Error Recovery

- **Graceful Degradation**: Components continue on non-critical failures
- **Retry Logic**: Automatic retries for transient failures
- **Fallback Mechanisms**: Alternative data sources and processing methods
- **Alert System**: Email notifications for critical failures

## Deployment and Operations

### Production Setup

1. **Environment Setup**:
   ```bash
   # Create virtual environment
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Configuration**:
   - Update `config.yaml` with production settings
   - Set environment variables for API keys
   - Configure logging and monitoring

3. **Task Scheduler Setup**:
   ```powershell
   # Setup automated daily runs
   powershell scripts\setup_task_scheduler.ps1
   ```

4. **Monitoring**:
   - Check daily logs in `logs/` directory
   - Review performance reports
   - Monitor system resources

### Maintenance

- **Daily Monitoring**: Check execution logs and performance metrics
- **Weekly Review**: Analyze strategy performance and regime changes
- **Monthly Optimization**: Update models and review system performance
- **Quarterly Updates**: Refresh macro data sources and strategy logic

## Troubleshooting

### Common Issues

1. **Data Ingestion Failures**:
   - Check API keys and rate limits
   - Verify network connectivity
   - Review data source availability

2. **Performance Issues**:
   - Run performance analysis: `python scripts/performance_monitoring.py --run-analysis`
   - Check GPU/CPU utilization
   - Review memory usage patterns

3. **Scheduling Issues**:
   - Verify Task Scheduler service is running
   - Check execution permissions
   - Review PowerShell execution policy

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Run components with debug logging
python scripts/run_data_ingestion.py --log-level DEBUG

# Run pipeline with verbose output
powershell scripts\daily_pipeline.ps1 -Verbose
```

## Future Enhancements

- **Real-time Data**: Streaming data integration
- **Advanced ML**: Deep learning and ensemble methods
- **Multi-Asset**: Cross-market strategy optimization
- **Risk Management**: Advanced portfolio risk controls
- **API Endpoints**: REST API for external integration</content>
<parameter name="filePath">c:\Users\spreu\Documents\agentic_forecast\PHASE2_IMPLEMENTATION_README.md