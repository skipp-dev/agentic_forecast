# agentic_forecast: Agentic Financial Forecasting Framework

An advanced, agent-driven forecasting system for financial markets using Interactive Brokers data. This framework leverages a sophisticated agentic architecture to autonomously manage the entire forecasting pipeline, from data ingestion to model deployment.

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support (recommended)
- Interactive Brokers account and API access
- Python 3.11+

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure the system (optional)
# Edit config.yaml to customize thresholds and settings

# Run the agentic forecasting pipeline
python main.py
```

## Architecture Overview

The agentic_forecast framework uses an agentic architecture built on LangGraph, where specialized agents handle different aspects of the forecasting pipeline:

### Core Agents
- **Orchestrator Agent**: Coordinates the entire workflow and manages sub-agents.
- **Hyperparameter Search Agent**: Optimizes model hyperparameters using Optuna.
- **Drift Monitor Agent**: Detects data, performance, and spectral drift.
- **Feature Engineer Agent**: Generates technical indicators and spectral features (cuFFT).
- **Forecast Agent**: Trains and runs ensemble forecasting models (LSTM, XGBoost, LightGBM).
- **Reporting Agent**: Generates summaries and reports.

### Key Features

---
## Deep System Overview

### 1. Agentic Multi-Agent Architecture
- **OrchestratorAgent**: Central workflow coordinator, manages sub-agents, optimizes GPU resources, and advanced decision-making.
- **SupervisorAgent**: LangChain-based, routes tasks and manages agent interactions.
- **FeatureEngineerAgent**: Generates technical and spectral features (cuFFT).
- **DriftMonitorAgent**: Monitors data, performance, and spectral drift.
- **ForecastAgent**: Ensemble forecasting (LSTM, XGBoost, LightGBM, RF, GBM), confidence intervals, stacking/blending.
- **HyperparameterSearchAgent**: Automated model selection and tuning (Optuna).
- **ReportingAgent**: Summarizes results, generates reports.

### 2. ML & Data Pipeline
- **DataPipeline**: Ingests, preprocesses, and stores market data.
- **GPU Services**: Device management, memory optimization, distributed training, spectral feature extraction.
- **Model Registry**: Tracks models, checkpoints, and training history.
- **Monitoring Service**: Prometheus metrics for system, GPU, and application health.

### 3. Deployment & Environment
- **Dockerfiles**: Multi-stage builds for Python 3.12 + CUDA 12.x, with all dependencies.
- **docker-compose.yml**: GPU device reservations, service orchestration.
- **WSL2 Integration**: Ensures GPU access and performance.
- **Scripts**: GPU verification, troubleshooting, quick tests, and environment setup.

### 4. Configuration
- **config.yaml**: Main system configuration.
- **settings.toml**: Advanced settings (ports, thresholds, batch sizes).
- **quality.yml**: Quality and validation settings.

### 5. Testing & Validation
- **Unit Tests**: Coverage for data pipeline, ensemble methods, GPU performance, agent workflows.
- **GPU Tests**: PyTorch and TensorFlow GPU checks, performance benchmarks.
- **Integration Tests**: End-to-end pipeline validation.

### 6. Documentation
- **README.md**: Quick start, architecture, configuration, project structure.
- **USER_GUIDE.md**: Detailed user instructions, troubleshooting, workflow examples.
- **ARCHITECTURE_IMPLEMENTATION_GUIDE.md**: Implementation details, migration steps, success metrics.
- **GPU_CONTAINER_SETUP.md**: Container build, verification, troubleshooting.
- **OPTIMIZED_GPU_CONTAINER_SUMMARY.md**: Reference implementation, CUDA stack, performance tips.
- **WSL2_AGENTIC_FORECAST_SETUP.md**: WSL2 setup and GPU configuration.
- **PRE_LAUNCH_CHECKLIST.md**: Launch checklist and expected outputs.

---

## Extension Points & Customization
- Add new agents for custom tasks (e.g., alternative data, new ML models).
- Extend GPU services for additional hardware or distributed training.
- Integrate with external APIs (news, alternative data, analytics).
- Customize workflow logic in OrchestratorAgent and SupervisorAgent.
- Modify configuration files for thresholds, batch sizes, and resource allocation.

---

## Troubleshooting & Best Practices
- Use provided scripts for GPU verification and troubleshooting.
- Follow container setup guides for reproducible environments.
- Monitor system and GPU health via Prometheus/Grafana.
- Review architecture and implementation guides for migration and scaling.

---
## Configuration

The system is configured via `config.yaml`.

## Project Structure

```
agentic_forecast/
├── agents/               # Specialized forecasting agents
├── graphs/               # State management and graph definitions
├── src/                  # Core services and utilities
│   ├── gpu_services.py   # GPU management and spectral analysis
│   ├── data_pipeline.py  # Data ingestion and processing
│   └── ensemble_methods.py # Ensemble forecasting logic
├── tests/                # Unit and integration tests
├── config.yaml           # System configuration
├── main.py               # Entry point
└── requirements.txt      # Python dependencies
```

## Development

### Running Tests
```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License.


