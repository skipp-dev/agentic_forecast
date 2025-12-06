from prometheus_client import Counter, Gauge, Histogram, start_http_server
import logging

logger = logging.getLogger(__name__)

# Metrics Definitions
PIPELINE_LATENCY = Histogram(
    'pipeline_execution_seconds', 
    'Time spent executing the full agentic pipeline',
    buckets=[10, 30, 60, 120, 300, 600, 1800]
)

AGENT_LATENCY = Histogram(
    'agent_execution_seconds',
    'Time spent by individual agents',
    ['agent_name']
)

PORTFOLIO_VALUE = Gauge(
    'portfolio_value_usd',
    'Current total value of the portfolio'
)

CASH_BALANCE = Gauge(
    'portfolio_cash_usd',
    'Current cash balance'
)

TRADE_COUNT = Counter(
    'trades_total',
    'Total number of trades executed',
    ['symbol', 'action', 'status']
)

FORECAST_ERROR = Gauge(
    'forecast_mape',
    'Mean Absolute Percentage Error of forecasts',
    ['symbol', 'model']
)

SYSTEM_ERRORS = Counter(
    'system_errors_total',
    'Total number of system errors/exceptions',
    ['component']
)

# Drift Monitor Meta-Metrics
DRIFT_MONITOR_RUNS = Counter(
    'drift_monitor_runs_total',
    'Total number of drift checks executed',
    ['symbol']
)

DRIFT_FLAGS_RAISED = Counter(
    'drift_flags_raised_total',
    'Total number of drift flags raised',
    ['symbol', 'drift_type']
)

DRIFT_SCORE = Gauge(
    'drift_score_current',
    'Current drift score (0-1)',
    ['symbol']
)

def start_metrics_server(port: int = 8000):
    """Start the Prometheus metrics server."""
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
