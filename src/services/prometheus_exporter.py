try:
    from prometheus_client import start_http_server, Summary, Counter, Gauge
except ImportError:
    print("prometheus_client not installed. Please install it to use PrometheusExporter.")
    # Mock classes to prevent import errors if package is missing
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def time(self): return self
        def __enter__(self): pass
        def __exit__(self, *args): pass
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, amount=1): pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, value): pass
        def labels(self, **kwargs): return self
    def start_http_server(*args, **kwargs): pass

from ..core.run_context import RunContext

# Metrics definitions
FORECAST_REQUESTS = Counter('forecast_requests_total', 'Total number of forecast requests')
LLM_TOKENS_USED = Counter('llm_tokens_total', 'Total LLM tokens used', ['model', 'agent'])
FORECAST_ACCURACY = Gauge('forecast_accuracy_mape', 'Forecast MAPE', ['symbol'])

# New metrics with run_type
METRICS_HEALTH_STATUS = Gauge(
    "metrics_health_status",
    "Overall health of the latest evaluation run (1=passed, 0=failed)",
    ["run_type", "run_id"],
)

FORECAST_MAE = Gauge(
    "forecast_mae",
    "Mean Absolute Error per symbol/horizon/model",
    ["run_type", "run_id", "symbol", "horizon", "model_family"],
)

FORECAST_GUARDRAIL_FLAG = Gauge(
    "forecast_guardrail_flag",
    "Guardrail flags per symbol (1=active, 0=inactive)",
    ["run_type", "run_id", "symbol", "flag"],
)

NEWS_ITEMS_COUNT = Gauge(
    "news_items_count",
    "Number of news items processed per symbol",
    ["run_type", "run_id", "symbol"]
)

NEWS_SENTIMENT_SCORE = Gauge(
    "news_sentiment_score",
    "Average news sentiment score per symbol",
    ["run_type", "run_id", "symbol"]
)

class PrometheusExporter:
    def __init__(self, port=9090):
        self.port = port
        
    def start(self):
        """Start the Prometheus metrics server."""
        try:
            start_http_server(self.port)
            print(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            print(f"Failed to start Prometheus server: {e}")

    def record_llm_tokens(self, model: str, agent: str, tokens: int):
        LLM_TOKENS_USED.labels(model=model, agent=agent).inc(tokens)

    def record_forecast_accuracy(self, symbol: str, mape: float):
        FORECAST_ACCURACY.labels(symbol=symbol).set(mape)

    # New helper methods
    def set_metrics_health_status(self, ctx: RunContext, value: float):
        METRICS_HEALTH_STATUS.labels(
            run_type=ctx.run_type.value,
            run_id=ctx.run_id,
        ).set(value)

    def set_forecast_mae(self, ctx: RunContext, symbol: str, horizon: int, model_family: str, value: float):
        FORECAST_MAE.labels(
            run_type=ctx.run_type.value,
            run_id=ctx.run_id,
            symbol=symbol,
            horizon=str(horizon),
            model_family=model_family,
        ).set(value)

    def set_guardrail_flag(self, ctx: RunContext, symbol: str, flag: str, active: bool):
        FORECAST_GUARDRAIL_FLAG.labels(
            run_type=ctx.run_type.value,
            run_id=ctx.run_id,
            symbol=symbol,
            flag=flag,
        ).set(1.0 if active else 0.0)

    def set_news_metrics(self, ctx: RunContext, symbol: str, count: int, sentiment: float):
        NEWS_ITEMS_COUNT.labels(
            run_type=ctx.run_type.value,
            run_id=ctx.run_id,
            symbol=symbol
        ).set(count)
        
        NEWS_SENTIMENT_SCORE.labels(
            run_type=ctx.run_type.value,
            run_id=ctx.run_id,
            symbol=symbol
        ).set(sentiment)
