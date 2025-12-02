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
    def start_http_server(*args, **kwargs): pass

# Metrics definitions
FORECAST_REQUESTS = Counter('forecast_requests_total', 'Total number of forecast requests')
LLM_TOKENS_USED = Counter('llm_tokens_total', 'Total LLM tokens used', ['model', 'agent'])
FORECAST_ACCURACY = Gauge('forecast_accuracy_mape', 'Forecast MAPE', ['symbol'])

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
