import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

# Conditional import for OTLP exporter
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    _HAS_OTLP = True
except ImportError:
    _HAS_OTLP = False

def setup_tracing(service_name: str = "agentic_forecast"):
    """
    Configures OpenTelemetry tracing.
    Defaults to Console exporter if OTLP endpoint is not set.
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    
    if otlp_endpoint and _HAS_OTLP:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
    else:
        # Fallback to Console for local dev
        exporter = ConsoleSpanExporter()
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)

def get_tracer(name: str):
    return trace.get_tracer(name)
