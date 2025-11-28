-- IB Forecast Database Initialization

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS forecast;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS audit;

-- Forecast schema tables
CREATE TABLE IF NOT EXISTS forecast.models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    framework VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB,
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS forecast.forecasts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES forecast.models(id),
    symbol VARCHAR(20) NOT NULL,
    forecast_date DATE NOT NULL,
    prediction_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    forecast_value DECIMAL(20,8) NOT NULL,
    confidence_lower DECIMAL(20,8),
    confidence_upper DECIMAL(20,8),
    features JSONB,
    metadata JSONB
);

-- Create hypertable for time-series data
SELECT create_hypertable('forecast.forecasts', 'prediction_date', if_not_exists => TRUE);

-- Analytics schema tables
CREATE TABLE IF NOT EXISTS analytics.metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    value DECIMAL(20,8) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags JSONB,
    metadata JSONB
);

-- Create hypertable for metrics
SELECT create_hypertable('analytics.metrics', 'timestamp', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS analytics.reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_type VARCHAR(100) NOT NULL,
    report_date DATE NOT NULL,
    content JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'completed'
);

-- Audit schema tables
CREATE TABLE IF NOT EXISTS audit.api_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    method VARCHAR(10) NOT NULL,
    endpoint VARCHAR(500) NOT NULL,
    user_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    request_body JSONB,
    response_status INTEGER,
    response_time DECIMAL(10,3),
    error_message TEXT
);

-- Create hypertable for audit logs
SELECT create_hypertable('audit.api_requests', 'timestamp', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS audit.model_operations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    operation VARCHAR(100) NOT NULL,
    model_id UUID,
    user_id VARCHAR(255),
    details JSONB,
    status VARCHAR(50) DEFAULT 'success',
    error_message TEXT
);

-- Create hypertable for model operations
SELECT create_hypertable('audit.model_operations', 'timestamp', if_not_exists => TRUE);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_forecasts_model_id ON forecast.forecasts(model_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_symbol_date ON forecast.forecasts(symbol, forecast_date);
CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON analytics.metrics(metric_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_tags ON analytics.metrics USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_reports_type_date ON analytics.reports(report_type, report_date);
CREATE INDEX IF NOT EXISTS idx_api_requests_timestamp ON audit.api_requests(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_api_requests_endpoint ON audit.api_requests(endpoint);
CREATE INDEX IF NOT EXISTS idx_model_operations_model_id ON audit.model_operations(model_id);

-- Retention policies (optional - adjust as needed)
-- SELECT add_retention_policy('forecast.forecasts', INTERVAL '1 year');
-- SELECT add_retention_policy('analytics.metrics', INTERVAL '6 months');
-- SELECT add_retention_policy('audit.api_requests', INTERVAL '2 years');

-- Create views for common queries
CREATE OR REPLACE VIEW analytics.latest_metrics AS
SELECT DISTINCT ON (metric_name) *
FROM analytics.metrics
ORDER BY metric_name, timestamp DESC;

CREATE OR REPLACE VIEW forecast.model_performance AS
SELECT
    m.name as model_name,
    m.version,
    COUNT(f.id) as forecast_count,
    AVG(f.forecast_value) as avg_forecast,
    STDDEV(f.forecast_value) as forecast_stddev,
    MIN(f.prediction_date) as first_prediction,
    MAX(f.prediction_date) as last_prediction
FROM forecast.models m
LEFT JOIN forecast.forecasts f ON m.id = f.model_id
WHERE m.status = 'active'
GROUP BY m.id, m.name, m.version;

-- Row Level Security (optional - for multi-tenant setup)
-- ALTER TABLE forecast.models ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE forecast.forecasts ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE analytics.metrics ENABLE ROW LEVEL SECURITY;

-- Create service user (password should be set via environment variable)
-- CREATE USER ib_service WITH PASSWORD 'set_via_env';
-- GRANT USAGE ON SCHEMA forecast, analytics, audit TO ib_service;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA forecast, analytics, audit TO ib_service;