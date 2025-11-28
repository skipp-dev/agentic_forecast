# MCP Server Extension Guide - Comprehensive Documentation

## Übersicht

Dieser Leitfaden zeigt dir, wie du deinen Interactive Analyst MCP Server erweitern kannst. Er enthält Best Practices, Code-Beispiele und detaillierte Anleitungen für alle Aspekte der Server-Erweiterung.

## Inhaltsverzeichnis

1. [Neue Tools hinzufügen](#1-neue-tools-hinzufügen)
2. [Erweiterte NLP-Funktionen](#2-erweiterte-nlp-funktionen)
3. [Datenbank-Integration](#3-datenbank-integration)
4. [Externe API Integration](#4-externe-api-integration)
5. [Caching-Strategien](#5-caching-strategien)
6. [Monitoring & Logging](#6-monitoring--logging)
7. [Error Handling Best Practices](#7-error-handling-best-practices)
8. [Testing](#8-testing)
9. [Deployment Considerations](#9-deployment-considerations)
10. [Performance Optimization](#10-performance-optimization)
11. [Sicherheit](#11-sicherheit)
12. [Beispiel: Vollständige Integration](#12-beispiel-vollständige-integration)

## 1. Neue Tools hinzufügen

### Grundstruktur eines Tools

```python
async def my_new_tool(
    self,
    required_param: str,
    optional_param: Optional[int] = None
) -> Dict[str, Any]:
    """
    Beschreibung des Tools.

    Args:
        required_param: Beschreibung von Parameter 1
        optional_param: Beschreibung von Parameter 2

    Returns:
        Dict mit Ergebnis und Status
    """
    try:
        # 1. Validate input
        if not required_param:
            raise ValueError("Required parameter missing")

        # 2. Execute logic
        result = await self.some_operation(required_param)

        # 3. Return success
        return {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        # 4. Handle errors
        logger.error(f"Error in my_new_tool: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
```

### Tool in handle_list_tools registrieren

```python
{
    "name": "my_new_tool",
    "description": "Was macht dein Tool?",
    "inputSchema": {
        "type": "object",
        "properties": {
            "parameter1": {
                "type": "string",
                "description": "Beschreibung"
            },
            "parameter2": {
                "type": "integer",
                "description": "Optional Parameter",
                "default": 10
            }
        },
        "required": ["parameter1"]
    }
}
```

### Tool in handle_call_tool einbinden

```python
elif name == "my_new_tool":
    result = await self.my_new_tool(
        arguments.get("parameter1"),
        arguments.get("parameter2")
    )
```

## 2. Erweiterte NLP-Funktionen

### Neue Intent-Patterns hinzufügen

```python
self.intent_patterns = {
    'existing_intent': [...],
    'new_intent': [
        r'\b(keyword1|keyword2|keyword3)\b',
        r'\b(pattern.*with.*wildcards)\b',
        r'\b(another.*pattern)\b'
    ]
}
```

### Custom Entity Extraction

```python
def extract_custom_entity(self, text: str) -> Optional[str]:
    """Extract custom entities from text."""
    pattern = r'your_regex_pattern'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None
```

## 3. Datenbank-Integration

### Beispiel: SQLite Integration

```python
import sqlite3
from contextlib import contextmanager

class DatabaseMixin:
    def __init__(self):
        self.db_path = "analyst_data.db"
        self._init_database()

    def _init_database(self):
        """Initialize database tables."""
        with self.get_db_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY,
                    query_text TEXT,
                    intent TEXT,
                    timestamp DATETIME,
                    result TEXT
                )
            ''')

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    async def log_query(self, query: str, intent: str, result: str):
        """Log query to database."""
        with self.get_db_connection() as conn:
            conn.execute(
                "INSERT INTO queries (query_text, intent, timestamp, result) VALUES (?, ?, ?, ?)",
                (query, intent, datetime.now(), result)
            )
```

## 4. Externe API Integration

### Beispiel: REST API Call

```python
import aiohttp

async def fetch_external_data(self, endpoint: str, params: Dict) -> Dict:
    """Fetch data from external API."""
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"API call failed: {response.status}")
```

### Beispiel: Webhook Integration

```python
async def send_webhook_notification(
    self,
    webhook_url: str,
    payload: Dict
) -> bool:
    """Send notification via webhook."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                return response.status == 200
    except Exception as e:
        logger.error(f"Webhook failed: {e}")
        return False
```

## 5. Caching-Strategien

### LRU Cache Implementation

```python
from functools import lru_cache
from collections import OrderedDict

class LRUCache:
    """Simple LRU Cache Implementation"""
    def __init__(self, capacity: int = 100):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

### Redis Cache (für Production)

```python
import redis.asyncio as redis

class RedisCache:
    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis = redis.from_url(redis_url)

    async def get(self, key: str) -> Optional[str]:
        return await self.redis.get(key)

    async def set(self, key: str, value: str, ttl: int = 300):
        await self.redis.setex(key, ttl, value)

    async def close(self):
        await self.redis.close()
```

## 6. Monitoring & Logging

### Strukturiertes Logging

```python
import logging
import json

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log_event(self, event_type: str, data: Dict):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.logger.info(json.dumps(log_entry))
```

### Performance Metrics

```python
import time
from functools import wraps

def measure_time(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper

@measure_time
async def expensive_operation(self):
    # Your code here
    pass
```

## 7. Error Handling Best Practices

### Custom Exceptions

```python
class AnalystException(Exception):
    """Base exception for analyst errors."""
    pass

class DataNotFoundError(AnalystException):
    """Raised when requested data is not found."""
    pass

class ValidationError(AnalystException):
    """Raised when input validation fails."""
    pass

# Usage
def validate_bucket(bucket: str):
    if bucket not in VALID_BUCKETS:
        raise ValidationError(f"Invalid bucket: {bucket}")
```

### Graceful Error Handling

```python
async def safe_execute(self, func, *args, **kwargs):
    """Execute function with comprehensive error handling."""
    try:
        return await func(*args, **kwargs)
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return {"status": "error", "type": "validation", "message": str(e)}
    except DataNotFoundError as e:
        logger.warning(f"Data not found: {e}")
        return {"status": "error", "type": "not_found", "message": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {"status": "error", "type": "internal", "message": "Internal server error"}
```

## 8. Testing

### Unit Tests

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_export_data():
    server = ExtendedInteractiveAnalystMCPServer()
    result = await server.export_analysis_report("summary", "json")
    assert result["status"] == "success"

@pytest.mark.asyncio
async def test_nlp_classification():
    nlp = EnhancedNaturalLanguageProcessor()
    intent, confidence = nlp.classify_intent("Show me the weakest buckets")
    assert intent == "weakest"
    assert confidence > 0.5
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_query_flow():
    server = InteractiveAnalystMCPServer()

    # Test the complete flow
    result = await server.process_natural_language_query(
        "Show me the weakest performing buckets",
        session_id="test_session"
    )
    assert "Natural Language Query" in result
    assert "Classified Intent: weakest" in result
```

## 9. Deployment Considerations

### Environment Variables

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG = os.getenv("DEBUG", "False") == "True"
    CACHE_TIMEOUT = int(os.getenv("CACHE_TIMEOUT", "300"))
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///analyst.db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "server.py"]
```

### Health Check Endpoint

```python
async def health_check(self) -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime": time.time() - self.start_time,
        "cache_size": len(self.query_cache),
        "active_sessions": len(self.conversation_contexts)
    }
```

## 10. Performance Optimization

### Async Processing

```python
async def process_multiple_buckets(self, buckets: List[str]):
    """Process multiple buckets in parallel."""
    tasks = [self.analyze_bucket(bucket) for bucket in buckets]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    successful_results = [
        r for r in results
        if not isinstance(r, Exception)
    ]

    return successful_results
```

### Batch Operations

```python
async def batch_process(self, items: List[Any], batch_size: int = 10):
    """Process items in batches."""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[self.process_item(item) for item in batch]
        )
        results.extend(batch_results)
    return results
```

## 11. Sicherheit

### Input Validation

```python
from typing import Optional
import re

def sanitize_input(text: str) -> str:
    """Sanitize user input."""
    # Remove potentially dangerous characters
    text = re.sub(r'[<>{}]', '', text)
    # Limit length
    return text[:1000]

def validate_file_path(path: str) -> bool:
    """Validate file path to prevent directory traversal."""
    # Normalize path
    normalized = os.path.normpath(path)
    # Check for directory traversal attempts
    if '..' in normalized or normalized.startswith('/'):
        return False
    return True
```

### Rate Limiting

```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int = 100, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)

    def is_allowed(self, user_id: str) -> bool:
        now = time.time()
        # Clean old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if now - req_time < self.window
        ]

        if len(self.requests[user_id]) < self.max_requests:
            self.requests[user_id].append(now)
            return True
        return False
```

## 12. Beispiel: Vollständige Integration

Hier ist ein vollständiges Beispiel, wie du alle Features kombinierst:

```python
class ProductionInteractiveAnalystMCPServer(
    InteractiveAnalystMCPServer,
    DatabaseMixin
):
    """Production-ready MCP server with all features."""

    def __init__(self):
        super().__init__()
        DatabaseMixin.__init__(self)

        self.cache = LRUCache(capacity=100)
        self.rate_limiter = RateLimiter()
        self.logger = StructuredLogger(__name__)
        self.start_time = time.time()

    @measure_time
    async def process_query_with_all_features(
        self,
        query: str,
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """Process query with caching, logging, rate limiting."""

        # Rate limiting
        if not self.rate_limiter.is_allowed(user_id):
            return {"status": "error", "message": "Rate limit exceeded"}

        # Input validation
        query = sanitize_input(query)

        # Check cache
        cache_key = f"{user_id}:{query}"
        cached = self.cache.get(cache_key)
        if cached:
            self.logger.log_event("cache_hit", {"user_id": user_id})
            return cached

        # Process query
        try:
            result = await self.process_natural_language_query(
                query,
                session_id=user_id
            )

            # Cache result
            self.cache.put(cache_key, result)

            # Log to database
            await self.log_query(query, "processed", str(result))

            # Log event
            self.logger.log_event("query_processed", {
                "user_id": user_id,
                "query": query
            })

            return result

        except Exception as e:
            self.logger.log_event("query_error", {
                "user_id": user_id,
                "error": str(e)
            })
            raise
```

## Zusammenfassung

Wichtigste Best Practices:
1. ✅ Async/await für alle I/O-Operationen
2. ✅ Proper error handling mit custom exceptions
3. ✅ Caching für Performance
4. ✅ Logging für Monitoring
5. ✅ Input validation für Sicherheit
6. ✅ Rate limiting zum Schutz
7. ✅ Tests für Zuverlässigkeit
8. ✅ Type hints für Klarheit

## Nächste Schritte

1. Wähle Features aus, die du brauchst
2. Integriere sie in deinen bestehenden Server
3. Teste gründlich
4. Deploy schrittweise
5. Monitor und optimiere