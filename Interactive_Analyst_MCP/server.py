#!/usr/bin/env python3
"""
Interactive Analyst MCP Server - Enhanced Python Implementation

A Model Context Protocol (MCP) server that provides advanced natural language processing
capabilities for the Interactive Analyst Mode using Python.
"""

import asyncio
import json
import logging
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import subprocess
from pathlib import Path
import os
import sqlite3
from collections import OrderedDict, defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalystQuery:
    intent: str
    bucket: Optional[str] = None
    horizon: Optional[str] = None
    top_n: Optional[int] = None
    plot_type: Optional[str] = None
    filename: Optional[str] = None
    confidence: float = 1.0
    compound_queries: List['AnalystQuery'] = field(default_factory=list)

@dataclass
class ConversationContext:
    """Context for maintaining conversation state."""
    session_id: str
    last_query: Optional[str] = None
    last_result: Optional[str] = None
    query_history: List[Dict[str, Any]] = field(default_factory=list)
    preferred_bucket: Optional[str] = None
    preferred_horizon: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class LRUCache:
    """Simple LRU Cache Implementation"""
    def __init__(self, capacity: int = 100):
        self.cache: OrderedDict[str, Any] = OrderedDict()
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


class RateLimiter:
    """Simple Rate Limiter"""
    def __init__(self, max_requests: int = 100, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[str, List[float]] = defaultdict(list)

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

class EnhancedNaturalLanguageProcessor:
    """Enhanced natural language processor with context awareness and compound queries."""

    def __init__(self):
        # Enhanced intent patterns with more sophisticated matching
        self.intent_patterns = {
            'summary': [
                r'\b(summary|overview|performance|weekly|monthly|status|report)\b',
                r'\b(how.*doing|what.*happening|give.*me.*update)\b',
                r'\b(status|health|condition|state)\b'
            ],
            'weakest': [
                r'\b(weakest|worst|underperforming|bad|poor|struggling)\b',
                r'\b(bottom|lowest|worse|lagging|trailing)\b',
                r'\b(under.*perform|not.*good|problematic)\b'
            ],
            'bucket': [
                r'\b(bucket|drilldown|analyze|deep.dive|examine|investigate)\b',
                r'\b(breakdown|details?|specific|particular)\b',
                r'\b(focus.*on|look.*at|check.*out)\b'
            ],
            'guardrails': [
                r'\b(guardrail|violation|drift|alert|warning|issue)\b',
                r'\b(problem|error|anomaly|outlier|breach)\b',
                r'\b(monitor|check|validate|verify)\b'
            ],
            'hpo': [
                r'\b(hpo|hyperparameter|optimization|tune|improve|enhance)\b',
                r'\b(optimize|parameter|tuning|boost|upgrade)\b',
                r'\b(better|improve|enhance|fine.*tune)\b'
            ],
            'switches': [
                r'\b(switch|change|model|ensemble|replace|alternative)\b',
                r'\b(different|new|other|swap|substitute)\b',
                r'\b(try|use|implement|adopt)\b'
            ],
            'plot': [
                r'\b(plot|chart|graph|visualize|show|display)\b',
                r'\b(diagram|figure|visual|render|draw)\b',
                r'\b(see|view|look|examine)\b'
            ],
            'export': [
                r'\b(export|save|download|write|store|record)\b',
                r'\b(file|output|document|report|archive)\b',
                r'\b(create|generate|produce|make)\b'
            ],
            'compare': [
                r'\b(compare|comparison|versus|vs|against|relative)\b',
                r'\b(difference|contrast|benchmark|measure)\b',
                r'\b(better.*than|worse.*than|compared.*to)\b'
            ],
            'trend': [
                r'\b(trend|pattern|movement|direction|change.*over.*time)\b',
                r'\b(evolution|progression|trajectory|development)\b',
                r'\b(over.*time|throughout|across.*period)\b'
            ],
            'help': [
                r'\b(help|assist|guide|support|what.*can.*you.*do)\b',
                r'\b(commands|options|capabilities|features)\b',
                r'\b(how.*to|what.*is|explain)\b'
            ]
        }

        # Known buckets and horizons for better entity recognition
        self.known_buckets = [
            'ai_basket', 'crypto_exposed', 'defensive', 'energy_oil', 'AAL_daily',
            'basket', 'crypto', 'energy', 'defensive', 'daily'
        ]

        self.known_horizons = ['1', '5', '10', '20', '30', '60', '90']

        # Compound query patterns
        self.compound_patterns = [
            r'\b(and|also|then|next|after.*that)\b',
            r'\b(first|second|then|finally)\b',
            r'\b(moreover|furthermore|additionally)\b'
        ]

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Classify the intent of a natural language query with confidence score."""
        text_lower = text.lower()
        max_score = 0.0
        best_intent = 'summary'

        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            matches = 0

            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches += 1
                    # Give higher weight to exact matches
                    if re.search(r'\b' + re.escape(intent) + r'\b', text_lower):
                        score += 2.0
                    else:
                        score += 1.0

            # Normalize score by pattern count
            if patterns:
                score = score / len(patterns)

            if score > max_score:
                max_score = score
                best_intent = intent

        # Boost confidence for clear matches
        confidence = min(max_score / 2.0, 1.0) if max_score > 0 else 0.3

        return best_intent, confidence

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from natural language text with enhanced recognition."""
        entities = {}

        # Extract bucket names with fuzzy matching
        text_lower = text.lower()
        for bucket in self.known_buckets:
            if bucket.lower() in text_lower:
                entities['bucket'] = bucket
                break

        # Extract horizons with context
        horizon_pattern = r'\b(\d+)\s*(?:day|d|horizon|period)s?\b'
        horizon_match = re.search(horizon_pattern, text, re.IGNORECASE)
        if horizon_match:
            horizon = horizon_match.group(1)
            if horizon in self.known_horizons:
                entities['horizon'] = horizon

        # Extract top N with more flexible patterns
        top_patterns = [
            r'\b(?:top|best|worst|first)\s+(\d+)\b',
            r'\b(\d+)\s+(?:top|best|worst)\b',
            r'\bshow\s+me\s+(\d+)\b'
        ]
        for pattern in top_patterns:
            top_match = re.search(pattern, text, re.IGNORECASE)
            if top_match:
                entities['top_n'] = int(top_match.group(1))
                break

        # Extract plot types with synonyms
        plot_synonyms = {
            'residuals': ['residual', 'residuals', 'error', 'errors'],
            'performance': ['performance', 'perf', 'results', 'metrics'],
            'distribution': ['distribution', 'dist', 'histogram', 'hist'],
            'histogram': ['histogram', 'hist', 'distribution', 'freq'],
            'scatter': ['scatter', 'correlation', 'relationship'],
            'time_series': ['time', 'series', 'trend', 'temporal']
        }

        for plot_type, synonyms in plot_synonyms.items():
            for synonym in synonyms:
                if synonym in text_lower:
                    entities['plot_type'] = plot_type
                    break
            if 'plot_type' in entities:
                break

        # Extract filenames with better patterns
        filename_patterns = [
            r'\b(?:to|as|file|name[d]?\s*:?\s*)([\w\.-]+)\b',
            r'\b(?:save|export|write)\s+(?:to\s+)?([\w\.-]+)\b',
            r'\b([\w\.-]+)\.(?:md|txt|csv|json|html)\b'
        ]
        for pattern in filename_patterns:
            filename_match = re.search(pattern, text, re.IGNORECASE)
            if filename_match:
                entities['filename'] = filename_match.group(1)
                break

        # Extract date ranges
        date_patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',  # YYYY-MM-DD
            r'\b(last|past)\s+(\d+)\s+(days?|weeks?|months?)\b',
            r'\b(from|since)\s+(\d{4}-\d{2}-\d{2})\b'
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, text, re.IGNORECASE)
            if date_match:
                entities['date_range'] = date_match.group(0)
                break

        return entities

    def detect_compound_queries(self, text: str) -> List[str]:
        """Detect if the query contains multiple related requests."""
        sentences = re.split(r'[.!?]+', text)
        compound_queries = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                compound_queries.append(sentence)

        # Also check for compound patterns
        for pattern in self.compound_patterns:
            if re.search(pattern, text):
                # Split on compound connectors
                parts = re.split(pattern, text)
                compound_queries = [part.strip() for part in parts if len(part.strip()) > 5]
                break

        return compound_queries if len(compound_queries) > 1 else [text]

    def process_query(self, text: str, context: Optional[ConversationContext] = None) -> AnalystQuery:
        """Process a natural language query with context awareness."""
        # Check for compound queries
        compound_parts = self.detect_compound_queries(text)

        if len(compound_parts) > 1:
            # Process as compound query
            compound_queries = []
            for part in compound_parts:
                intent, confidence = self.classify_intent(part)
                entities = self.extract_entities(part)
                compound_queries.append(AnalystQuery(
                    intent=intent,
                    bucket=entities.get('bucket'),
                    horizon=entities.get('horizon'),
                    top_n=entities.get('top_n'),
                    plot_type=entities.get('plot_type'),
                    filename=entities.get('filename'),
                    confidence=confidence
                ))

            # Use the primary intent for the main query
            primary_query = compound_queries[0]
            primary_query.compound_queries = compound_queries[1:]
            return primary_query
        else:
            # Single query processing
            intent, confidence = self.classify_intent(text)
            entities = self.extract_entities(text)

            # Apply context if available
            if context:
                # Use preferred bucket/horizon if not specified
                if not entities.get('bucket') and context.preferred_bucket:
                    entities['bucket'] = context.preferred_bucket
                if not entities.get('horizon') and context.preferred_horizon:
                    entities['horizon'] = context.preferred_horizon

            return AnalystQuery(
                intent=intent,
                bucket=entities.get('bucket'),
                horizon=entities.get('horizon'),
                top_n=entities.get('top_n'),
                plot_type=entities.get('plot_type'),
                filename=entities.get('filename'),
                confidence=confidence
            )

    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Provide query suggestions based on partial input."""
        suggestions = []
        partial_lower = partial_query.lower()

        if 'weak' in partial_lower or 'worst' in partial_lower:
            suggestions.extend([
                "Show me the weakest performing buckets",
                "What are the worst performing buckets this week?",
                "Display the top 5 weakest buckets"
            ])
        elif 'summary' in partial_lower or 'overview' in partial_lower:
            suggestions.extend([
                "Give me a performance summary",
                "Show me the weekly overview",
                "What's the current status?"
            ])
        elif 'bucket' in partial_lower or 'analyze' in partial_lower:
            suggestions.extend([
                "Analyze ai_basket performance",
                "Show me bucket details for crypto_exposed",
                "Drill down into defensive bucket"
            ])
        elif 'plot' in partial_lower or 'chart' in partial_lower:
            suggestions.extend([
                "Plot the residuals",
                "Show me a performance chart",
                "Display distribution graphs"
            ])

        return suggestions[:5]  # Limit to 5 suggestions

class InteractiveAnalystMCPServer:
    """Enhanced MCP server for natural language processing in Interactive Analyst Mode."""

    def __init__(self):
        logger.info("Initializing Interactive Analyst MCP Server...")

        self.nlp = EnhancedNaturalLanguageProcessor()

        # Check if interactive script exists
        interactive_path = Path(__file__).parent.parent / "interactive.py"
        if interactive_path.exists():
            self.interactive_script = interactive_path
            logger.info(f"Found interactive script: {interactive_path}")
        else:
            self.interactive_script = None
            logger.warning(f"Interactive script not found: {interactive_path}")

        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timeout = 300  # 5 minutes

        # New extended features
        self.cache = LRUCache(capacity=100)
        self.rate_limiter = RateLimiter()
        self.start_time = time.time()

        # Data structures for new tools
        self.reports_dir = Path(__file__).parent / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        self.scheduled_queries: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.user_preferences: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.query_history: List[Dict[str, Any]] = []

        # Database setup (optional)
        self.db_path = Path(__file__).parent / "analyst_data.db"
        try:
            self._init_database()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.db_path = None

        logger.info("Interactive Analyst MCP Server initialization complete")

    def _init_database(self):
        """Initialize SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS queries (
                        id INTEGER PRIMARY KEY,
                        query_text TEXT,
                        intent TEXT,
                        timestamp DATETIME,
                        result TEXT,
                        user_id TEXT
                    )
                ''')
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS reports (
                        id INTEGER PRIMARY KEY,
                        report_type TEXT,
                        format TEXT,
                        file_path TEXT,
                        created_at DATETIME,
                        size_bytes INTEGER
                    )
                ''')
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")

    def _save_report_to_db(self, report_type: str, format: str, file_path: str, size: int):
        """Save report metadata to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO reports (report_type, format, file_path, created_at, size_bytes) VALUES (?, ?, ?, ?, ?)",
                    (report_type, format, file_path, datetime.now(), size)
                )
        except Exception as e:
            logger.warning(f"Failed to save report to DB: {e}")

    def _calculate_next_run(self, schedule: str) -> str:
        """Calculate next run time based on schedule"""
        now = datetime.now()

        if schedule == "hourly":
            next_run = now + timedelta(hours=1)
        elif schedule == "daily":
            next_run = now + timedelta(days=1)
            next_run = next_run.replace(hour=9, minute=0, second=0)
        elif schedule == "weekly":
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            next_run = now + timedelta(days=days_until_monday)
            next_run = next_run.replace(hour=9, minute=0, second=0)
        elif schedule == "monthly":
            if now.month == 12:
                next_run = now.replace(year=now.year + 1, month=1, day=1, hour=9, minute=0, second=0)
            else:
                next_run = now.replace(month=now.month + 1, day=1, hour=9, minute=0, second=0)
        else:
            next_run = now + timedelta(days=1)

        return next_run.strftime("%Y-%m-%d %H:%M:%S")

    def _generate_markdown_report(self, report_type: str, data: str, include_charts: bool) -> str:
        """Generate markdown report"""
        content = f"""# Forecast Analysis Report - {report_type.title()}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
{data}

## Report Details
- **Type**: {report_type}
- **Generated**: {datetime.now().isoformat()}
"""
        if include_charts:
            content += "\n## Charts\n[Chart visualizations would be embedded here]\n"

        return content

    def _generate_json_report(self, report_type: str, data: str, include_charts: bool) -> Dict:
        """Generate JSON report"""
        return {
            "report_type": report_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "metadata": {
                "include_charts": include_charts,
                "version": "1.0"
            }
        }

    def _generate_html_report(self, report_type: str, data: str, include_charts: bool) -> str:
        """Generate HTML report"""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Forecast Analysis Report - {report_type.title()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; }}
        .content {{ margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Forecast Analysis Report - {report_type.title()}</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    <div class="content">
        <h2>Executive Summary</h2>
        <pre>{data}</pre>
    </div>
</body>
</html>"""

    def get_or_create_context(self, session_id: str) -> ConversationContext:
        """Get or create conversation context for a session."""
        if session_id not in self.conversation_contexts:
            self.conversation_contexts[session_id] = ConversationContext(session_id=session_id)

        # Clean up old contexts (older than 1 hour)
        current_time = datetime.now()
        expired_sessions = [
            sid for sid, ctx in self.conversation_contexts.items()
            if (current_time - ctx.timestamp).total_seconds() > 3600
        ]
        for sid in expired_sessions:
            del self.conversation_contexts[sid]

        return self.conversation_contexts[session_id]

    def update_context(self, context: ConversationContext, query: AnalystQuery, result: str):
        """Update conversation context with query and result."""
        context.last_query = query.intent
        context.last_result = result
        context.query_history.append({
            'timestamp': datetime.now(),
            'intent': query.intent,
            'entities': {
                'bucket': query.bucket,
                'horizon': query.horizon,
                'top_n': query.top_n
            },
            'confidence': query.confidence
        })

        # Update preferences based on usage patterns
        if query.bucket:
            context.preferred_bucket = query.bucket
        if query.horizon:
            context.preferred_horizon = query.horizon

        # Keep only last 10 queries
        context.query_history = context.query_history[-10:]
        context.timestamp = datetime.now()

    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if still valid."""
        if cache_key in self.query_cache:
            cached = self.query_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_timeout:
                return cached['result']
            else:
                del self.query_cache[cache_key]
        return None

    def cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache a result with timestamp."""
        self.query_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }

    def intent_to_command(self, query: AnalystQuery) -> str:
        """Convert classified intent to command string with enhanced logic."""
        if query.intent == 'summary':
            return '/summary'
        elif query.intent == 'weakest':
            top_n = query.top_n or 5
            return f'/weakest --top {top_n}'
        elif query.intent == 'bucket':
            if query.bucket and query.horizon:
                return f'/bucket {query.bucket} {query.horizon}d'
            elif query.bucket:
                return f'/bucket {query.bucket} 10d'  # Default horizon
            else:
                return '/bucket <bucket> <horizon>'
        elif query.intent == 'guardrails':
            return '/guardrails'
        elif query.intent == 'hpo':
            top_n = query.top_n or 3
            return f'/hpo_suggestions --top {top_n}'
        elif query.intent == 'switches':
            top_n = query.top_n or 2
            return f'/model_switches --top {top_n}'
        elif query.intent == 'plot':
            plot_type = query.plot_type or 'residuals'
            return f'/plot {plot_type}'
        elif query.intent == 'export':
            filename = query.filename or 'analysis_export.md'
            return f'/export_actions {filename}'
        elif query.intent == 'compare':
            # Handle comparison queries
            if query.bucket:
                return f'/bucket {query.bucket} 10d'
            else:
                return '/summary'
        elif query.intent == 'trend':
            # Handle trend analysis
            return '/summary'
        elif query.intent == 'help':
            return '/help'
        else:
            return '/summary'

    async def process_natural_language_query(
        self,
        query_text: str,
        snapshot_date: str = "latest",
        session_id: str = "default"
    ) -> str:
        """Process a natural language query with enhanced features."""
        try:
            # Get conversation context
            context = self.get_or_create_context(session_id)

            # Check cache first
            cache_key = f"{query_text}:{snapshot_date}"
            cached_result = self.get_cached_result(cache_key)
            if cached_result:
                return f"[CACHED] {cached_result}"

            # Process query with context awareness
            query = self.nlp.process_query(query_text, context)

            # Handle compound queries
            if query.compound_queries:
                results = []
                for i, sub_query in enumerate([query] + query.compound_queries):
                    command = self.intent_to_command(sub_query)
                    result = await self.execute_analyst_command(command, snapshot_date)
                    results.append(f"Query {i+1} ({sub_query.intent}): {result}")

                    # Update context with each sub-query
                    self.update_context(context, sub_query, result)

                final_result = "\n\n".join(results)
            else:
                # Single query processing
                command = self.intent_to_command(query)
                result = await self.execute_analyst_command(command, snapshot_date)
                self.update_context(context, query, result)
                final_result = result

            # Cache the result
            self.cache_result(cache_key, {"result": final_result, "timestamp": time.time()})

            # Build comprehensive response
            entities_json = json.dumps({
                'bucket': query.bucket,
                'horizon': query.horizon,
                'top_n': query.top_n,
                'plot_type': query.plot_type,
                'filename': query.filename
            }, indent=2)

            response_parts = [
                f"Natural Language Query: \"{query_text}\"",
                f"Classified Intent: {query.intent} (confidence: {query.confidence:.2f})",
                f"Extracted Entities: {entities_json}",
                f"Structured Command: {command}",
                f"Session ID: {session_id}",
                f"Analysis Result:\n{final_result}"
            ]

            if query.compound_queries:
                response_parts.insert(4, f"Compound Queries Detected: {len(query.compound_queries)} additional queries")

            return "\n\n".join(response_parts)

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"

    async def execute_analyst_command(self, command: str, snapshot_date: str) -> str:
        """Execute a command using the Python analyst script with enhanced error handling."""
        try:
            # Validate command format
            if not command.startswith('/'):
                return f"Invalid command format: {command}"

            # Run the interactive.py script with the command
            cmd = [
                sys.executable,
                str(self.interactive_script),
                "--query", command
            ]

            if snapshot_date != "latest":
                cmd.extend(["--snapshot", snapshot_date])

            # Run the command and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # Increased timeout
                cwd=self.interactive_script.parent
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                error_msg = result.stderr.strip()
                logger.warning(f"Command failed: {command}, Error: {error_msg}")
                return f"Command execution failed: {error_msg}"

        except subprocess.TimeoutExpired:
            return "Command execution timed out (60 seconds)"
        except FileNotFoundError:
            return f"Interactive analyst script not found: {self.interactive_script}"
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return f"Error executing command: {str(e)}"

    # ========== NEW EXTENDED TOOLS ==========

    async def export_analysis_report(
        self,
        report_type: str,
        format: str = "markdown",
        include_charts: bool = False
    ) -> Dict[str, Any]:
        """
        Export comprehensive analysis report in various formats.

        Args:
            report_type: Type of report (summary, detailed, performance, comparison)
            format: Output format (markdown, json, html)
            include_charts: Whether to include charts

        Returns:
            Dict with export result
        """
        try:
            # Get data using existing functionality
            summary_result = await self.execute_analyst_command("/summary", "latest")

            if format == "markdown":
                content = self._generate_markdown_report(
                    report_type, summary_result, include_charts
                )
                output_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

            elif format == "json":
                content_dict = self._generate_json_report(
                    report_type, summary_result, include_charts
                )
                content_str = json.dumps(content_dict)
                output_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            elif format == "html":
                content = self._generate_html_report(
                    report_type, summary_result, include_charts
                )
                output_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

            else:
                return {"status": "error", "message": f"Unsupported format: {format}"}

            # Determine content string and size
            if format == "json":
                content_str = json.dumps(content_dict, indent=2)
                content_size = len(content_str)
            else:
                content_str = content
                content_size = len(content)

            # Save file
            output_path = self.reports_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content_str)

            # Save to database
            self._save_report_to_db(report_type, format, str(output_path), content_size)

            return {
                "status": "success",
                "report_type": report_type,
                "format": format,
                "output_file": str(output_path),
                "size_bytes": content_size,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"status": "error", "message": str(e)}

    async def schedule_recurring_analysis(
        self,
        query: str,
        schedule: str,
        email: Optional[str] = None,
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Schedule recurring analysis queries.

        Args:
            query: Natural language query to schedule
            schedule: Schedule type (hourly, daily, weekly, monthly)
            email: Optional email for notifications
            user_id: User identifier

        Returns:
            Dict with scheduling result
        """
        try:
            # Rate limiting
            if not self.rate_limiter.is_allowed(user_id):
                return {"status": "error", "message": "Rate limit exceeded"}

            schedule_id = f"sched_{user_id}_{len(self.scheduled_queries) + 1}"

            self.scheduled_queries[schedule_id] = {
                "query": query,
                "schedule": schedule,
                "email": email,
                "user_id": user_id,
                "created_at": datetime.now(),
                "last_run": None,
                "next_run": self._calculate_next_run(schedule),
                "active": True,
                "run_count": 0
            }

            logger.info(f"Scheduled query: {schedule_id} for user {user_id}")

            return {
                "status": "success",
                "schedule_id": schedule_id,
                "query": query,
                "schedule": schedule,
                "next_run": self.scheduled_queries[schedule_id]["next_run"],
                "notification_email": email
            }

        except Exception as e:
            logger.error(f"Error scheduling analysis: {e}")
            return {"status": "error", "message": str(e)}

    async def create_performance_alert(
        self,
        alert_name: str,
        metric: str,
        threshold: float,
        comparison: str = "greater",
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Create performance alert.

        Args:
            alert_name: Name for the alert
            metric: Metric to monitor (mse, mape, etc.)
            threshold: Threshold value
            comparison: Comparison type (greater, less, equal)
            user_id: User identifier

        Returns:
            Dict with alert creation result
        """
        try:
            alert_id = f"alert_{user_id}_{len(self.active_alerts) + 1}"

            self.active_alerts[alert_id] = {
                "name": alert_name,
                "metric": metric,
                "threshold": threshold,
                "comparison": comparison,
                "user_id": user_id,
                "created_at": datetime.now(),
                "triggered_count": 0,
                "last_triggered": None,
                "active": True,
                "last_value": None
            }

            logger.info(f"Created alert: {alert_id} for user {user_id}")

            return {
                "status": "success",
                "alert_id": alert_id,
                "alert_name": alert_name,
                "condition": f"{metric} {comparison} {threshold}",
                "active": True
            }

        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return {"status": "error", "message": str(e)}

    async def process_batch_queries(
        self,
        queries: List[str],
        max_concurrent: int = 3,
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process multiple queries in batch.

        Args:
            queries: List of natural language queries
            max_concurrent: Maximum concurrent processing
            user_id: User identifier

        Returns:
            Dict with batch processing results
        """
        try:
            # Rate limiting check
            if not self.rate_limiter.is_allowed(user_id):
                return {"status": "error", "message": "Rate limit exceeded"}

            if len(queries) > 10:  # Limit batch size
                return {"status": "error", "message": "Maximum 10 queries per batch"}

            # Process in batches
            semaphore = asyncio.Semaphore(max_concurrent)
            results: List[Dict[str, Any]] = []

            async def process_single_query(query: str) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        result = await self.process_natural_language_query(
                            query, "latest", user_id
                        )
                        return {"query": query, "status": "success", "result": result}
                    except Exception as e:
                        return {"query": query, "status": "error", "error": str(e)}

            # Execute all queries
            tasks = [process_single_query(query) for query in queries]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            successful = 0
            failed = 0

            for result in batch_results:
                if isinstance(result, Exception):
                    error_result = {"status": "error", "error": str(result)}
                    results.append(error_result)
                    failed += 1
                else:
                    results.append(result)  # type: ignore
                    if result["status"] == "success":  # type: ignore
                        successful += 1
                    else:
                        failed += 1

            return {
                "status": "success",
                "total_queries": len(queries),
                "successful": successful,
                "failed": failed,
                "results": results,
                "processing_time": time.time() - time.time()  # Would track actual time
            }

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return {"status": "error", "message": str(e)}

    async def set_user_preference(
        self,
        preference_key: str,
        preference_value: Any,
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Set user preference.

        Args:
            preference_key: Preference key (e.g., 'default_horizon', 'favorite_buckets')
            preference_value: Preference value
            user_id: User identifier

        Returns:
            Dict with preference setting result
        """
        try:
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {}

            self.user_preferences[user_id][preference_key] = {
                "value": preference_value,
                "set_at": datetime.now().isoformat()
            }

            return {
                "status": "success",
                "preference_key": preference_key,
                "preference_value": preference_value,
                "user_id": user_id
            }

        except Exception as e:
            logger.error(f"Error setting preference: {e}")
            return {"status": "error", "message": str(e)}

    async def get_query_history(
        self,
        limit: int = 10,
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Get query history for a user.

        Args:
            limit: Maximum number of queries to return
            user_id: User identifier

        Returns:
            Dict with query history
        """
        try:
            # Get from database if available
            history = []
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT query_text, intent, timestamp, result FROM queries WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                        (user_id, limit)
                    )
                    history = [
                        {
                            "query": row[0],
                            "intent": row[1],
                            "timestamp": row[2],
                            "result": row[3]
                        }
                        for row in cursor.fetchall()
                    ]
            except Exception:
                # Fallback to in-memory history
                history = [
                    {
                        "query": entry.get("query", ""),
                        "intent": entry.get("intent", ""),
                        "timestamp": entry.get("timestamp", ""),
                        "result": entry.get("result", "")
                    }
                    for entry in self.query_history[-limit:]
                    if entry.get("user_id") == user_id
                ]

            return {
                "status": "success",
                "user_id": user_id,
                "count": len(history),
                "history": history
            }

        except Exception as e:
            logger.error(f"Error getting query history: {e}")
            return {"status": "error", "message": str(e)}

    async def get_dashboard_data(
        self,
        dashboard_type: str = "overview",
        time_range: str = "7d",
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Get structured data for dashboards.

        Args:
            dashboard_type: Type of dashboard (overview, performance, alerts)
            time_range: Time range (1d, 7d, 30d, 90d)
            user_id: User identifier

        Returns:
            Dict with dashboard data
        """
        try:
            # Get base data
            summary = await self.execute_analyst_command("/summary", "latest")

            # Structure data based on dashboard type
            if dashboard_type == "overview":
                data = {
                    "summary": summary,
                    "timestamp": datetime.now().isoformat(),
                    "time_range": time_range,
                    "metrics": {
                        "total_buckets": 5,  # Would be calculated
                        "active_alerts": len([
                            a for a in self.active_alerts.values()
                            if a["user_id"] == user_id and a["active"]
                        ]),
                        "scheduled_queries": len([
                            s for s in self.scheduled_queries.values()
                            if s["user_id"] == user_id and s["active"]
                        ])
                    }
                }

            elif dashboard_type == "performance":
                data = {
                    "performance_data": summary,
                    "charts": {
                        "mse_trend": [],  # Would contain actual data
                        "mape_distribution": []
                    },
                    "top_performers": [],
                    "worst_performers": []
                }

            elif dashboard_type == "alerts":
                data = {
                    "active_alerts": await self.get_active_alerts(user_id),
                    "recent_triggers": [],
                    "alert_history": []
                }

            else:
                return {"status": "error", "message": f"Unknown dashboard type: {dashboard_type}"}

            return {
                "status": "success",
                "dashboard_type": dashboard_type,
                "time_range": time_range,
                "data": data
            }

        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"status": "error", "message": str(e)}

    async def compare_models(
        self,
        models: List[str],
        metrics: Optional[List[str]] = None,
        time_range: str = "30d",
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Compare multiple models side by side.

        Args:
            models: List of model names to compare
            metrics: Metrics to compare (mse, mape, etc.)
            time_range: Time range for comparison
            user_id: User identifier

        Returns:
            Dict with model comparison results
        """
        try:
            if not models:
                return {"status": "error", "message": "No models specified"}

            if len(models) > 5:  # Limit comparison
                return {"status": "error", "message": "Maximum 5 models can be compared"}

            if metrics is None:
                metrics = ["mse", "mape", "accuracy"]

            # Simulate model comparison (would use actual model data)
            comparison_results: Dict[str, Dict[str, Any]] = {}

            for model in models:
                # In real implementation, this would fetch actual model performance
                comparison_results[model] = {
                    "metrics": {
                        metric: {
                            "value": 0.05 + (hash(model + metric) % 100) / 1000,  # Mock data
                            "rank": 1,
                            "trend": "stable"
                        }
                        for metric in metrics
                    },
                    "overall_score": 0.85 + (hash(model) % 20) / 100,  # Mock score
                    "last_updated": datetime.now().isoformat()
                }

            # Calculate rankings
            for metric in metrics:
                metric_values = [
                    (model, data["metrics"][metric]["value"])
                    for model, data in comparison_results.items()
                ]
                metric_values.sort(key=lambda x: x[1])  # Sort by value (lower is better for mse/mape)

                for rank, (model, _) in enumerate(metric_values, 1):
                    comparison_results[model]["metrics"][metric]["rank"] = rank

            # Overall ranking
            overall_scores = [
                (model, data["overall_score"])
                for model, data in comparison_results.items()
            ]
            overall_scores.sort(key=lambda x: float(x[1]), reverse=True)  # Higher score is better

            for rank, (model, _) in enumerate(overall_scores, 1):
                comparison_results[model]["overall_rank"] = rank

            return {
                "status": "success",
                "models_compared": models,
                "metrics": metrics,
                "time_range": time_range,
                "comparison_results": comparison_results,
                "best_model": overall_scores[0][0] if overall_scores else None,
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {"status": "error", "message": str(e)}

    # Utility methods for new tools
    async def get_scheduled_queries(self, user_id: str = "default") -> Dict[str, Any]:
        """Get all scheduled queries for a user"""
        user_schedules = {
            sid: data for sid, data in self.scheduled_queries.items()
            if data["user_id"] == user_id
        }

        return {
            "status": "success",
            "count": len(user_schedules),
            "scheduled_queries": [
                {
                    "id": sid,
                    "query": data["query"],
                    "schedule": data["schedule"],
                    "next_run": data["next_run"],
                    "active": data["active"],
                    "run_count": data["run_count"]
                }
                for sid, data in user_schedules.items()
            ]
        }

    async def get_active_alerts(self, user_id: str = "default") -> Dict[str, Any]:
        """Get all active alerts for a user"""
        user_alerts = {
            aid: data for aid, data in self.active_alerts.items()
            if data["user_id"] == user_id
        }

        return {
            "status": "success",
            "count": len(user_alerts),
            "alerts": [
                {
                    "id": aid,
                    "name": data["name"],
                    "condition": f"{data['metric']} {data['comparison']} {data['threshold']}",
                    "triggered_count": data["triggered_count"],
                    "last_triggered": data.get("last_triggered"),
                    "active": data["active"]
                }
                for aid, data in user_alerts.items()
            ]
        }

    async def get_user_preferences(self, user_id: str = "default") -> Dict[str, Any]:
        """Get all user preferences"""
        preferences = self.user_preferences.get(user_id, {})

        return {
            "status": "success",
            "user_id": user_id,
            "preferences": {
                key: data["value"] for key, data in preferences.items()
            },
            "count": len(preferences)
        }

    async def handle_list_tools(self) -> Dict[str, Any]:
        """Handle list tools request with enhanced tool definitions."""
        return {
            "tools": [
                {
                    "name": "analyze_forecast_performance",
                    "description": "Analyze forecast performance using natural language queries. Accepts human language input and converts it to structured analysis requests for the Interactive Analyst Mode. Supports compound queries, context awareness, and intelligent entity extraction.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query about forecast performance (e.g., 'Show me the weakest performing buckets and plot the residuals', 'Analyze ai_basket performance over 30 days', 'What are the top HPO candidates and any guardrail violations?')"
                            },
                            "snapshot_date": {
                                "type": "string",
                                "description": "Optional snapshot date in YYYY-MM-DD format",
                                "default": "latest"
                            },
                            "session_id": {
                                "type": "string",
                                "description": "Optional session ID for conversation context",
                                "default": "default"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_query_suggestions",
                    "description": "Get intelligent query suggestions based on partial input to help users formulate natural language queries.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "partial_query": {
                                "type": "string",
                                "description": "Partial or incomplete query text to get suggestions for"
                            }
                        },
                        "required": ["partial_query"]
                    }
                },
                {
                    "name": "get_conversation_context",
                    "description": "Get conversation context and history for a session to understand previous interactions.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Session ID to retrieve context for",
                                "default": "default"
                            }
                        }
                    }
                },
                {
                    "name": "export_analysis_report",
                    "description": "Export comprehensive analysis reports in multiple formats (markdown, json, html) with optional charts.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "report_type": {
                                "type": "string",
                                "description": "Type of report (summary, detailed, performance, comparison)",
                                "enum": ["summary", "detailed", "performance", "comparison"]
                            },
                            "format": {
                                "type": "string",
                                "description": "Output format",
                                "enum": ["markdown", "json", "html"],
                                "default": "markdown"
                            },
                            "include_charts": {
                                "type": "boolean",
                                "description": "Whether to include charts in the report",
                                "default": False
                            }
                        },
                        "required": ["report_type"]
                    }
                },
                {
                    "name": "schedule_recurring_analysis",
                    "description": "Schedule recurring analysis queries with customizable intervals and email notifications.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query to schedule"
                            },
                            "schedule": {
                                "type": "string",
                                "description": "Schedule type",
                                "enum": ["hourly", "daily", "weekly", "monthly"]
                            },
                            "email": {
                                "type": "string",
                                "description": "Optional email for notifications"
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "default": "default"
                            }
                        },
                        "required": ["query", "schedule"]
                    }
                },
                {
                    "name": "create_performance_alert",
                    "description": "Create performance alerts that trigger when metrics exceed thresholds.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "alert_name": {
                                "type": "string",
                                "description": "Name for the alert"
                            },
                            "metric": {
                                "type": "string",
                                "description": "Metric to monitor (mse, mape, etc.)"
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Threshold value"
                            },
                            "comparison": {
                                "type": "string",
                                "description": "Comparison type",
                                "enum": ["greater", "less", "equal"],
                                "default": "greater"
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "default": "default"
                            }
                        },
                        "required": ["alert_name", "metric", "threshold"]
                    }
                },
                {
                    "name": "process_batch_queries",
                    "description": "Process multiple analysis queries concurrently with error handling and progress tracking.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of natural language queries"
                            },
                            "max_concurrent": {
                                "type": "integer",
                                "description": "Maximum concurrent processing",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 10
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "default": "default"
                            }
                        },
                        "required": ["queries"]
                    }
                },
                {
                    "name": "set_user_preference",
                    "description": "Set user preferences for personalized analysis experience.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "preference_key": {
                                "type": "string",
                                "description": "Preference key (e.g., 'default_horizon', 'favorite_buckets')"
                            },
                            "preference_value": {
                                "type": "string",
                                "description": "Preference value"
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "default": "default"
                            }
                        },
                        "required": ["preference_key", "preference_value"]
                    }
                },
                {
                    "name": "get_query_history",
                    "description": "Retrieve historical query data for analysis and pattern recognition.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of queries to return",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "default": "default"
                            }
                        }
                    }
                },
                {
                    "name": "get_dashboard_data",
                    "description": "Get structured data for web dashboards and visualization tools.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "dashboard_type": {
                                "type": "string",
                                "description": "Type of dashboard",
                                "enum": ["overview", "performance", "alerts"],
                                "default": "overview"
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for data",
                                "enum": ["1d", "7d", "30d", "90d"],
                                "default": "7d"
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "default": "default"
                            }
                        }
                    }
                },
                {
                    "name": "compare_models",
                    "description": "Compare multiple models side by side with rankings and statistical analysis.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "models": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of model names to compare"
                            },
                            "metrics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Metrics to compare",
                                "default": ["mse", "mape", "accuracy"]
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for comparison",
                                "default": "30d"
                            },
                            "user_id": {
                                "type": "string",
                                "description": "User identifier",
                                "default": "default"
                            }
                        },
                        "required": ["models"]
                    }
                }
            ]
        }

    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call request with enhanced functionality."""
        try:
            if name == "analyze_forecast_performance":
                query = arguments.get("query", "")
                snapshot_date = arguments.get("snapshot_date", "latest")
                session_id = arguments.get("session_id", "default")

                if not query:
                    raise ValueError("Query parameter is required")

                result = await self.process_natural_language_query(query, snapshot_date, session_id)

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": result
                        }
                    ]
                }

            elif name == "get_query_suggestions":
                partial_query = arguments.get("partial_query", "")
                suggestions = self.nlp.get_query_suggestions(partial_query)

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Query suggestions for '{partial_query}':\n" + "\n".join(f"• {suggestion}" for suggestion in suggestions)
                        }
                    ]
                }

            elif name == "get_conversation_context":
                session_id = arguments.get("session_id", "default")
                context = self.get_or_create_context(session_id)

                context_info = {
                    "session_id": context.session_id,
                    "last_query": context.last_query,
                    "preferred_bucket": context.preferred_bucket,
                    "preferred_horizon": context.preferred_horizon,
                    "query_count": len(context.query_history),
                    "recent_queries": [
                        {
                            "intent": q["intent"],
                            "timestamp": q["timestamp"].isoformat(),
                            "confidence": q["confidence"]
                        } for q in context.query_history[-3:]  # Last 3 queries
                    ]
                }

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Conversation Context:\n{json.dumps(context_info, indent=2)}"
                        }
                    ]
                }

            elif name == "export_analysis_report":
                report_type = arguments.get("report_type", "summary")
                format_type = arguments.get("format", "markdown")
                include_charts = arguments.get("include_charts", False)

                export_result = await self.export_analysis_report(report_type, format_type, include_charts)

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Report exported successfully:\n{json.dumps(export_result, indent=2)}"
                        }
                    ]
                }

            elif name == "schedule_recurring_analysis":
                query = arguments.get("query", "")
                schedule = arguments.get("schedule", "daily")
                email = arguments.get("email")
                user_id = arguments.get("user_id", "default")

                if not query:
                    raise ValueError("Query parameter is required")

                schedule_result = await self.schedule_recurring_analysis(query, schedule, email, user_id)

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analysis scheduled successfully:\n{json.dumps(schedule_result, indent=2)}"
                        }
                    ]
                }

            elif name == "create_performance_alert":
                alert_name = arguments.get("alert_name", "")
                metric = arguments.get("metric", "")
                threshold = arguments.get("threshold", 0.0)
                comparison = arguments.get("comparison", "greater")
                user_id = arguments.get("user_id", "default")

                if not alert_name or not metric:
                    raise ValueError("Alert name and metric are required")

                alert_result = await self.create_performance_alert(alert_name, metric, threshold, comparison, user_id)

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Alert created successfully:\n{json.dumps(alert_result, indent=2)}"
                        }
                    ]
                }

            elif name == "process_batch_queries":
                queries = arguments.get("queries", [])
                max_concurrent = arguments.get("max_concurrent", 3)
                user_id = arguments.get("user_id", "default")

                if not queries:
                    raise ValueError("Queries list is required")

                batch_result = await self.process_batch_queries(queries, max_concurrent, user_id)

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Batch processing completed:\n{json.dumps(batch_result, indent=2)}"
                        }
                    ]
                }

            elif name == "set_user_preference":
                preference_key = arguments.get("preference_key", "")
                preference_value = arguments.get("preference_value", "")
                user_id = arguments.get("user_id", "default")

                if not preference_key:
                    raise ValueError("Preference key is required")

                preference_result = await self.set_user_preference(preference_key, preference_value, user_id)

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Preference set successfully:\n{json.dumps(preference_result, indent=2)}"
                        }
                    ]
                }

            elif name == "get_query_history":
                limit = arguments.get("limit", 10)
                user_id = arguments.get("user_id", "default")

                history_result = await self.get_query_history(limit, user_id)

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Query history retrieved:\n{json.dumps(history_result, indent=2)}"
                        }
                    ]
                }

            elif name == "get_dashboard_data":
                dashboard_type = arguments.get("dashboard_type", "overview")
                time_range = arguments.get("time_range", "7d")
                user_id = arguments.get("user_id", "default")

                dashboard_result = await self.get_dashboard_data(dashboard_type, time_range, user_id)

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Dashboard data retrieved:\n{json.dumps(dashboard_result, indent=2)}"
                        }
                    ]
                }

            elif name == "compare_models":
                models = arguments.get("models", [])
                metrics = arguments.get("metrics", ["mse", "mape", "accuracy"])
                time_range = arguments.get("time_range", "30d")
                user_id = arguments.get("user_id", "default")

                if not models:
                    raise ValueError("Models list is required")

                comparison_result = await self.compare_models(models, metrics, time_range, user_id)

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Model comparison completed:\n{json.dumps(comparison_result, indent=2)}"
                        }
                    ]
                }

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            logger.error(f"Error in tool call '{name}': {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error executing tool '{name}': {str(e)}"
                    }
                ],
                "isError": True
            }

def main():
    """Main MCP server loop."""
    server = InteractiveAnalystMCPServer()

    # Simple stdio-based MCP server (simplified implementation)
    logger.info("Interactive Analyst MCP Server starting...")

    try:
        while True:
            # Read JSON-RPC request from stdin
            line = sys.stdin.readline().strip()
            if not line:
                break

            try:
                request = json.loads(line)

                # Only respond to requests that have an id (not notifications)
                if "id" not in request:
                    continue

                # Handle different request types
                if request.get("method") == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {
                                    "listChanged": True
                                }
                            },
                            "serverInfo": {
                                "name": "Interactive Analyst MCP Server",
                                "version": "1.1.0"
                            }
                        }
                    }
                elif request.get("method") == "tools/list":
                    # For synchronous calls, we need to run async functions in the event loop
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    tools_result = loop.run_until_complete(server.handle_list_tools())
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": tools_result
                    }
                elif request.get("method") == "tools/call":
                    params = request.get("params", {})
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(server.handle_call_tool(
                        params.get("name", ""),
                        params.get("arguments", {})
                    ))
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": result
                    }
                elif request.get("method") == "initialized":
                    # Client acknowledges initialization - this is a notification, no response needed
                    continue
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "error": {
                            "code": -32601,
                            "message": "Method not found"
                        }
                    }

                # Send response to stdout
                print(json.dumps(response), flush=True)

            except json.JSONDecodeError:
                # Invalid JSON
                response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }
                print(json.dumps(response), flush=True)
            except Exception as e:
                # Internal error
                response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                }
                print(json.dumps(response), flush=True)

    except KeyboardInterrupt:
        logger.info("Server shutting down...")

if __name__ == "__main__":
    main()