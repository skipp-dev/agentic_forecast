#!/usr/bin/env python3
"""
Praktisches Beispiel: Integration neuer Tools in deinen bestehenden Server

Diese Datei zeigt dir Schritt für Schritt, wie du neue Tools hinzufügst.
"""

# ========== SCHRITT 1: Imports erweitern ==========
# Füge diese Imports zu deinem server.py hinzu:

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Optional für erweiterte Features
try:
    import aiohttp  # Für API calls
    import sqlite3  # Für Database
    from collections import OrderedDict  # Für LRU Cache
except ImportError:
    pass  # Diese sind optional

logger = logging.getLogger(__name__)


# ========== SCHRITT 2: Neue Methode zum Server hinzufügen ==========
# Füge diese Methode zur InteractiveAnalystMCPServer Klasse hinzu:

async def export_analysis_report(
    self,
    report_type: str,
    format: str = "markdown",
    include_charts: bool = False
) -> Dict[str, Any]:
    """
    NEU: Exportiere Analyse-Report in verschiedenen Formaten.

    Beispiel-Integration in deinen bestehenden Server.
    """
    try:
        # Verwende bestehende Funktionalität
        summary_cmd = "/summary"
        summary_result = await self.execute_analyst_command(summary_cmd, "latest")

        if format == "markdown":
            report_content = f"""# Forecast Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
{summary_result}

## Report Type: {report_type}
"""
            if include_charts:
                report_content += "\n## Charts\n[Charts would be embedded here]\n"

            # Speichere Report
            output_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            output_path = Path(__file__).parent / "reports" / output_file
            output_path.parent.mkdir(exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(report_content)

            return {
                "status": "success",
                "report_type": report_type,
                "format": format,
                "output_file": str(output_path),
                "size_bytes": len(report_content)
            }

        elif format == "json":
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "report_type": report_type,
                "summary": summary_result,
                "metadata": {
                    "include_charts": include_charts
                }
            }

            output_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path = Path(__file__).parent / "reports" / output_file
            output_path.parent.mkdir(exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            return {
                "status": "success",
                "report_type": report_type,
                "format": format,
                "output_file": str(output_path),
                "data": report_data
            }

        else:
            return {
                "status": "error",
                "message": f"Unsupported format: {format}"
            }

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


# ========== SCHRITT 3: Tool in handle_list_tools registrieren ==========
# Füge dieses Tool-Definition zu deiner handle_list_tools Methode hinzu:

"""
In der handle_list_tools Methode, füge zur "tools" Liste hinzu:

{
    "name": "export_analysis_report",
    "description": "Export comprehensive analysis report in various formats (Markdown, JSON, HTML)",
    "inputSchema": {
        "type": "object",
        "properties": {
            "report_type": {
                "type": "string",
                "description": "Type of report to generate",
                "enum": ["summary", "detailed", "performance", "comparison"],
                "default": "summary"
            },
            "format": {
                "type": "string",
                "description": "Output format",
                "enum": ["markdown", "json", "html"],
                "default": "markdown"
            },
            "include_charts": {
                "type": "boolean",
                "description": "Include visualization charts in report",
                "default": False
            }
        },
        "required": ["report_type"]
    }
}
"""


# ========== SCHRITT 4: Tool in handle_call_tool einbinden ==========
# Füge diesen elif-Block zu deiner handle_call_tool Methode hinzu:

"""
In der handle_call_tool Methode, füge hinzu:

elif name == "export_analysis_report":
    result = await self.export_analysis_report(
        arguments.get("report_type"),
        arguments.get("format", "markdown"),
        arguments.get("include_charts", False)
    )
"""


# ========== VOLLSTÄNDIGES INTEGRATION-BEISPIEL ==========

class IntegratedInteractiveAnalystMCPServer:
    """
    Beispiel: Wie dein Server nach der Integration aussehen könnte.

    Dies zeigt die Struktur - du würdest dies in deine existierende
    InteractiveAnalystMCPServer Klasse integrieren.
    """

    def __init__(self):
        # Bestehender Code
        self.nlp = None  # Dein NLP Processor
        self.interactive_script = Path(__file__).parent / "interactive.py"
        self.conversation_contexts = {}
        self.query_cache = {}
        self.cache_timeout = 300

        # NEU: Zusätzliche Features
        self.reports_dir = Path(__file__).parent / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        self.scheduled_queries = {}  # Für scheduled analysis
        self.active_alerts = {}      # Für alerts

    # ========== BESTEHENDE METHODEN ==========
    async def process_natural_language_query(self, query_text: str, snapshot_date: str = "latest", session_id: str = "default") -> str:
        """Deine bestehende Methode - bleibt unverändert"""
        pass

    async def execute_analyst_command(self, command: str, snapshot_date: str) -> str:
        """Deine bestehende Methode - bleibt unverändert"""
        pass

    # ========== NEUE METHODEN - FÜGE DIESE HINZU ==========

    async def export_analysis_report(self, report_type: str, format: str = "markdown", include_charts: bool = False) -> Dict[str, Any]:
        """NEU: Report Export Tool"""
        # Implementation wie oben
        pass

    async def schedule_recurring_analysis(self, query: str, schedule: str, email: Optional[str] = None) -> Dict[str, Any]:
        """
        NEU: Schedule recurring analysis.

        Diese Methode speichert scheduled queries und könnte mit einem
        Background Worker (z.B. Celery, APScheduler) integriert werden.
        """
        try:
            schedule_id = f"sched_{len(self.scheduled_queries) + 1}"

            self.scheduled_queries[schedule_id] = {
                "query": query,
                "schedule": schedule,
                "email": email,
                "created_at": datetime.now(),
                "last_run": None,
                "next_run": self._calculate_next_run(schedule),
                "active": True
            }

            logger.info(f"Scheduled query: {schedule_id}")

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

    def _calculate_next_run(self, schedule: str) -> str:
        """Helper: Calculate next run time"""
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

    async def create_performance_alert(self, alert_name: str, metric: str, threshold: float, comparison: str = "greater") -> Dict[str, Any]:
        """
        NEU: Create performance alerts.

        Diese Methode würde mit einem Monitoring-System integriert werden.
        """
        try:
            alert_id = f"alert_{len(self.active_alerts) + 1}"

            self.active_alerts[alert_id] = {
                "name": alert_name,
                "metric": metric,
                "threshold": threshold,
                "comparison": comparison,  # 'greater', 'less', 'equal'
                "created_at": datetime.now(),
                "triggered_count": 0,
                "last_triggered": None,
                "active": True
            }

            logger.info(f"Created alert: {alert_id}")

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

    async def get_scheduled_queries(self) -> Dict[str, Any]:
        """NEU: Get all scheduled queries"""
        return {
            "status": "success",
            "count": len(self.scheduled_queries),
            "scheduled_queries": [
                {
                    "id": sid,
                    "query": data["query"],
                    "schedule": data["schedule"],
                    "next_run": data["next_run"],
                    "active": data["active"]
                }
                for sid, data in self.scheduled_queries.items()
            ]
        }

    async def get_active_alerts(self) -> Dict[str, Any]:
        """NEU: Get all active alerts"""
        return {
            "status": "success",
            "count": len(self.active_alerts),
            "alerts": [
                {
                    "id": aid,
                    "name": data["name"],
                    "condition": f"{data['metric']} {data['comparison']} {data['threshold']}",
                    "triggered_count": data["triggered_count"],
                    "active": data["active"]
                }
                for aid, data in self.active_alerts.items()
            ]
        }

    # ========== ERWEITERTE handle_list_tools ==========
    async def handle_list_tools(self) -> Dict[str, Any]:
        """Erweiterte Tool-Liste mit allen neuen Tools"""
        return {
            "tools": [
                # BESTEHENDE TOOLS
                {
                    "name": "analyze_forecast_performance",
                    "description": "Analyze forecast performance using natural language queries",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "snapshot_date": {"type": "string", "default": "latest"},
                            "session_id": {"type": "string", "default": "default"}
                        },
                        "required": ["query"]
                    }
                },

                # NEUE TOOLS
                {
                    "name": "export_analysis_report",
                    "description": "Export comprehensive analysis report in various formats",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "report_type": {
                                "type": "string",
                                "enum": ["summary", "detailed", "performance", "comparison"]
                            },
                            "format": {
                                "type": "string",
                                "enum": ["markdown", "json", "html"],
                                "default": "markdown"
                            },
                            "include_charts": {"type": "boolean", "default": False}
                        },
                        "required": ["report_type"]
                    }
                },
                {
                    "name": "schedule_analysis",
                    "description": "Schedule recurring analysis queries",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "schedule": {
                                "type": "string",
                                "enum": ["hourly", "daily", "weekly", "monthly"]
                            },
                            "email": {"type": "string"}
                        },
                        "required": ["query", "schedule"]
                    }
                },
                {
                    "name": "create_alert",
                    "description": "Create performance alert",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "alert_name": {"type": "string"},
                            "metric": {"type": "string"},
                            "threshold": {"type": "number"},
                            "comparison": {
                                "type": "string",
                                "enum": ["greater", "less", "equal"],
                                "default": "greater"
                            }
                        },
                        "required": ["alert_name", "metric", "threshold"]
                    }
                },
                {
                    "name": "get_scheduled_queries",
                    "description": "Get list of all scheduled queries",
                    "inputSchema": {"type": "object", "properties": {}}
                },
                {
                    "name": "get_active_alerts",
                    "description": "Get list of all active alerts",
                    "inputSchema": {"type": "object", "properties": {}}
                }
            ]
        }

    # ========== ERWEITERTE handle_call_tool ==========
    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Erweiterte Tool-Handling mit allen neuen Tools"""
        try:
            # BESTEHENDE TOOLS
            if name == "analyze_forecast_performance":
                result = await self.process_natural_language_query(
                    arguments.get("query", ""),
                    arguments.get("snapshot_date", "latest"),
                    arguments.get("session_id", "default")
                )

            # NEUE TOOLS
            elif name == "export_analysis_report":
                result = await self.export_analysis_report(
                    arguments.get("report_type"),
                    arguments.get("format", "markdown"),
                    arguments.get("include_charts", False)
                )

            elif name == "schedule_analysis":
                result = await self.schedule_recurring_analysis(
                    arguments.get("query"),
                    arguments.get("schedule"),
                    arguments.get("email")
                )

            elif name == "create_alert":
                result = await self.create_performance_alert(
                    arguments.get("alert_name"),
                    arguments.get("metric"),
                    arguments.get("threshold"),
                    arguments.get("comparison", "greater")
                )

            elif name == "get_scheduled_queries":
                result = await self.get_scheduled_queries()

            elif name == "get_active_alerts":
                result = await self.get_active_alerts()

            else:
                raise ValueError(f"Unknown tool: {name}")

            # Format response
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2) if isinstance(result, dict) else result
                    }
                ]
            }

        except Exception as e:
            logger.error(f"Error in tool call '{name}': {e}")
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True
            }


# ========== TESTING CODE ==========

async def test_new_tools():
    """Test die neuen Tools"""
    print("Testing new tools integration...\n")

    server = IntegratedInteractiveAnalystMCPServer()

    # Test 1: Export Report
    print("1. Testing report export:")
    result = await server.export_analysis_report("summary", "markdown", False)
    print(json.dumps(result, indent=2))
    print()

    # Test 2: Schedule Analysis
    print("2. Testing scheduled analysis:")
    result = await server.schedule_recurring_analysis(
        "Show weakest buckets",
        "daily",
        "user@example.com"
    )
    print(json.dumps(result, indent=2))
    print()

    # Test 3: Create Alert
    print("3. Testing alert creation:")
    result = await server.create_performance_alert(
        "High MSE Alert",
        "mse",
        0.05,
        "greater"
    )
    print(json.dumps(result, indent=2))
    print()

    # Test 4: Get Scheduled Queries
    print("4. Testing get scheduled queries:")
    result = await server.get_scheduled_queries()
    print(json.dumps(result, indent=2))
    print()

    # Test 5: Get Active Alerts
    print("5. Testing get active alerts:")
    result = await server.get_active_alerts()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    print("=" * 70)
    print("Integration Example: Adding New Tools to MCP Server")
    print("=" * 70)
    print()

    asyncio.run(test_new_tools())

    print("\n" + "=" * 70)
    print("Integration Steps Summary:")
    print("=" * 70)
    print("""
1. ✅ Add new method to server class
2. ✅ Register tool in handle_list_tools()
3. ✅ Add handling in handle_call_tool()
4. ✅ Test the new tool
5. ✅ Update configuration if needed
6. ✅ Restart Claude Desktop

Your new tools are ready to use!
    """)