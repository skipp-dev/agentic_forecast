# 🚀 Schnellstart: MCP Server erweitern

## Übersicht der erstellten Dateien

1. **server_extended.py** - Vollständige Beispiel-Implementierung mit 8 neuen Tools
2. **SERVER_EXTENSION_GUIDE.md** - Umfassende Dokumentation mit Best Practices
3. **integration_example.py** - Praktisches Integrations-Beispiel
4. Diese Datei - Schnellstart-Anleitung

## 🎯 Schnelle Integration (15 Minuten)

### Option 1: Einzelnes Tool hinzufügen

Beispiel: Report Export Tool

**Schritt 1:** Füge die Methode zu deinem `server.py` hinzu:

```python
async def export_analysis_report(
    self,
    report_type: str,
    format: str = "markdown"
) -> Dict[str, Any]:
    try:
        # Nutze bestehende Funktionalität
        summary = await self.execute_analyst_command("/summary", "latest")

        # Erstelle Report
        if format == "markdown":
            content = f"# Analysis Report\n\n{summary}"
            filename = f"report_{datetime.now():%Y%m%d_%H%M%S}.md"

            with open(filename, 'w') as f:
                f.write(content)

            return {
                "status": "success",
                "file": filename,
                "format": format
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

**Schritt 2:** Registriere in `handle_list_tools`:

```python
# In der tools Liste, füge hinzu:
{
    "name": "export_report",
    "description": "Export analysis report",
    "inputSchema": {
        "type": "object",
        "properties": {
            "report_type": {"type": "string"},
            "format": {"type": "string", "default": "markdown"}
        },
        "required": ["report_type"]
    }
}
```

**Schritt 3:** Handle in `handle_call_tool`:

```python
elif name == "export_report":
    result = await self.export_analysis_report(
        arguments.get("report_type"),
        arguments.get("format", "markdown")
    )
```

**Schritt 4:** Teste es:

```bash
# Neustart nicht nötig, wenn Server läuft
# Rufe in Claude:
# "Export a summary report"
```

### Option 2: Mehrere Tools auf einmal

**Kopiere aus `server_extended.py`:**

1. Wähle Tools die du brauchst (z.B. export_data, schedule_analysis)
2. Kopiere die async def Methoden
3. Kopiere die Tool-Definitionen
4. Füge die Handler hinzu
5. Teste!

## 📊 Verfügbare Tool-Templates

### 1. Data Export Tool
- **Was:** Exportiert Daten in verschiedene Formate
- **Wofür:** Reports, Backups, Datenaustausch
- **Aufwand:** ⭐ Einfach (30 Min)

### 2. Scheduled Analysis Tool
- **Was:** Plant wiederkehrende Analysen
- **Wofür:** Automatische Reports, Monitoring
- **Aufwand:** ⭐⭐ Mittel (1 Std)

### 3. Alert System Tool
- **Was:** Erstellt Performance-Alerts
- **Wofür:** Proaktives Monitoring
- **Aufwand:** ⭐⭐ Mittel (1 Std)

### 4. Model Comparison Tool
- **Was:** Vergleicht mehrere Modelle
- **Wofür:** A/B Testing, Benchmarking
- **Aufwand:** ⭐⭐⭐ Komplex (2 Std)

### 5. Quality Report Tool
- **Was:** Datenqualitäts-Analyse
- **Wofür:** Data Validation, Quality Gates
- **Aufwand:** ⭐⭐ Mittel (1 Std)

### 6. Batch Processing Tool
- **Was:** Verarbeitet viele Queries auf einmal
- **Wofür:** Bulk Operations, Automation
- **Aufwand:** ⭐⭐ Mittel (1 Std)

### 7. User Preferences Tool
- **Was:** Verwaltet Benutzereinstellungen
- **Wofür:** Personalisierung
- **Aufwand:** ⭐ Einfach (30 Min)

### 8. Dashboard Data Tool
- **Was:** Strukturierte Daten für Dashboards
- **Wofür:** Web-Dashboards, Visualisierung
- **Aufwand:** ⭐⭐ Mittel (1 Std)

## 🔧 Erweiterte Funktionen

### Caching hinzufügen

```python
from functools import lru_cache

class InteractiveAnalystMCPServer:
    def __init__(self):
        # ... bestehender code
        self.cache = {}

    async def process_with_cache(self, query: str):
        if query in self.cache:
            return self.cache[query]

        result = await self.process_natural_language_query(query)
        self.cache[query] = result
        return result
```

### Logging hinzufügen

```python
import logging

# Am Anfang der Datei
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analyst_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# In deinen Methoden
logger.info(f"Processing query: {query}")
logger.error(f"Error occurred: {e}")
```

### Error Handling verbessern

```python
class AnalystError(Exception):
    """Custom exception für bessere Fehlerbehandlung"""
    pass

async def safe_execute(self, func, *args):
    try:
        return await func(*args)
    except AnalystError as e:
        logger.error(f"Analyst error: {e}")
        return {"status": "error", "type": "analyst", "message": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {"status": "error", "type": "internal", "message": "Internal error"}
```

### Async Performance verbessern

```python
import asyncio

async def process_multiple_buckets(self, buckets: List[str]):
    """Verarbeite mehrere Buckets parallel"""
    tasks = [self.analyze_bucket(b) for b in buckets]
    results = await asyncio.gather(*tasks)
    return results
```

## 🧪 Testing

### Unit Test Beispiel

```python
import pytest

@pytest.mark.asyncio
async def test_export_report():
    server = InteractiveAnalystMCPServer()
    result = await server.export_analysis_report("summary", "json")

    assert result["status"] == "success"
    assert result["format"] == "json"
    assert "output_path" in result
```

### Integration Test

```python
@pytest.mark.asyncio
async def test_full_workflow():
    server = InteractiveAnalystMCPServer()

    # Query
    query_result = await server.process_natural_language_query(
        "Show weakest buckets"
    )
    assert "weakest" in query_result.lower()

    # Export
    export_result = await server.export_analysis_report("summary", "markdown")
    assert export_result["status"] == "success"
```

## 📝 Checkliste für neue Tools

- [ ] Methode implementiert mit try/except
- [ ] Type hints für alle Parameter
- [ ] Docstring mit Args und Returns
- [ ] Tool in handle_list_tools registriert
- [ ] Vollständiges inputSchema definiert
- [ ] Handler in handle_call_tool hinzugefügt
- [ ] Error handling implementiert
- [ ] Logging hinzugefügt
- [ ] Unit Tests geschrieben
- [ ] Dokumentation aktualisiert
- [ ] Claude Desktop neugestartet

## 🚨 Häufige Fehler vermeiden

### ❌ Fehler 1: Sync statt Async
```python
# FALSCH
def process_query(self, query):
    return result

# RICHTIG
async def process_query(self, query):
    result = await self.some_async_operation()
    return result
```

### ❌ Fehler 2: Keine Error Handling
```python
# FALSCH
async def my_tool(self, param):
    result = await risky_operation(param)
    return result

# RICHTIG
async def my_tool(self, param):
    try:
        result = await risky_operation(param)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"status": "error", "message": str(e)}
```

### ❌ Fehler 3: Tool nicht registriert
```python
# Stelle sicher dass ALLE drei Schritte gemacht sind:
# 1. Methode implementiert
# 2. In handle_list_tools registriert
# 3. In handle_call_tool handled
```

## 🎓 Nächste Schritte

### Anfänger
1. Starte mit einem einfachen Tool (Export oder Preferences)
2. Teste es gründlich
3. Füge Logging hinzu
4. Dokumentiere deine Änderungen

### Fortgeschritten
1. Implementiere 2-3 Tools gleichzeitig
2. Füge Caching hinzu
3. Implementiere Rate Limiting
4. Erstelle Unit Tests

### Experte
1. Integriere externe APIs
2. Füge Datenbank-Layer hinzu
3. Implementiere Webhook-System
4. Erstelle Production-ready Setup mit Docker

## 📚 Weitere Ressourcen

- **server_extended.py** - Vollständige Implementierung aller Tools
- **SERVER_EXTENSION_GUIDE.md** - Detaillierte Best Practices
- **integration_example.py** - Schritt-für-Schritt Integration

## 💡 Tipps

1. **Klein anfangen:** Implementiere erst ein Tool, teste es, dann das nächste
2. **Logging ist wichtig:** Füge von Anfang an Logging hinzu
3. **Error Handling:** Jedes Tool sollte graceful fails haben
4. **Type Hints:** Macht den Code wartbarer und verhindert Fehler
5. **Tests schreiben:** Spart Zeit beim Debugging
6. **Dokumentieren:** Zukünftiges-Du wird es dir danken

## 🤝 Support

Wenn du Fragen hast oder Hilfe brauchst:
1. Schau in die Beispiel-Dateien
2. Teste mit kleinen Änderungen
3. Nutze die Logging-Ausgaben zum Debuggen
4. Frage Claude nach spezifischer Hilfe