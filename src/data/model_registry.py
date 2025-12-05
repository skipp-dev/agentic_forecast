import sqlite3
import json
import os
import time
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Manages model metadata using SQLite for robustness and concurrency.
    Replaces the JSON-based registry.
    """
    def __init__(self, registry_path: str = "data/model_registry.db"):
        self.registry_path = registry_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        
        # Models table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            version_id TEXT NOT NULL,
            model_family TEXT NOT NULL,
            artifact_path TEXT,
            metrics_json TEXT,
            params_json TEXT,
            created_at REAL,
            is_champion BOOLEAN DEFAULT 0,
            champion_reason TEXT,
            valid_until REAL
        )
        ''')
        
        # Metadata table for symbol-level info (like last_hpo_run)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS symbol_metadata (
            symbol TEXT PRIMARY KEY,
            last_hpo_run_ts REAL,
            updated_at REAL
        )
        ''')

        # System settings table for global flags like Kill Switch
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at REAL
        )
        ''')
        
        # Initialize trading_enabled if not present (Default: True)
        cursor.execute('''
        INSERT OR IGNORE INTO system_settings (key, value, updated_at)
        VALUES ('trading_enabled', 'true', ?)
        ''', (time.time(),))
        
        conn.commit()
        conn.close()

    def _get_conn(self):
        return sqlite3.connect(self.registry_path)

    def register_model(self, symbol: str, model_family: str, artifact_path: str, 
                      metrics: Dict[str, float], params: Dict[str, Any], 
                      is_champion: bool = False, valid_until: Optional[float] = None):
        """
        Registers a new model version.
        """
        version_id = f"{model_family}_{int(time.time())}"
        created_at = time.time()
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
            # If this is a new champion, demote previous champions for this symbol
            if is_champion:
                cursor.execute('''
                UPDATE models SET is_champion = 0 WHERE symbol = ?
                ''', (symbol,))
            
            cursor.execute('''
            INSERT INTO models (symbol, version_id, model_family, artifact_path, 
                              metrics_json, params_json, created_at, is_champion, valid_until)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, version_id, model_family, artifact_path, 
                  json.dumps(metrics), json.dumps(params), created_at, is_champion, valid_until))
            
            conn.commit()
            logger.info(f"Registered model {version_id} for {symbol} (Champion: {is_champion})")
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_champion_model(self, symbol: str) -> Optional[str]:
        """
        Returns the model family of the champion model.
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT model_family FROM models 
        WHERE symbol = ? AND is_champion = 1 
        ORDER BY created_at DESC LIMIT 1
        ''', (symbol,))
        
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else None

    def get_champion_details(self, symbol: str) -> Dict[str, Any]:
        """
        Returns full details of the champion model.
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT version_id, model_family, artifact_path, metrics_json, params_json, created_at, valid_until
        FROM models 
        WHERE symbol = ? AND is_champion = 1 
        ORDER BY created_at DESC LIMIT 1
        ''', (symbol,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "version": row[0],
                "model_family": row[1],
                "artifact_path": row[2],
                "metrics": json.loads(row[3]) if row[3] else {},
                "hyperparameters": json.loads(row[4]) if row[4] else {},
                "created_at": row[5],
                "valid_until": row[6]
            }
        return {}

    def get_last_hpo_run(self, symbol: str) -> Optional[float]:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('SELECT last_hpo_run_ts FROM symbol_metadata WHERE symbol = ?', (symbol,))
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else None

    def set_last_hpo_run(self, symbol: str, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
            
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO symbol_metadata (symbol, last_hpo_run_ts, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            last_hpo_run_ts = excluded.last_hpo_run_ts,
            updated_at = excluded.updated_at
        ''', (symbol, timestamp, time.time()))
        
        conn.commit()
        conn.close()

    def set_champion_model(self, symbol: str, model_family: str, reason: Optional[str] = None):
        """
        Legacy support: Promotes the latest model of a family to champion, or creates a placeholder.
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Try to find the latest model of this family
        cursor.execute('''
        SELECT id FROM models 
        WHERE symbol = ? AND model_family = ? 
        ORDER BY created_at DESC LIMIT 1
        ''', (symbol, model_family))
        
        row = cursor.fetchone()
        
        if row:
            # Demote others
            cursor.execute('UPDATE models SET is_champion = 0 WHERE symbol = ?', (symbol,))
            # Promote this one
            cursor.execute('UPDATE models SET is_champion = 1, champion_reason = ? WHERE id = ?', (reason, row[0]))
        else:
            # Create a placeholder champion (legacy behavior)
            version_id = f"{model_family}_legacy_{int(time.time())}"
            cursor.execute('UPDATE models SET is_champion = 0 WHERE symbol = ?', (symbol,))
            cursor.execute('''
            INSERT INTO models (symbol, version_id, model_family, created_at, is_champion, champion_reason)
            VALUES (?, ?, ?, ?, 1, ?)
            ''', (symbol, version_id, model_family, time.time(), reason))
            
        conn.commit()
        conn.close()

    def get_trading_status(self) -> bool:
        """
        Check if trading is globally enabled (Kill Switch).
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_settings WHERE key = 'trading_enabled'")
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return row[0].lower() == 'true'
        return True # Default to true if missing (though init should catch this)

    def set_trading_status(self, enabled: bool):
        """
        Set the global trading status (Kill Switch).
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        value = 'true' if enabled else 'false'
        cursor.execute('''
        INSERT INTO system_settings (key, value, updated_at)
        VALUES ('trading_enabled', ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        ''', (value, time.time()))
        conn.commit()
        conn.close()
        logger.info(f"Global Trading Status set to: {enabled}")
