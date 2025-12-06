"""
Checkpointer Factory

Provides persistent checkpointing for LangGraph workflows.
Supports SQLite for local persistence and Memory for testing.
"""

import os
import logging
from typing import Optional, Any
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3
    _HAS_SQLITE = True
except ImportError:
    _HAS_SQLITE = False

logger = logging.getLogger(__name__)

def get_checkpointer(persistence_type: str = "sqlite", db_path: str = "checkpoints.db") -> Any:
    """
    Get a checkpointer instance based on configuration.
    
    Args:
        persistence_type: 'sqlite', 'memory', or 'postgres' (future)
        db_path: Path to SQLite DB file (if using sqlite)
        
    Returns:
        A LangGraph checkpointer
    """
    if persistence_type == "memory":
        logger.info("Using in-memory checkpointer (state will be lost on restart)")
        return MemorySaver()
        
    if persistence_type == "sqlite":
        if not _HAS_SQLITE:
            logger.warning("langgraph.checkpoint.sqlite not found. Falling back to MemorySaver.")
            return MemorySaver()
            
        logger.info(f"Using SQLite checkpointer at {db_path}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
        
    # Default fallback
    logger.warning(f"Unknown persistence type '{persistence_type}'. Falling back to MemorySaver.")
    return MemorySaver()
