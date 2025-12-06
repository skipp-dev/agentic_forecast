import sqlite3
import pandas as pd
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
try:
    from src.data.contracts import MarketDataSchema
except ImportError:
    MarketDataSchema = None

logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Service for handling all database interactions.
    Supports both SQLite and PostgreSQL via SQLAlchemy.
    """
    
    def __init__(self, db_path: str = "data/forecast.db", connection_string: Optional[str] = None):
        """
        Initialize DatabaseService.
        
        Args:
            db_path: Path for SQLite database (fallback).
            connection_string: Full SQLAlchemy connection string (e.g., postgresql://user:pass@host/db).
                             If provided, overrides db_path.
        """
        if connection_string:
            self.engine = create_engine(connection_string)
            self.is_sqlite = connection_string.startswith("sqlite")
        else:
            # Default to SQLite
            self.db_path = db_path
            self._ensure_db_dir()
            self.engine = create_engine(f"sqlite:///{self.db_path}")
            self.is_sqlite = True
            
        self.Session = sessionmaker(bind=self.engine)
        self._initialize_db()
        
    def _ensure_db_dir(self):
        if hasattr(self, 'db_path'):
            dirname = os.path.dirname(self.db_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
        
    def _get_connection(self):
        # Legacy method for direct connection access, prefer using engine.connect() or Session
        if self.is_sqlite and hasattr(self, 'db_path'):
             return sqlite3.connect(self.db_path)
        return self.engine.raw_connection()
        
    def _initialize_db(self):
        """Initialize the database schema."""
        # Use SQLAlchemy inspector to check for tables
        inspector = inspect(self.engine)
        existing_tables = inspector.get_table_names()
        
        with self.engine.connect() as conn:
            # Market Data Table
            if 'market_data' not in existing_tables:
                conn.execute(text("""
                CREATE TABLE market_data (
                    symbol VARCHAR(20),
                    date VARCHAR(50),
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    volume FLOAT,
                    adjusted_close FLOAT,
                    sma FLOAT,
                    ema FLOAT,
                    rsi FLOAT,
                    macd FLOAT,
                    updated_at VARCHAR(50),
                    PRIMARY KEY (symbol, date)
                )
                """))
            
            # Forecasts Table
            if 'forecasts' not in existing_tables:
                conn.execute(text("""
                CREATE TABLE forecasts (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20),
                    model_type VARCHAR(50),
                    forecast_date VARCHAR(50),
                    generated_at VARCHAR(50),
                    value FLOAT,
                    horizon INTEGER
                )
                """ if not self.is_sqlite else """
                CREATE TABLE forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    model_type TEXT,
                    forecast_date TEXT,
                    generated_at TEXT,
                    value REAL,
                    horizon INTEGER
                )
                """))
            
            # Portfolio Table
            if 'portfolio' not in existing_tables:
                conn.execute(text("""
                CREATE TABLE portfolio (
                    id SERIAL PRIMARY KEY,
                    timestamp VARCHAR(50),
                    cash FLOAT,
                    total_value FLOAT,
                    is_backtest BOOLEAN
                )
                """ if not self.is_sqlite else """
                CREATE TABLE portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    cash REAL,
                    total_value REAL,
                    is_backtest BOOLEAN
                )
                """))
            
            # Positions Table
            if 'positions' not in existing_tables:
                conn.execute(text("""
                CREATE TABLE positions (
                    id SERIAL PRIMARY KEY,
                    portfolio_id INTEGER,
                    symbol VARCHAR(20),
                    quantity FLOAT,
                    avg_price FLOAT,
                    FOREIGN KEY(portfolio_id) REFERENCES portfolio(id)
                )
                """ if not self.is_sqlite else """
                CREATE TABLE positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER,
                    symbol TEXT,
                    quantity REAL,
                    avg_price REAL,
                    FOREIGN KEY(portfolio_id) REFERENCES portfolio(id)
                )
                """))
            
            # Trades Table
            if 'trades' not in existing_tables:
                conn.execute(text("""
                CREATE TABLE trades (
                    id VARCHAR(50) PRIMARY KEY,
                    symbol VARCHAR(20),
                    action VARCHAR(10),
                    quantity FLOAT,
                    price FLOAT,
                    timestamp VARCHAR(50),
                    status VARCHAR(20),
                    is_backtest BOOLEAN
                )
                """))
            
            conn.commit()
        
        conn.commit()
        conn.close()
        
    def save_market_data(self, symbol: str, df: pd.DataFrame):
        """
        Save market data to the database.
        Expects DataFrame with index as Date and columns matching schema.
        """
        if df.empty:
            return
            
        # Validate Data Contract
        if MarketDataSchema:
            try:
                # Prepare for validation (reset index to get date column)
                df_val = df.reset_index()
                
                # Handle unnamed index which becomes 'index' column
                if 'index' in df_val.columns and 'date' not in df_val.columns:
                     df_val = df_val.rename(columns={'index': 'date'})
                
                if 'date' not in df_val.columns and 'Date' in df_val.columns:
                    df_val = df_val.rename(columns={'Date': 'date'})
                elif df.index.name == 'date' or df.index.name == 'Date':
                    df_val = df_val.rename(columns={df.index.name: 'date'})
                
                df_val['symbol'] = symbol
                # Ensure date is datetime
                df_val['date'] = pd.to_datetime(df_val['date'])
                
                MarketDataSchema.validate(df_val)
            except Exception as e:
                logger.error(f"Data Contract Violation for {symbol}: {e}")
                # We can choose to raise or just log. For now, log and return to prevent bad data.
                return

        # Prepare DataFrame for insertion
        df_save = df.copy()
        df_save['symbol'] = symbol
        df_save['date'] = df_save.index.strftime('%Y-%m-%d')
        df_save['updated_at'] = datetime.now().isoformat()
        
        # Ensure all columns exist
        expected_cols = ['open', 'high', 'low', 'close', 'volume', 'adjusted_close', 
                         'sma', 'ema', 'rsi', 'macd']
        for col in expected_cols:
            if col not in df_save.columns:
                df_save[col] = None
                
        # Select columns in order
        cols_to_save = ['symbol', 'date', 'updated_at'] + expected_cols
        df_save = df_save[cols_to_save]
        
        # Use pandas to_sql with SQLAlchemy engine
        try:
            # For SQLite, we can use 'replace' if we want to overwrite the table, but we want to upsert rows.
            # Pandas doesn't support native upsert easily across DBs.
            # We will use a custom method or just append and handle duplicates if possible, 
            # but 'append' fails on PK constraint.
            
            # Simple approach for now: Delete existing for this symbol/date range and insert
            # This is not atomic but works for single-threaded ingestion
            
            with self.engine.connect() as conn:
                min_date = df_save['date'].min()
                max_date = df_save['date'].max()
                
                # Delete overlapping
                conn.execute(text("DELETE FROM market_data WHERE symbol = :symbol AND date >= :min_date AND date <= :max_date"),
                             {"symbol": symbol, "min_date": min_date, "max_date": max_date})
                conn.commit()
                
            df_save.to_sql('market_data', self.engine, if_exists='append', index=False)
            logger.info(f"Saved {len(df_save)} rows for {symbol} to DB")
            
        except Exception as e:
            logger.error(f"Error saving market data: {e}")

    def _upsert_market_data(self, table, conn, keys, data_iter):
        """Custom upsert method for SQLite."""
        data = [dict(zip(keys, row)) for row in data_iter]
        
        # Pandas might pass a connection or a cursor depending on version/driver
        if hasattr(conn, 'cursor'):
            try:
                cursor = conn.cursor()
            except AttributeError:
                # Fallback if it looks like a connection but behaves like a cursor (rare)
                cursor = conn
        else:
            cursor = conn
        
        sql = """
        INSERT OR REPLACE INTO market_data 
        (symbol, date, updated_at, open, high, low, close, volume, adjusted_close, sma, ema, rsi, macd)
        VALUES (:symbol, :date, :updated_at, :open, :high, :low, :close, :volume, :adjusted_close, :sma, :ema, :rsi, :macd)
        """
        
        cursor.executemany(sql, data)
        return cursor.rowcount

    def get_market_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve market data from DB."""
        query = "SELECT * FROM market_data WHERE symbol = :symbol"
        params = {"symbol": symbol}
        
        if start_date:
            query += " AND date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND date <= :end_date"
            params["end_date"] = end_date
            
        query += " ORDER BY date ASC"
        
        try:
            df = pd.read_sql(text(query), self.engine.connect(), params=params)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            return pd.DataFrame()

    def save_forecast(self, symbol: str, model_type: str, forecast_date: str, value: float, horizon: int = 1):
        """Save a single forecast."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                INSERT INTO forecasts (symbol, model_type, forecast_date, generated_at, value, horizon)
                VALUES (:symbol, :model_type, :forecast_date, :generated_at, :value, :horizon)
                """), {
                    "symbol": symbol, 
                    "model_type": model_type, 
                    "forecast_date": forecast_date, 
                    "generated_at": datetime.now().isoformat(), 
                    "value": float(value), 
                    "horizon": horizon
                })
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving forecast: {e}")

    def get_latest_portfolio(self, is_backtest: bool = False) -> Dict[str, Any]:
        """Get the most recent portfolio state."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                SELECT id, cash, total_value FROM portfolio 
                WHERE is_backtest = :is_backtest 
                ORDER BY timestamp DESC LIMIT 1
                """), {"is_backtest": is_backtest})
                
                row = result.fetchone()
                if not row:
                    return None
                    
                portfolio_id, cash, total_value = row
                
                # Get positions
                pos_result = conn.execute(text("""
                SELECT symbol, quantity, avg_price FROM positions WHERE portfolio_id = :portfolio_id
                """), {"portfolio_id": portfolio_id})
                
                positions = {}
                for sym, qty, price in pos_result.fetchall():
                    positions[sym] = qty 
                    
                return {
                    "cash": cash,
                    "positions": positions,
                    "total_value": total_value
                }
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return None

    def save_portfolio_state(self, cash: float, positions: Dict[str, float], total_value: float, is_backtest: bool = False):
        """Save current portfolio state."""
        try:
            with self.engine.connect() as conn:
                # Insert Portfolio
                result = conn.execute(text("""
                INSERT INTO portfolio (timestamp, cash, total_value, is_backtest)
                VALUES (:timestamp, :cash, :total_value, :is_backtest)
                """ if not self.is_sqlite else """
                INSERT INTO portfolio (timestamp, cash, total_value, is_backtest)
                VALUES (:timestamp, :cash, :total_value, :is_backtest)
                RETURNING id
                """), {
                    "timestamp": datetime.now().isoformat(), 
                    "cash": cash, 
                    "total_value": total_value, 
                    "is_backtest": is_backtest
                })
                
                # Get ID
                if self.is_sqlite:
                    portfolio_id = result.fetchone()[0]
                else:
                    # For Postgres, we need RETURNING id in the query above, but let's handle it generically if possible
                    # SQLAlchemy usually handles this with result.inserted_primary_key but raw SQL needs care.
                    # Let's assume RETURNING works for both (SQLite >= 3.35)
                    # If older SQLite, we might need cursor.lastrowid logic which is harder with engine.
                    # For now, let's assume modern SQLite.
                    portfolio_id = result.fetchone()[0]
                
                # Insert Positions
                for symbol, qty in positions.items():
                    conn.execute(text("""
                    INSERT INTO positions (portfolio_id, symbol, quantity, avg_price)
                    VALUES (:portfolio_id, :symbol, :quantity, :avg_price)
                    """), {
                        "portfolio_id": portfolio_id, 
                        "symbol": symbol, 
                        "quantity": qty, 
                        "avg_price": 0.0
                    })
                    
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
            # Fallback for older SQLite if RETURNING fails
            if "syntax error" in str(e).lower() and self.is_sqlite:
                 logger.warning("Fallback to legacy SQLite insert for portfolio")
                 # ... (legacy code could go here, but let's assume environment is modern)

    def log_trade(self, trade: Dict[str, Any], is_backtest: bool = False):
        """Log a trade execution."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                INSERT INTO trades (id, symbol, action, quantity, price, timestamp, status, is_backtest)
                VALUES (:id, :symbol, :action, :quantity, :price, :timestamp, :status, :is_backtest)
                """), {
                    "id": trade.get('order_id'),
                    "symbol": trade.get('symbol'),
                    "action": trade.get('action'),
                    "quantity": trade.get('quantity'),
                    "price": trade.get('price'),
                    "timestamp": trade.get('timestamp'),
                    "status": trade.get('status'),
                    "is_backtest": is_backtest
                })
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging trade: {e}")

    def clear_backtest_data(self):
        """Clear all backtest data from the database."""
        try:
            with self.engine.connect() as conn:
                # Delete positions associated with backtest portfolios
                conn.execute(text("DELETE FROM positions WHERE portfolio_id IN (SELECT id FROM portfolio WHERE is_backtest = :is_backtest)"), {"is_backtest": True})
                # Delete backtest portfolios
                conn.execute(text("DELETE FROM portfolio WHERE is_backtest = :is_backtest"), {"is_backtest": True})
                # Delete backtest trades
                conn.execute(text("DELETE FROM trades WHERE is_backtest = :is_backtest"), {"is_backtest": True})
                
                conn.commit()
            logger.info("Cleared backtest data from database.")
        except Exception as e:
            logger.error(f"Error clearing backtest data: {e}")
