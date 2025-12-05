import sqlite3
import pandas as pd
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Service for handling all database interactions.
    Currently uses SQLite, but designed to be extensible to Postgres.
    """
    
    def __init__(self, db_path: str = "data/forecast.db"):
        self.db_path = db_path
        self._ensure_db_dir()
        self._initialize_db()
        
    def _ensure_db_dir(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
        
    def _initialize_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Market Data Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            adjusted_close REAL,
            sma REAL,
            ema REAL,
            rsi REAL,
            macd REAL,
            updated_at TEXT,
            PRIMARY KEY (symbol, date)
        )
        """)
        
        # Forecasts Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            model_type TEXT,
            forecast_date TEXT,
            generated_at TEXT,
            value REAL,
            horizon INTEGER
        )
        """)
        
        # Portfolio Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            cash REAL,
            total_value REAL,
            is_backtest BOOLEAN
        )
        """)
        
        # Positions Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER,
            symbol TEXT,
            quantity REAL,
            avg_price REAL,
            FOREIGN KEY(portfolio_id) REFERENCES portfolio(id)
        )
        """)
        
        # Trades Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            symbol TEXT,
            action TEXT,
            quantity REAL,
            price REAL,
            timestamp TEXT,
            status TEXT,
            is_backtest BOOLEAN
        )
        """)
        
        conn.commit()
        conn.close()
        
    def save_market_data(self, symbol: str, df: pd.DataFrame):
        """
        Save market data to the database.
        Expects DataFrame with index as Date and columns matching schema.
        """
        if df.empty:
            return
            
        conn = self._get_connection()
        
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
        
        # Upsert (Replace)
        # SQLite doesn't have a simple UPSERT in older versions, but REPLACE works
        # Or we can use INSERT OR REPLACE
        try:
            df_save.to_sql('market_data', conn, if_exists='append', index=False, method=self._upsert_market_data)
            logger.info(f"Saved {len(df_save)} rows for {symbol} to DB")
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
        finally:
            conn.close()

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
        conn = self._get_connection()
        
        query = "SELECT * FROM market_data WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date ASC"
        
        try:
            df = pd.read_sql(query, conn, params=params)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                # Drop symbol and updated_at if not needed for analysis, or keep them
                # Usually we want just the numeric data for analysis
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def save_forecast(self, symbol: str, model_type: str, forecast_date: str, value: float, horizon: int = 1):
        """Save a single forecast."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
            INSERT INTO forecasts (symbol, model_type, forecast_date, generated_at, value, horizon)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, model_type, forecast_date, datetime.now().isoformat(), float(value), horizon))
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving forecast: {e}")
        finally:
            conn.close()

    def get_latest_portfolio(self, is_backtest: bool = False) -> Dict[str, Any]:
        """Get the most recent portfolio state."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
            SELECT id, cash, total_value FROM portfolio 
            WHERE is_backtest = ? 
            ORDER BY timestamp DESC LIMIT 1
            """, (is_backtest,))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            portfolio_id, cash, total_value = row
            
            # Get positions
            cursor.execute("""
            SELECT symbol, quantity, avg_price FROM positions WHERE portfolio_id = ?
            """, (portfolio_id,))
            
            positions = {}
            for sym, qty, price in cursor.fetchall():
                positions[sym] = qty # We could store price too if needed
                
            return {
                "cash": cash,
                "positions": positions,
                "total_value": total_value
            }
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            return None
        finally:
            conn.close()

    def save_portfolio_state(self, cash: float, positions: Dict[str, float], total_value: float, is_backtest: bool = False):
        """Save current portfolio state."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Insert Portfolio
            cursor.execute("""
            INSERT INTO portfolio (timestamp, cash, total_value, is_backtest)
            VALUES (?, ?, ?, ?)
            """, (datetime.now().isoformat(), cash, total_value, is_backtest))
            
            portfolio_id = cursor.lastrowid
            
            # Insert Positions
            for symbol, qty in positions.items():
                cursor.execute("""
                INSERT INTO positions (portfolio_id, symbol, quantity, avg_price)
                VALUES (?, ?, ?, ?)
                """, (portfolio_id, symbol, qty, 0.0)) # We don't track avg_price in simple dict yet
                
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
        finally:
            conn.close()

    def log_trade(self, trade: Dict[str, Any], is_backtest: bool = False):
        """Log a trade execution."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
            INSERT INTO trades (id, symbol, action, quantity, price, timestamp, status, is_backtest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get('order_id'),
                trade.get('symbol'),
                trade.get('action'),
                trade.get('quantity'),
                trade.get('price'),
                trade.get('timestamp'),
                trade.get('status'),
                is_backtest
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
        finally:
            conn.close()
