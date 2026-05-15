"""
Trade Journal & Portfolio History
SQLite-backed log of all trades and daily portfolio snapshots.

DB path: ~/.portfolio_manager/journal.db
Tables:
  trades         — every buy/sell/rebalance event
  daily_snapshots — end-of-day portfolio value + benchmark close
"""

from __future__ import annotations

import sqlite3
import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

_DB_PATH = Path.home() / ".portfolio_manager" / "journal.db"


# ── Connection / schema ───────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,          -- ISO datetime
            date        TEXT    NOT NULL,          -- YYYY-MM-DD
            ticker      TEXT    NOT NULL,
            action      TEXT    NOT NULL,          -- BUY / SELL / REBALANCE / OPTION_BUY / OPTION_SELL
            shares      REAL,
            price       REAL,
            total_value REAL,
            cost_basis  REAL,
            regime      TEXT,
            notes       TEXT,
            source      TEXT DEFAULT 'manual',     -- manual / paper / imported
            strategy    TEXT,                      -- for options
            option_type TEXT,                      -- call / put
            strike      REAL,
            expiration  TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date              TEXT PRIMARY KEY,    -- YYYY-MM-DD
            portfolio_value   REAL,
            benchmark_close   REAL,               -- SPY close
            cash              REAL DEFAULT 0,
            positions_json    TEXT,               -- JSON snapshot of positions
            regime            TEXT
        )
    """)
    con.commit()
    return con


# ── Trade logging ─────────────────────────────────────────────────────────────

def log_trade(
    ticker: str,
    action: str,
    shares: float,
    price: float,
    cost_basis: float = 0.0,
    regime: str = "",
    notes: str = "",
    source: str = "manual",
    strategy: str = "",
    option_type: str = "",
    strike: float = 0.0,
    expiration: str = "",
) -> int:
    """
    Insert a trade record.  Returns the new row id.
    action: BUY | SELL | REBALANCE | OPTION_BUY | OPTION_SELL
    """
    now = datetime.now()
    total = round(shares * price, 4) if shares and price else 0.0
    con = _conn()
    cur = con.execute(
        """INSERT INTO trades
           (ts, date, ticker, action, shares, price, total_value, cost_basis,
            regime, notes, source, strategy, option_type, strike, expiration)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            now.isoformat(),
            now.strftime("%Y-%m-%d"),
            ticker.upper(),
            action.upper(),
            shares,
            price,
            total,
            cost_basis,
            regime,
            notes,
            source,
            strategy,
            option_type,
            strike if strike else None,
            expiration or None,
        ),
    )
    row_id = cur.lastrowid
    con.commit()
    con.close()
    return row_id


def get_trades(
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Return trades as a DataFrame, newest first."""
    con = _conn()
    clauses = []
    params: list = []
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper())
    if start_date:
        clauses.append("date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("date <= ?")
        params.append(end_date)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    df = pd.read_sql_query(
        f"SELECT * FROM trades {where} ORDER BY ts DESC LIMIT {limit}",
        con,
        params=params,
    )
    con.close()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def delete_trade(row_id: int) -> bool:
    """Delete a single trade by id."""
    try:
        con = _conn()
        con.execute("DELETE FROM trades WHERE id = ?", (row_id,))
        con.commit()
        con.close()
        return True
    except Exception:
        return False


# ── Daily snapshot ────────────────────────────────────────────────────────────

def save_daily_snapshot(
    portfolio_value: float,
    benchmark_close: float,
    positions_df: pd.DataFrame | None = None,
    cash: float = 0.0,
    regime: str = "",
) -> None:
    """Upsert today's portfolio snapshot."""
    today = date.today().isoformat()
    positions_json = positions_df.to_json(orient="records") if positions_df is not None else "[]"
    con = _conn()
    con.execute(
        """INSERT OR REPLACE INTO daily_snapshots
           (date, portfolio_value, benchmark_close, cash, positions_json, regime)
           VALUES (?,?,?,?,?,?)""",
        (today, portfolio_value, benchmark_close, cash, positions_json, regime),
    )
    con.commit()
    con.close()


def get_portfolio_history(days: int = 365) -> pd.DataFrame:
    """
    Return daily portfolio_value + benchmark_close for the past *days*.
    Indexed by date (datetime).
    """
    con = _conn()
    df = pd.read_sql_query(
        """SELECT date, portfolio_value, benchmark_close, cash, regime
           FROM daily_snapshots
           ORDER BY date ASC
           LIMIT ?""",
        con,
        params=(days,),
    )
    con.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def compute_equity_curve(history: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise portfolio and benchmark to 100 at first data point.
    Returns DataFrame with columns: portfolio_norm, benchmark_norm, alpha.
    """
    if history.empty:
        return pd.DataFrame()

    h = history[["portfolio_value", "benchmark_close"]].dropna()
    if h.empty:
        return pd.DataFrame()

    base_p = h["portfolio_value"].iloc[0]
    base_b = h["benchmark_close"].iloc[0]

    if base_p == 0 or base_b == 0:
        return pd.DataFrame()

    h = h.copy()
    h["portfolio_norm"]  = h["portfolio_value"]  / base_p * 100
    h["benchmark_norm"]  = h["benchmark_close"]  / base_b * 100
    h["alpha"]           = h["portfolio_norm"]   - h["benchmark_norm"]
    return h


def compute_trade_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """
    For each SELL trade, compute realized P&L using the cost_basis field.
    Returns DataFrame with realised_pnl, realised_pnl_pct added.
    """
    if trades.empty:
        return trades
    df = trades.copy()
    df["realised_pnl"] = None
    df["realised_pnl_pct"] = None
    mask = df["action"].isin(["SELL", "OPTION_SELL"])
    df.loc[mask, "realised_pnl"] = (
        (df.loc[mask, "price"] - df.loc[mask, "cost_basis"])
        * df.loc[mask, "shares"]
    )
    df.loc[mask, "realised_pnl_pct"] = (
        (df.loc[mask, "price"] / df.loc[mask, "cost_basis"] - 1) * 100
    ).where(df.loc[mask, "cost_basis"] > 0)
    return df
