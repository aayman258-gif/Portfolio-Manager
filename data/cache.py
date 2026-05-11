"""
SQLite TTL cache — shared across all pages.

Usage:
    from data.cache import cache_get, cache_set

    val = cache_get("my_key", ttl=3600)   # None if missing / expired
    if val is None:
        val = expensive_call()
        cache_set("my_key", val)

Supports any pickle-able Python object (DataFrame, Series, dict, scalar).
Cache file: data/cache.db (auto-created on first use).

TTL constants:
    TTL_PRICES     = 3 600  s  (1 h)   — intraday price bars
    TTL_OPTIONS    =   900  s  (15 min) — live options chain
    TTL_FRED_DAILY = 14 400 s  (4 h)   — daily FRED series
    TTL_FRED_MO    = 86 400 s  (24 h)  — monthly FRED series
    TTL_FUND       = 86 400 s  (24 h)  — fundamentals / screener
"""

from __future__ import annotations

import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any

_DB_PATH = Path(__file__).parent / "cache.db"

TTL_PRICES     =  3_600
TTL_OPTIONS    =    900
TTL_FRED_DAILY = 14_400
TTL_FRED_MO    = 86_400
TTL_FUND       = 86_400


# ── Connection helper ─────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS cache_v1 (
            key  TEXT    PRIMARY KEY,
            blob BLOB    NOT NULL,
            ts   REAL    NOT NULL
        )
    """)
    con.commit()
    return con


# ── Public API ────────────────────────────────────────────────────────────────

def cache_get(key: str, ttl: int) -> Any | None:
    """
    Return the cached value for *key* if it exists and was stored within
    *ttl* seconds.  Returns ``None`` on cache miss, expiry, or any error.
    """
    try:
        con = _conn()
        row = con.execute(
            "SELECT blob, ts FROM cache_v1 WHERE key = ?", (key,)
        ).fetchone()
        con.close()
        if row is None:
            return None
        blob, ts = row
        if time.time() - ts > ttl:
            return None
        return pickle.loads(blob)
    except Exception:
        return None


def cache_set(key: str, value: Any) -> None:
    """
    Persist *value* under *key* with the current timestamp.
    Silently ignores errors (cache is best-effort).
    """
    try:
        blob = pickle.dumps(value, protocol=4)
        con  = _conn()
        con.execute(
            "INSERT OR REPLACE INTO cache_v1 (key, blob, ts) VALUES (?, ?, ?)",
            (key, blob, time.time()),
        )
        con.commit()
        con.close()
    except Exception:
        pass


def cache_purge(max_age: int = 86_400 * 7) -> int:
    """
    Delete entries older than *max_age* seconds.
    Returns the number of rows removed.
    """
    try:
        con = _conn()
        cur = con.execute(
            "DELETE FROM cache_v1 WHERE ts < ?", (time.time() - max_age,)
        )
        deleted = cur.rowcount
        con.commit()
        con.close()
        return deleted
    except Exception:
        return 0


def cache_clear() -> None:
    """Wipe the entire cache (useful during development)."""
    try:
        con = _conn()
        con.execute("DELETE FROM cache_v1")
        con.commit()
        con.close()
    except Exception:
        pass
