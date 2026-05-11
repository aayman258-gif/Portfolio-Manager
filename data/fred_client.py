"""
FRED Macro Data Client
Fetches public time-series from the St. Louis Fed FRED API.

API key resolution order (first found wins):
  1. Environment variable   FRED_API_KEY
  2. Streamlit secrets      FRED_API_KEY
  3. No key → falls back to the public fredgraph.csv endpoint (no key needed,
     but rate-limited and occasionally unreliable)

All series are cached in data/cache.db with appropriate TTLs.

Series catalogue
────────────────
  T10Y2Y       — 10Y-2Y Treasury yield spread    (daily, %)
  DGS10        — 10-Year Treasury yield           (daily, %)
  DGS2         — 2-Year Treasury yield            (daily, %)
  BAMLH0A0HYM2 — ICE BofA US HY OAS              (daily, %)
  FEDFUNDS     — Federal Funds effective rate     (monthly, %)
  CPIAUCSL     — CPI All Urban Consumers SA       (monthly index)
  UNRATE       — Unemployment rate                (monthly, %)

Derived series (computed here):
  CPI_YOY      — CPI YoY % change
  REAL_RATE    — FEDFUNDS − CPI_YOY (approximate real rate)
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

from data.cache import cache_get, cache_set, TTL_FRED_DAILY, TTL_FRED_MO

_API_BASE = "https://api.stlouisfed.org/fred/series/observations"
_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_TIMEOUT  = 12  # seconds per request

# Series → (ttl, lookback_years)
_SERIES_CONF: dict[str, tuple[int, int]] = {
    "T10Y2Y":       (TTL_FRED_DAILY, 5),
    "DGS10":        (TTL_FRED_DAILY, 5),
    "DGS2":         (TTL_FRED_DAILY, 5),
    "BAMLH0A0HYM2": (TTL_FRED_DAILY, 5),
    "FEDFUNDS":     (TTL_FRED_MO,    5),
    "CPIAUCSL":     (TTL_FRED_MO,    6),   # extra year for YoY
    "UNRATE":       (TTL_FRED_MO,    5),
}


# ── API key resolution ────────────────────────────────────────────────────────

def _get_api_key() -> str | None:
    """Return FRED API key from env → Streamlit secrets → None."""
    key = os.getenv("FRED_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st
        key = st.secrets.get("FRED_API_KEY", "").strip()
        if key:
            return key
    except Exception:
        pass
    return None


# ── Low-level fetch ───────────────────────────────────────────────────────────

def _fetch_via_api(series_id: str, api_key: str, lookback_years: int) -> pd.Series:
    """Fetch series from FRED JSON API using an API key."""
    cutoff = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
    params = {
        "series_id":    series_id,
        "api_key":      api_key,
        "file_type":    "json",
        "observation_start": cutoff,
        "sort_order":   "asc",
    }
    resp = requests.get(_API_BASE, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()

    data = resp.json()
    if "error_code" in data:
        raise ValueError(f"FRED API error {data['error_code']}: {data.get('error_message')}")

    observations = data.get("observations", [])
    if not observations:
        raise ValueError(f"No observations returned for {series_id}")

    records = {
        pd.Timestamp(o["date"]): float(o["value"])
        for o in observations
        if o["value"] != "."
    }
    s = pd.Series(records, name=series_id)
    s.index.name = "DATE"
    return s.dropna()


def _fetch_via_csv(series_id: str, lookback_years: int) -> pd.Series:
    """Fallback: fetch series from public fredgraph.csv (no API key)."""
    resp = requests.get(_CSV_BASE, params={"id": series_id}, timeout=_TIMEOUT)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "DATE"
    df.columns = [series_id]
    df = df.replace(".", np.nan)
    s  = pd.to_numeric(df[series_id], errors="coerce")

    cutoff = pd.Timestamp.now() - pd.DateOffset(years=lookback_years)
    return s[s.index >= cutoff].dropna()


def _fetch_series(series_id: str, ttl: int, lookback_years: int = 5) -> pd.Series:
    """
    Return a pd.Series for *series_id*, indexed by date.
    Checks SQLite cache first; downloads from FRED on miss.
    Uses the authenticated JSON API when a key is available,
    falling back to the public CSV endpoint otherwise.
    Raises on unrecoverable errors — callers must handle.
    """
    key = f"fred:{series_id}"
    cached = cache_get(key, ttl)
    if cached is not None:
        return cached

    api_key = _get_api_key()
    if api_key:
        s = _fetch_via_api(series_id, api_key, lookback_years)
    else:
        s = _fetch_via_csv(series_id, lookback_years)

    cache_set(key, s)
    return s


# ── Public accessors ──────────────────────────────────────────────────────────

def get_all_macro() -> dict[str, pd.Series]:
    """
    Fetch all macro series.  Individual entries may be empty pd.Series on
    network errors — callers should check `.empty` before use.

    Returns
    -------
    dict with keys matching _SERIES_CONF plus derived:
        CPI_YOY, REAL_RATE
    """
    result: dict[str, pd.Series] = {}
    for sid, (ttl, years) in _SERIES_CONF.items():
        try:
            result[sid] = _fetch_series(sid, ttl, years)
        except Exception as exc:
            print(f"[fred] {sid} unavailable: {exc}")
            result[sid] = pd.Series(dtype=float, name=sid)

    # Derived series
    cpi = result.get("CPIAUCSL", pd.Series(dtype=float))
    fed = result.get("FEDFUNDS",  pd.Series(dtype=float))
    if not cpi.empty:
        cpi_yoy = cpi.pct_change(12) * 100
        result["CPI_YOY"] = cpi_yoy.dropna()
    else:
        result["CPI_YOY"] = pd.Series(dtype=float)

    if not fed.empty and not result["CPI_YOY"].empty:
        common = fed.index.intersection(result["CPI_YOY"].index)
        result["REAL_RATE"] = (fed.loc[common] - result["CPI_YOY"].loc[common]).dropna()
    else:
        result["REAL_RATE"] = pd.Series(dtype=float)

    return result


def get_macro_snapshot(macro: Optional[dict[str, pd.Series]] = None,
                       ref: Optional[pd.Timestamp] = None) -> dict[str, float] | None:
    """
    Return a dict of current scalar readings for all macro signals.

    Parameters
    ----------
    macro : pre-fetched dict from get_all_macro() (fetched internally if None)
    ref   : reference timestamp; defaults to now

    Returns None if core series (yield curve + credit) are both missing.
    """
    if macro is None:
        try:
            macro = get_all_macro()
        except Exception:
            return None

    ref = ref or pd.Timestamp.now()

    def _latest(s: pd.Series) -> float | None:
        if s is None or s.empty:
            return None
        valid = s[s.index <= ref]
        return float(valid.iloc[-1]) if not valid.empty else None

    spread = _latest(macro.get("T10Y2Y",       pd.Series(dtype=float)))
    hy     = _latest(macro.get("BAMLH0A0HYM2", pd.Series(dtype=float)))
    dgs10  = _latest(macro.get("DGS10",         pd.Series(dtype=float)))
    dgs2   = _latest(macro.get("DGS2",          pd.Series(dtype=float)))
    fed    = _latest(macro.get("FEDFUNDS",       pd.Series(dtype=float)))
    cpi    = _latest(macro.get("CPI_YOY",        pd.Series(dtype=float)))
    real   = _latest(macro.get("REAL_RATE",      pd.Series(dtype=float)))
    unemp  = _latest(macro.get("UNRATE",         pd.Series(dtype=float)))

    if spread is None and hy is None:
        return None

    return {
        "yield_spread":  spread,
        "hy_spread":     hy,
        "dgs10":         dgs10,
        "dgs2":          dgs2,
        "fed_funds":     fed,
        "cpi_yoy":       cpi,
        "real_rate":     real,
        "unemployment":  unemp,
    }


def align_macro_to_index(macro: dict[str, pd.Series],
                          idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Forward-fill all macro series to a daily price index.
    Returns a DataFrame with one column per series, same DatetimeIndex as idx.
    """
    frames = {}
    for name, s in macro.items():
        if s.empty:
            continue
        s_clean = s.copy()
        s_clean.index = pd.to_datetime(s_clean.index)
        aligned = s_clean.reindex(idx, method="ffill", limit=65)
        frames[name] = aligned

    if not frames:
        return pd.DataFrame(index=idx)
    return pd.DataFrame(frames, index=idx)
