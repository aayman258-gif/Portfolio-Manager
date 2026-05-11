"""
Alpaca Market Data Client
Replaces yfinance for stock price data.
Requires: pip install alpaca-py

API keys via environment variables:
    ALPACA_API_KEY   — your Alpaca API key ID
    ALPACA_SECRET_KEY — your Alpaca secret key

Or via Streamlit secrets (secrets.toml):
    [alpaca]
    api_key    = "..."
    secret_key = "..."

Note: VIX (^VIX), options chains, earnings calendars, and
fundamental data are not available through Alpaca — those
calls are kept on yfinance in the relevant modules.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Union
import pandas as pd

from data.cache import cache_get, cache_set, TTL_PRICES

# ── Lazy singleton client ──────────────────────────────────────────────────────
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    try:
        from alpaca.data import StockHistoricalDataClient
    except ImportError as exc:
        raise ImportError(
            "alpaca-py is not installed. Run: pip install alpaca-py"
        ) from exc

    api_key    = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")

    # Try Streamlit secrets if keys not in env
    if not api_key or not secret_key:
        try:
            import streamlit as st
            _sec = st.secrets.get("alpaca", {})
            api_key    = api_key    or _sec.get("api_key", "")
            secret_key = secret_key or _sec.get("secret_key", "")
        except Exception:
            pass

    # None → free/IEX tier (delayed data, no auth required)
    _client = StockHistoricalDataClient(
        api_key    or None,
        secret_key or None,
    )
    return _client


# ── Period helpers ─────────────────────────────────────────────────────────────
_PERIOD_MAP: dict[str, timedelta] = {
    "1d":  timedelta(days=1),
    "5d":  timedelta(days=5),
    "1mo": timedelta(days=31),
    "3mo": timedelta(days=92),
    "6mo": timedelta(days=183),
    "1y":  timedelta(days=365),
    "2y":  timedelta(days=730),
    "3y":  timedelta(days=1095),
    "5y":  timedelta(days=1825),
    "10y": timedelta(days=3650),
}


def _period_to_start(period: str) -> datetime:
    delta = _PERIOD_MAP.get(period, timedelta(days=365))
    return datetime.now(tz=timezone.utc) - delta


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ── Core bar-fetching helpers ──────────────────────────────────────────────────

def get_bars(
    ticker: str,
    period: str = "1y",
    start: datetime | None = None,
    end:   datetime | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars for a single ticker.

    Returns a DataFrame with a DatetimeIndex (UTC-normalized dates)
    and columns: Open, High, Low, Close, Volume

    Compatible drop-in for:
        yf.download(ticker, period=period)
        yf.Ticker(ticker).history(period=period)
        yf.Ticker(ticker).history(start=start, end=end)
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    _start = _to_utc(start) if start else _period_to_start(period)
    _end   = _to_utc(end)   if end   else datetime.now(tz=timezone.utc)

    # Check cache (skip for intraday/very-recent requests)
    _cache_key = f"bars:{ticker.upper()}:{period}:{start}:{end}"
    if use_cache:
        cached = cache_get(_cache_key, TTL_PRICES)
        if cached is not None:
            return cached

    req = StockBarsRequest(
        symbol_or_symbols=ticker.upper(),
        timeframe=TimeFrame.Day,
        start=_start,
        end=_end,
    )

    try:
        raw = _get_client().get_stock_bars(req)
        df  = raw.df

        if df is None or df.empty:
            raise ValueError("Empty response from Alpaca")

        # Drop symbol level from MultiIndex → flat DatetimeIndex
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[ticker.upper()] if ticker.upper() in df.index.get_level_values(0) else df.droplevel(0)

        df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
        df.index.name = "Date"

        col_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns=col_map)

        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        result = df[keep].sort_index()
        if use_cache and not result.empty:
            cache_set(_cache_key, result)
        return result

    except Exception:
        # yfinance fallback (SIP subscription or other Alpaca error)
        try:
            import yfinance as yf
            kwargs: dict = {"progress": False, "auto_adjust": True}
            if start or end:
                if start:
                    kwargs["start"] = _start.date()
                if end:
                    kwargs["end"] = _end.date()
            else:
                kwargs["period"] = period
            yf_df = yf.download(ticker.upper(), **kwargs)
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            if yf_df.empty:
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            yf_df.index = pd.to_datetime(yf_df.index).normalize().tz_localize(None)
            yf_df.index.name = "Date"
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in yf_df.columns]
            result = yf_df[keep].sort_index()
            if use_cache and not result.empty:
                cache_set(_cache_key, result)
            return result
        except Exception:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def get_close_prices(
    tickers: list[str],
    period: str = "1y",
    start: datetime | None = None,
    end:   datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch daily Close prices for multiple tickers.

    Returns a DataFrame with a DatetimeIndex and tickers as columns.

    Compatible drop-in for:
        yf.download(tickers, period=period, auto_adjust=True)["Close"]
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    if not tickers:
        return pd.DataFrame()

    _start = _to_utc(start) if start else _period_to_start(period)
    _end   = _to_utc(end)   if end   else datetime.now(tz=timezone.utc)

    req = StockBarsRequest(
        symbol_or_symbols=[t.upper() for t in tickers],
        timeframe=TimeFrame.Day,
        start=_start,
        end=_end,
    )

    raw = _get_client().get_stock_bars(req)
    df  = raw.df

    if df is None or df.empty:
        return pd.DataFrame(columns=[t.upper() for t in tickers])

    # df has MultiIndex (symbol, timestamp) → pivot to wide format
    close = df["close"].unstack(level=0)
    close.index = pd.to_datetime(close.index).normalize().tz_localize(None)
    close.index.name = "Date"
    close.columns = [c.upper() for c in close.columns]
    close = close.sort_index()

    # Ensure all requested tickers are in the result (fill missing with NaN)
    for t in [t.upper() for t in tickers]:
        if t not in close.columns:
            close[t] = float("nan")

    return close[[t.upper() for t in tickers]]


def get_ohlcv(
    tickers: list[str],
    period: str = "1y",
    start: datetime | None = None,
    end:   datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV for multiple tickers.

    Returns a DataFrame with MultiIndex columns: (Field, Ticker)
    where Field ∈ {Open, High, Low, Close, Volume}.

    Compatible drop-in for yf.download(tickers, ...) when you
    need raw["Close"], raw["Open"], etc.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    if not tickers:
        return pd.DataFrame()

    _start = _to_utc(start) if start else _period_to_start(period)
    _end   = _to_utc(end)   if end   else datetime.now(tz=timezone.utc)

    req = StockBarsRequest(
        symbol_or_symbols=[t.upper() for t in tickers],
        timeframe=TimeFrame.Day,
        start=_start,
        end=_end,
    )

    raw = _get_client().get_stock_bars(req)
    df  = raw.df

    if df is None or df.empty:
        return pd.DataFrame()

    col_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    df = df.rename(columns=col_map)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]

    # Unstack symbol to MultiIndex columns: (Field, Ticker)
    result = df.unstack(level=0)
    result.index = pd.to_datetime(result.index).normalize().tz_localize(None)
    result.index.name = "Date"
    result.columns = pd.MultiIndex.from_tuples(
        [(field, sym.upper()) for field, sym in result.columns],
        names=["Field", "Ticker"],
    )
    return result.sort_index()


def get_latest_price(ticker: str) -> float | None:
    """
    Return the most recent close price for a single ticker.

    Compatible drop-in for:
        yf.Ticker(ticker).history(period='5d')['Close'].iloc[-1]
    """
    try:
        from alpaca.data.requests import StockLatestBarRequest
        req    = StockLatestBarRequest(symbol_or_symbols=ticker.upper())
        result = _get_client().get_stock_latest_bar(req)
        bar    = result.get(ticker.upper())
        return float(bar.close) if bar else None
    except Exception:
        return None


def get_latest_prices(tickers: list[str]) -> dict[str, float]:
    """
    Return a dict of {ticker: latest_close_price} for a list of tickers.

    Compatible drop-in for fetching current prices via:
        yf.Ticker(ticker).history(period='5d')['Close'].iloc[-1]
    """
    if not tickers:
        return {}
    try:
        from alpaca.data.requests import StockLatestBarRequest
        req    = StockLatestBarRequest(symbol_or_symbols=[t.upper() for t in tickers])
        result = _get_client().get_stock_latest_bar(req)
        return {sym: float(bar.close) for sym, bar in result.items() if bar}
    except Exception:
        return {}
