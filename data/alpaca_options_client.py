"""
Alpaca Options Market Data Client
Replaces yfinance for options chains, snapshots, and greeks.

Requires:
  - pip install alpaca-py
  - ALPACA_API_KEY / ALPACA_SECRET_KEY  (or [alpaca] block in .streamlit/secrets.toml)
  - Options Data subscription on your Alpaca account

OCC symbol format: AAPL240119C00190000
  Underlying  — variable length uppercase letters
  Expiry      — YYMMDD
  Type        — C (call) or P (put)
  Strike      — 8-digit integer = strike * 1000 (right-padded with zeros)

Falls back to yfinance if keys are missing or subscription is inactive.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional

import pandas as pd

# ── Credential resolution (shared with alpaca_client.py) ──────────────────────

def _get_credentials() -> tuple[str | None, str | None]:
    api_key    = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        try:
            import streamlit as st
            _sec   = st.secrets.get("alpaca", {})
            api_key    = api_key    or _sec.get("api_key", "")
            secret_key = secret_key or _sec.get("secret_key", "")
        except Exception:
            pass
    return api_key or None, secret_key or None


# ── Lazy singleton client ──────────────────────────────────────────────────────

_opt_client = None


def _get_client():
    global _opt_client
    if _opt_client is not None:
        return _opt_client

    try:
        from alpaca.data.historical import OptionHistoricalDataClient
    except ImportError as exc:
        raise ImportError("alpaca-py is not installed. Run: pip install alpaca-py") from exc

    api_key, secret_key = _get_credentials()
    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API keys required for options data. "
            "Set ALPACA_API_KEY / ALPACA_SECRET_KEY or add [alpaca] to secrets.toml."
        )

    _opt_client = OptionHistoricalDataClient(api_key, secret_key)
    return _opt_client


# ── OCC symbol helpers ─────────────────────────────────────────────────────────

_OCC_RE = re.compile(r'^([A-Z]+)(\d{6})([CP])(\d{8})$')


def parse_occ_symbol(symbol: str) -> dict | None:
    """
    Parse an OCC option symbol into its components.

    Returns dict with keys: underlying, expiration (YYYY-MM-DD),
    option_type ('call'/'put'), strike (float).
    Returns None if symbol cannot be parsed.
    """
    m = _OCC_RE.match(symbol.replace(" ", "").upper())
    if not m:
        return None
    underlying  = m.group(1)
    date_str    = m.group(2)          # YYMMDD
    option_type = "call" if m.group(3) == "C" else "put"
    strike      = int(m.group(4)) / 1000
    expiration  = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:]}"
    return {
        "symbol":      symbol,
        "underlying":  underlying,
        "expiration":  expiration,
        "option_type": option_type,
        "strike":      strike,
    }


def build_occ_symbol(underlying: str, expiration: str, option_type: str, strike: float) -> str:
    """
    Build an OCC symbol from components.

    expiration : YYYY-MM-DD
    option_type: 'call' or 'put'
    strike     : float (e.g. 190.0)
    """
    dt      = datetime.strptime(expiration, "%Y-%m-%d")
    date_s  = dt.strftime("%y%m%d")
    cp      = "C" if option_type.lower() == "call" else "P"
    strike_i = int(round(strike * 1000))
    return f"{underlying.upper()}{date_s}{cp}{strike_i:08d}"


# ── Core data functions ────────────────────────────────────────────────────────

def get_option_chain(
    underlying: str,
    expiration: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Fetch calls and puts for an underlying (and optional expiration).

    Returns (calls_df, puts_df, underlying_price).
    DataFrames match the yfinance option_chain() column schema, plus
    native greeks columns: delta, gamma, theta, vega.

    Falls back to yfinance on any Alpaca error.
    """
    try:
        from alpaca.data.requests import OptionChainRequest

        client = _get_client()
        req_kwargs: dict = {"underlying_symbol": underlying.upper()}
        if expiration:
            from datetime import date as _date
            req_kwargs["expiration_date"] = datetime.strptime(expiration, "%Y-%m-%d").date()

        raw = client.get_option_chain(OptionChainRequest(**req_kwargs))
        if not raw:
            raise ValueError("Empty chain returned")

        rows = []
        for symbol, snap in raw.items():
            parsed = parse_occ_symbol(symbol)
            if not parsed:
                continue

            quote = snap.latest_quote
            trade = snap.latest_trade
            greeks = snap.greeks

            bid   = float(quote.bid_price)  if quote and quote.bid_price  else 0.0
            ask   = float(quote.ask_price)  if quote and quote.ask_price  else 0.0
            last  = float(trade.price)      if trade and trade.price      else (bid + ask) / 2
            iv    = float(snap.implied_volatility) if snap.implied_volatility else 0.0
            vol   = int(quote.bid_size + quote.ask_size) if quote else 0

            rows.append({
                "contractSymbol":    symbol,
                "strike":            parsed["strike"],
                "lastPrice":         last,
                "bid":               bid,
                "ask":               ask,
                "midPrice":          (bid + ask) / 2,
                "volume":            vol,
                "openInterest":      0,
                "impliedVolatility": iv,
                "expiration":        parsed["expiration"],
                "option_type":       parsed["option_type"],
                # greeks (None if snap.greeks is None)
                "delta": float(greeks.delta) if greeks else None,
                "gamma": float(greeks.gamma) if greeks else None,
                "theta": float(greeks.theta) if greeks else None,
                "vega":  float(greeks.vega)  if greeks else None,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No rows after parsing")

        # Current underlying price via stock client
        from data.alpaca_client import get_latest_price
        underlying_price = get_latest_price(underlying) or 0.0

        calls = _enrich(df[df["option_type"] == "call"].copy(), underlying_price, "call")
        puts  = _enrich(df[df["option_type"] == "put"].copy(),  underlying_price, "put")
        return calls.reset_index(drop=True), puts.reset_index(drop=True), underlying_price

    except Exception as exc:
        print(f"[alpaca_options] chain error ({underlying}): {exc} — falling back to yfinance")
        return _yf_fallback_chain(underlying, expiration)


def get_option_expirations(underlying: str) -> list[str]:
    """
    Return sorted list of available expiration dates (YYYY-MM-DD).
    Falls back to yfinance on error.
    """
    try:
        from alpaca.data.requests import OptionChainRequest

        client = _get_client()
        raw    = client.get_option_chain(OptionChainRequest(underlying_symbol=underlying.upper()))
        if not raw:
            raise ValueError("Empty response")

        expirations: set[str] = set()
        for symbol in raw:
            parsed = parse_occ_symbol(symbol)
            if parsed:
                expirations.add(parsed["expiration"])

        return sorted(expirations)

    except Exception as exc:
        print(f"[alpaca_options] expirations error ({underlying}): {exc} — falling back to yfinance")
        try:
            import yfinance as yf
            return list(yf.Ticker(underlying).options)
        except Exception:
            return []


def get_option_snapshot(
    underlying: str,
    strike: float,
    expiration: str,
    option_type: str,
) -> dict:
    """
    Get a real-time snapshot for a single contract (quote + greeks).
    Returns dict with bid, ask, last, iv, delta, gamma, theta, vega.
    """
    symbol = build_occ_symbol(underlying, expiration, option_type, strike)
    try:
        from alpaca.data.requests import OptionSnapshotRequest

        client = _get_client()
        raw    = client.get_option_snapshot(OptionSnapshotRequest(symbol_or_symbols=symbol))
        snap   = raw.get(symbol)
        if not snap:
            raise ValueError(f"No snapshot for {symbol}")

        quote  = snap.latest_quote
        trade  = snap.latest_trade
        greeks = snap.greeks

        bid  = float(quote.bid_price) if quote and quote.bid_price else 0.0
        ask  = float(quote.ask_price) if quote and quote.ask_price else 0.0
        last = float(trade.price)     if trade and trade.price     else (bid + ask) / 2
        iv   = float(snap.implied_volatility) if snap.implied_volatility else 0.0

        return {
            "symbol":      symbol,
            "underlying":  underlying,
            "expiration":  expiration,
            "option_type": option_type,
            "strike":      strike,
            "bid":         bid,
            "ask":         ask,
            "mid":         (bid + ask) / 2,
            "last":        last,
            "iv":          iv,
            "delta":       float(greeks.delta) if greeks else None,
            "gamma":       float(greeks.gamma) if greeks else None,
            "theta":       float(greeks.theta) if greeks else None,
            "vega":        float(greeks.vega)  if greeks else None,
            "open_interest": 0,
        }

    except Exception as exc:
        print(f"[alpaca_options] snapshot error ({symbol}): {exc}")
        return {}


# ── Enrichment (moneyness, spread, time value) ────────────────────────────────

def _enrich(df: pd.DataFrame, underlying_price: float, option_type: str) -> pd.DataFrame:
    if df.empty or underlying_price == 0:
        return df

    if option_type == "call":
        df["moneyness"]     = underlying_price - df["strike"]
        df["moneyness_pct"] = (underlying_price / df["strike"] - 1) * 100
    else:
        df["moneyness"]     = df["strike"] - underlying_price
        df["moneyness_pct"] = (df["strike"] / underlying_price - 1) * 100

    df["ITM"]            = df["moneyness"] > 0
    df["intrinsicValue"] = df["moneyness"].clip(lower=0)
    df["timeValue"]      = (df["lastPrice"] - df["intrinsicValue"]).clip(lower=0)
    df["bidAskSpread"]   = (df["ask"] - df["bid"]).clip(lower=0)
    df["bidAskSpreadPct"] = (
        (df["bidAskSpread"] / df["lastPrice"].replace(0, float("nan"))) * 100
    ).fillna(0)

    return df.sort_values("strike").reset_index(drop=True)


# ── yfinance fallback ──────────────────────────────────────────────────────────

def _yf_fallback_chain(
    underlying: str,
    expiration: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Fetch option chain from yfinance when Alpaca is unavailable."""
    try:
        import yfinance as yf

        stock = yf.Ticker(underlying)
        hist  = stock.history(period="1d")
        underlying_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0

        expirations = list(stock.options)
        if not expirations:
            return pd.DataFrame(), pd.DataFrame(), underlying_price

        exp = expiration if expiration in expirations else expirations[0]
        chain = stock.option_chain(exp)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()

        # Normalise column names to match Alpaca schema
        for df in (calls, puts):
            df["midPrice"] = (df["bid"] + df["ask"]) / 2
            df["expiration"] = exp
            for col in ("delta", "gamma", "theta", "vega"):
                df[col] = None       # yfinance doesn't return greeks

        calls = _enrich(calls, underlying_price, "call")
        puts  = _enrich(puts,  underlying_price, "put")
        return calls, puts, underlying_price

    except Exception as exc:
        print(f"[yfinance fallback] chain error ({underlying}): {exc}")
        return pd.DataFrame(), pd.DataFrame(), 0.0
