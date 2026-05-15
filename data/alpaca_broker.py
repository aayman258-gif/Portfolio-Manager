"""
Alpaca Paper Trading Broker Client
Uses the Alpaca Trading API (separate from the market-data API).

Credential resolution (uses SAME keys as market data — Alpaca paper
trading is enabled via the base_url, not separate credentials):
  1. env vars  ALPACA_API_KEY / ALPACA_SECRET_KEY
  2. Streamlit secrets  [alpaca] api_key / secret_key

Paper trading endpoint: https://paper-api.alpaca.markets

All methods return plain dicts / DataFrames so the UI layer stays
decoupled from the SDK objects.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd

_PAPER_URL = "https://paper-api.alpaca.markets"

# ── Credential resolution ─────────────────────────────────────────────────────

def _get_credentials() -> tuple[str | None, str | None]:
    api_key    = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        try:
            import streamlit as st
            _sec       = st.secrets.get("alpaca", {})
            api_key    = api_key    or _sec.get("api_key", "")
            secret_key = secret_key or _sec.get("secret_key", "")
        except Exception:
            pass
    return api_key or None, secret_key or None


# ── Lazy singleton trading client ─────────────────────────────────────────────

_trading_client = None


def _get_trading_client():
    global _trading_client
    if _trading_client is not None:
        return _trading_client

    try:
        from alpaca.trading.client import TradingClient
    except ImportError as exc:
        raise ImportError("alpaca-py is not installed. Run: pip install alpaca-py") from exc

    api_key, secret_key = _get_credentials()
    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API keys required. "
            "Set ALPACA_API_KEY / ALPACA_SECRET_KEY or add [alpaca] to secrets.toml."
        )
    _trading_client = TradingClient(api_key, secret_key, paper=True)
    return _trading_client


# ── Account ────────────────────────────────────────────────────────────────────

def get_account() -> dict:
    """
    Return paper account details as a plain dict.
    Keys: equity, buying_power, cash, portfolio_value, currency, status.
    """
    client = _get_trading_client()
    acct   = client.get_account()
    return {
        "equity":          float(acct.equity          or 0),
        "buying_power":    float(acct.buying_power     or 0),
        "cash":            float(acct.cash             or 0),
        "portfolio_value": float(acct.portfolio_value  or 0),
        "currency":        str(acct.currency           or "USD"),
        "status":          str(acct.status             or ""),
        "daytrade_count":  int(acct.daytrade_count     or 0),
    }


# ── Positions ─────────────────────────────────────────────────────────────────

def get_positions() -> pd.DataFrame:
    """
    Return all open paper positions as a DataFrame.
    Columns: symbol, qty, avg_entry_price, current_price, market_value,
             unrealized_pl, unrealized_plpc, side.
    """
    client = _get_trading_client()
    raw    = client.get_all_positions()
    if not raw:
        return pd.DataFrame()

    rows = []
    for p in raw:
        rows.append({
            "symbol":           str(p.symbol),
            "qty":              float(p.qty or 0),
            "avg_entry_price":  float(p.avg_entry_price  or 0),
            "current_price":    float(p.current_price     or 0),
            "market_value":     float(p.market_value      or 0),
            "unrealized_pl":    float(p.unrealized_pl     or 0),
            "unrealized_plpc":  float(p.unrealized_plpc   or 0),
            "side":             str(p.side or "long"),
        })
    return pd.DataFrame(rows)


# ── Orders ────────────────────────────────────────────────────────────────────

def get_orders(status: str = "all", limit: int = 50) -> pd.DataFrame:
    """
    Return recent orders as a DataFrame.
    status: 'open' | 'closed' | 'all'
    """
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums    import QueryOrderStatus

    _status_map = {
        "open":   QueryOrderStatus.OPEN,
        "closed": QueryOrderStatus.CLOSED,
        "all":    QueryOrderStatus.ALL,
    }
    client = _get_trading_client()
    req    = GetOrdersRequest(
        status=_status_map.get(status, QueryOrderStatus.ALL),
        limit=limit,
    )
    raw = client.get_orders(req)
    if not raw:
        return pd.DataFrame()

    rows = []
    for o in raw:
        rows.append({
            "id":            str(o.id),
            "created_at":    str(o.created_at)[:19] if o.created_at else "",
            "symbol":        str(o.symbol),
            "side":          str(o.side).replace("OrderSide.", ""),
            "type":          str(o.type).replace("OrderType.", ""),
            "qty":           float(o.qty  or 0),
            "filled_qty":    float(o.filled_qty or 0),
            "limit_price":   float(o.limit_price  or 0) if o.limit_price  else None,
            "filled_avg_price": float(o.filled_avg_price or 0) if o.filled_avg_price else None,
            "status":        str(o.status).replace("OrderStatus.", ""),
            "time_in_force": str(o.time_in_force).replace("TimeInForce.", ""),
        })
    return pd.DataFrame(rows)


# ── Place orders ──────────────────────────────────────────────────────────────

def place_market_order(
    symbol: str,
    qty: float,
    side: str,                   # 'buy' | 'sell'
    time_in_force: str = "day",  # 'day' | 'gtc' | 'ioc'
) -> dict:
    """
    Submit a market order.  Returns order dict on success.
    Raises on API errors.
    """
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums    import OrderSide, TimeInForce

    _side_map = {"buy": OrderSide.BUY, "sell": OrderSide.SELL}
    _tif_map  = {"day": TimeInForce.DAY, "gtc": TimeInForce.GTC, "ioc": TimeInForce.IOC}

    client = _get_trading_client()
    req = MarketOrderRequest(
        symbol=symbol.upper(),
        qty=qty,
        side=_side_map[side.lower()],
        time_in_force=_tif_map.get(time_in_force.lower(), TimeInForce.DAY),
    )
    o = client.submit_order(req)
    return _order_to_dict(o)


def place_limit_order(
    symbol: str,
    qty: float,
    side: str,
    limit_price: float,
    time_in_force: str = "day",
) -> dict:
    """Submit a limit order."""
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums    import OrderSide, TimeInForce

    _side_map = {"buy": OrderSide.BUY, "sell": OrderSide.SELL}
    _tif_map  = {"day": TimeInForce.DAY, "gtc": TimeInForce.GTC, "ioc": TimeInForce.IOC}

    client = _get_trading_client()
    req = LimitOrderRequest(
        symbol=symbol.upper(),
        qty=qty,
        side=_side_map[side.lower()],
        limit_price=round(limit_price, 2),
        time_in_force=_tif_map.get(time_in_force.lower(), TimeInForce.DAY),
    )
    o = client.submit_order(req)
    return _order_to_dict(o)


def cancel_order(order_id: str) -> bool:
    """Cancel an open order by id.  Returns True on success."""
    try:
        client = _get_trading_client()
        client.cancel_order_by_id(order_id)
        return True
    except Exception:
        return False


def cancel_all_orders() -> int:
    """Cancel all open orders.  Returns number cancelled."""
    try:
        client = _get_trading_client()
        cancelled = client.cancel_orders()
        return len(cancelled) if cancelled else 0
    except Exception:
        return 0


def close_position(symbol: str) -> dict:
    """Close (liquidate) an entire position."""
    client = _get_trading_client()
    o = client.close_position(symbol.upper())
    return _order_to_dict(o)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _order_to_dict(o) -> dict:
    return {
        "id":               str(o.id),
        "symbol":           str(o.symbol),
        "side":             str(o.side).replace("OrderSide.", ""),
        "type":             str(o.type).replace("OrderType.", ""),
        "qty":              float(o.qty or 0),
        "filled_qty":       float(o.filled_qty or 0),
        "limit_price":      float(o.limit_price or 0) if o.limit_price else None,
        "filled_avg_price": float(o.filled_avg_price or 0) if o.filled_avg_price else None,
        "status":           str(o.status).replace("OrderStatus.", ""),
        "created_at":       str(o.created_at)[:19] if o.created_at else "",
        "time_in_force":    str(o.time_in_force).replace("TimeInForce.", ""),
    }


def is_available() -> bool:
    """Return True if the broker client can be instantiated (keys present + alpaca-py installed)."""
    try:
        _get_trading_client()
        return True
    except Exception:
        return False
