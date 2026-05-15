"""
Regime-Aware Portfolio Manager
Section 12: Paper Trading Engine
Alpaca paper trading — stock orders, position management, order history.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from utils.carbon_theme import (
    ACCENT, AMBER, BG, BORDER, CARD, CARD2, DIM, FG, GAIN, LOSS, SUBTLE,
    apply_carbon_theme, carbon_plotly_layout, flex_table,
    page_header, section_header, top_nav, regime_color,
)
from utils.portfolio_store import restore_portfolio_to_session
from data.trade_journal import log_trade

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Paper Trading", page_icon="◈", layout="wide")
apply_carbon_theme()
top_nav("Paper Trading")
page_header("Paper Trading Engine", "Simulate trades via Alpaca paper account · All activity is virtual")
restore_portfolio_to_session()

# ── Broker availability check ─────────────────────────────────────────────────
try:
    import data.alpaca_broker as broker
    _broker_ok = broker.is_available()
except Exception as _broker_err:
    _broker_ok = False
    _broker_err_msg = str(_broker_err)

if not _broker_ok:
    st.error(
        "Alpaca broker client is unavailable. "
        "Ensure `alpaca-py` is installed and API keys are set in `.streamlit/secrets.toml` "
        "under `[alpaca]`."
    )
    with st.expander("Setup Instructions"):
        st.markdown("""
**1. Install alpaca-py**
```bash
pip install alpaca-py
```

**2. Add keys to `.streamlit/secrets.toml`**
```toml
[alpaca]
api_key    = "PKxxxxxxxxxxxxxxxx"
secret_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Your Alpaca account automatically has a **paper trading** sub-account.
No separate registration is needed — just use the same API keys.
        """)
    st.stop()

# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def _load_account() -> dict:
    return broker.get_account()

@st.cache_data(ttl=30, show_spinner=False)
def _load_positions() -> pd.DataFrame:
    return broker.get_positions()

@st.cache_data(ttl=30, show_spinner=False)
def _load_orders(status: str) -> pd.DataFrame:
    return broker.get_orders(status=status, limit=100)

# ── Refresh button ────────────────────────────────────────────────────────────
_col_r1, _col_r2 = st.columns([8, 1])
with _col_r2:
    if st.button("Refresh", type="secondary", key="_pt_refresh"):
        st.cache_data.clear()
        st.rerun()

# ── Account summary ───────────────────────────────────────────────────────────
section_header("Account Summary")
try:
    acct = _load_account()
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Portfolio Value",  f"${acct['portfolio_value']:,.2f}")
    a2.metric("Equity",           f"${acct['equity']:,.2f}")
    a3.metric("Cash",             f"${acct['cash']:,.2f}")
    a4.metric("Buying Power",     f"${acct['buying_power']:,.2f}")
    a5.metric("Status",           acct["status"].replace("_", " ").title())
except Exception as e:
    st.error(f"Could not load account: {e}")
    acct = {}

# ── Sub-tabs ──────────────────────────────────────────────────────────────────
pt_pos, pt_trade, pt_orders = st.tabs(["Positions", "Place Order", "Order History"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB: POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
with pt_pos:
    section_header("Open Positions")
    try:
        pos_df = _load_positions()
    except Exception as e:
        st.error(f"Could not load positions: {e}")
        pos_df = pd.DataFrame()

    if pos_df.empty:
        st.info("No open paper positions. Use the **Place Order** tab to simulate trades.")
    else:
        # Compute P&L percentage
        pos_df["pnl_pct"] = pos_df["unrealized_plpc"] * 100

        flex_table(
            pos_df[[
                "symbol", "qty", "side",
                "avg_entry_price", "current_price", "market_value",
                "unrealized_pl", "pnl_pct"
            ]],
            columns=[
                {"key": "symbol",          "label": "Symbol",     "width": "10%", "align": "left"},
                {"key": "qty",             "label": "Qty",        "width": "8%",  "align": "right",  "numeric": True},
                {"key": "side",            "label": "Side",       "width": "7%",  "align": "center"},
                {"key": "avg_entry_price", "label": "Avg Cost",   "width": "11%", "align": "right",  "numeric": True,
                 "fmt": lambda v: f"${v:.2f}" if pd.notna(v) else "—"},
                {"key": "current_price",   "label": "Price",      "width": "11%", "align": "right",  "numeric": True,
                 "fmt": lambda v: f"${v:.2f}" if pd.notna(v) else "—"},
                {"key": "market_value",    "label": "Mkt Value",  "width": "12%", "align": "right",  "numeric": True,
                 "fmt": lambda v: f"${v:,.2f}" if pd.notna(v) else "—"},
                {"key": "unrealized_pl",   "label": "Unreal. P&L","width": "13%", "align": "right",  "numeric": True,
                 "color_scale": "rg",
                 "fmt": lambda v: f"${v:+,.2f}" if pd.notna(v) else "—"},
                {"key": "pnl_pct",         "label": "P&L %",     "width": "10%", "align": "right",  "numeric": True,
                 "color_scale": "rg",
                 "fmt": lambda v: f"{v:+.2f}%" if pd.notna(v) else "—"},
            ],
            key="pt_pos_tbl",
        )

        # Close position buttons
        st.markdown("---")
        st.markdown(f'<span style="font-size:0.75rem;color:{DIM};text-transform:uppercase;letter-spacing:0.08em;">Close a Position</span>',
                    unsafe_allow_html=True)
        _symbols = pos_df["symbol"].tolist()
        _close_col1, _close_col2 = st.columns([2, 1])
        _close_sym = _close_col1.selectbox("Select symbol", _symbols, key="_pt_close_sym")
        if _close_col2.button("Close Position", type="secondary", key="_pt_close_btn"):
            try:
                result = broker.close_position(_close_sym)
                st.success(f"Close order submitted for {_close_sym} (order ID: {result['id'][:8]}…)")
                log_trade(
                    ticker=_close_sym,
                    action="SELL",
                    shares=result.get("qty", 0),
                    price=result.get("filled_avg_price") or 0,
                    notes="Paper: close position",
                    source="paper",
                )
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to close position: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: PLACE ORDER
# ══════════════════════════════════════════════════════════════════════════════
with pt_trade:
    section_header("Place a Paper Trade")

    # Regime context
    _regime_label = "Unknown"
    try:
        from data.market_data import MarketDataLoader as _ML
        from calculations.regime_detector import RegimeDetector as _RD
        _ml4 = _ML(); _det4 = _RD()
        _spy4 = _ml4.load_index_data("SPY", "2y")
        _vix4 = _ml4.load_vix_data("2y")
        _spx4, _vxp4 = _ml4.align_data(_spy4, _vix4)
        _reg4, _ = _det4.classify_regime(_spx4, _vxp4)
        _regime_label = str(_reg4.iloc[-1])
    except Exception:
        pass

    _rc = regime_color(_regime_label)
    st.markdown(
        f'<div style="background:{_rc}15;border:1px solid {_rc}44;border-radius:8px;'
        f'padding:10px 16px;margin-bottom:16px;display:inline-block;">'
        f'<span style="font-size:10px;color:{_rc};text-transform:uppercase;letter-spacing:0.1em;">Current Regime: </span>'
        f'<span style="font-size:16px;font-weight:600;color:{_rc};margin-left:6px;">{_regime_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.form("place_order_form"):
        _oc1, _oc2, _oc3, _oc4 = st.columns(4)
        _pt_sym   = _oc1.text_input("Ticker", placeholder="AAPL", key="_pt_sym").upper().strip()
        _pt_side  = _oc2.selectbox("Side", ["buy", "sell"], key="_pt_side",
                                    format_func=str.upper)
        _pt_qty   = _oc3.number_input("Shares", min_value=0.01, value=1.0, step=1.0, key="_pt_qty")
        _pt_type  = _oc4.selectbox("Order Type", ["market", "limit"], key="_pt_type")

        _oc5, _oc6, _oc7 = st.columns(3)
        _pt_limit = _oc5.number_input("Limit Price ($)", min_value=0.0, value=0.0, step=0.01,
                                       key="_pt_lp",
                                       help="Required for limit orders")
        _pt_tif   = _oc6.selectbox("Time in Force", ["day", "gtc", "ioc"], key="_pt_tif")
        _pt_notes = _oc7.text_input("Notes (journal)", placeholder="Optional", key="_pt_notes")

        _submit_order = st.form_submit_button("Submit Paper Order", type="primary")

    if _submit_order:
        if not _pt_sym:
            st.warning("Ticker is required.")
        elif _pt_qty <= 0:
            st.warning("Quantity must be > 0.")
        elif _pt_type == "limit" and _pt_limit <= 0:
            st.warning("Limit price required for limit orders.")
        else:
            try:
                with st.spinner(f"Submitting {_pt_side.upper()} {_pt_qty} {_pt_sym}…"):
                    if _pt_type == "market":
                        result = broker.place_market_order(_pt_sym, _pt_qty, _pt_side, _pt_tif)
                    else:
                        result = broker.place_limit_order(_pt_sym, _pt_qty, _pt_side, _pt_limit, _pt_tif)

                st.success(
                    f"Order submitted: {result['side'].upper()} {result['qty']} {result['symbol']} "
                    f"({result['type']}) · Status: {result['status']} · ID: {result['id'][:8]}…"
                )

                # Log to trade journal
                _exec_price = result.get("filled_avg_price") or _pt_limit or 0.0
                log_trade(
                    ticker=_pt_sym,
                    action=_pt_side.upper(),
                    shares=_pt_qty,
                    price=_exec_price,
                    regime=_regime_label,
                    notes=_pt_notes or f"Paper: {_pt_type} order",
                    source="paper",
                )

                st.cache_data.clear()
                st.rerun()

            except Exception as e:
                st.error(f"Order failed: {e}")

    # Quick-reference: current positions during order entry
    st.markdown("---")
    st.caption("Current positions (for reference)")
    try:
        _ref_pos = _load_positions()
        if not _ref_pos.empty:
            for _, r in _ref_pos.iterrows():
                color = GAIN if r["unrealized_pl"] >= 0 else LOSS
                st.markdown(
                    f'<span style="color:{ACCENT};font-weight:500;">{r["symbol"]}</span> '
                    f'<span style="color:{DIM};">· {r["qty"]:.4g} shares · </span>'
                    f'<span style="color:{color};">{r["unrealized_pl"]:+.2f} ({r["unrealized_plpc"]*100:+.2f}%)</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No open positions.")
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# TAB: ORDER HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with pt_orders:
    section_header("Order History")

    _oh_filter = st.radio("Filter", ["all", "open", "closed"], horizontal=True,
                          key="_pt_oh_filter")
    try:
        orders_df = _load_orders(_oh_filter)
    except Exception as e:
        st.error(f"Could not load orders: {e}")
        orders_df = pd.DataFrame()

    if orders_df.empty:
        st.info(f"No {_oh_filter} orders found.")
    else:
        flex_table(
            orders_df[[
                "created_at", "symbol", "side", "type",
                "qty", "filled_qty", "limit_price",
                "filled_avg_price", "status", "time_in_force"
            ]],
            columns=[
                {"key": "created_at",       "label": "Time",       "width": "14%", "align": "left"},
                {"key": "symbol",           "label": "Symbol",     "width": "8%",  "align": "left"},
                {"key": "side",             "label": "Side",       "width": "7%",  "align": "center"},
                {"key": "type",             "label": "Type",       "width": "8%",  "align": "center"},
                {"key": "qty",              "label": "Qty",        "width": "7%",  "align": "right",  "numeric": True},
                {"key": "filled_qty",       "label": "Filled",     "width": "7%",  "align": "right",  "numeric": True},
                {"key": "limit_price",      "label": "Limit",      "width": "9%",  "align": "right",  "numeric": True,
                 "fmt": lambda v: f"${v:.2f}" if pd.notna(v) and v else "—"},
                {"key": "filled_avg_price", "label": "Avg Fill",   "width": "9%",  "align": "right",  "numeric": True,
                 "fmt": lambda v: f"${v:.2f}" if pd.notna(v) and v else "—"},
                {"key": "status",           "label": "Status",     "width": "11%", "align": "center"},
                {"key": "time_in_force",    "label": "TIF",        "width": "7%",  "align": "center"},
            ],
            key="pt_orders_tbl",
        )

        # Cancel buttons
        _open_orders = orders_df[~orders_df["status"].isin(["filled", "canceled", "expired"])]
        if not _open_orders.empty:
            st.markdown("---")
            _cancel_col1, _cancel_col2, _cancel_col3 = st.columns([2, 2, 1])
            _cancel_id  = _cancel_col1.selectbox(
                "Cancel order",
                _open_orders["id"].tolist(),
                format_func=lambda x: f"{x[:8]}… ({_open_orders.set_index('id').loc[x,'symbol']})",
                key="_pt_cancel_sel",
            )
            if _cancel_col2.button("Cancel Selected", type="secondary", key="_pt_cancel_btn"):
                if broker.cancel_order(_cancel_id):
                    st.success("Order cancelled.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Failed to cancel order.")

            if _cancel_col3.button("Cancel ALL", type="secondary", key="_pt_cancel_all"):
                n = broker.cancel_all_orders()
                st.success(f"{n} orders cancelled.")
                st.cache_data.clear()
                st.rerun()
