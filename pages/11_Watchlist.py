"""
Regime-Aware Portfolio Manager
Section 16: Watchlist & Alerts Manager
Track tickers with price, volume, and volatility alerts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
sys.path.append(str(Path(__file__).parent.parent))
import data.alpaca_client as alpaca

from utils.carbon_theme import (
    ACCENT, AMBER, BG, BORDER, CARD, DIM, FG, GAIN, LOSS, SUBTLE,
    apply_carbon_theme, carbon_plotly_layout, page_header, top_nav,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Watchlist & Alerts", page_icon="◈", layout="wide")
apply_carbon_theme()
top_nav("Watchlist")

page_header("Watchlist & Alerts", "Monitor tickers with custom price, volume, and volatility triggers")

# ── Persistence ───────────────────────────────────────────────────────────────
_WL_PATH = Path.home() / ".portfolio_manager" / "watchlist.json"

def _load_watchlist() -> list[dict]:
    try:
        if _WL_PATH.exists():
            with open(_WL_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _save_watchlist(wl: list[dict]) -> None:
    _WL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_WL_PATH, "w") as f:
        json.dump(wl, f, indent=2, default=str)

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = _load_watchlist()

watchlist: list[dict] = st.session_state["watchlist"]

# ── Add ticker form ───────────────────────────────────────────────────────────
st.subheader("Add to Watchlist")
with st.form("add_wl_form", clear_on_submit=True):
    _a1, _a2, _a3, _a4, _a5, _a6 = st.columns([1, 1, 1, 1, 1, 1])
    with _a1:
        new_ticker = st.text_input("Ticker", placeholder="AAPL").upper().strip()
    with _a2:
        notes = st.text_input("Notes / Thesis", placeholder="Optional")
    with _a3:
        alert_price_above = st.number_input("Alert if price above ($)", min_value=0.0, value=0.0, step=1.0)
    with _a4:
        alert_price_below = st.number_input("Alert if price below ($)", min_value=0.0, value=0.0, step=1.0)
    with _a5:
        alert_vol_spike = st.number_input("Alert if Vol/OI ratio above", min_value=0.0, value=0.0, step=0.5,
                                          help="0 = disabled")
    with _a6:
        alert_pct_move = st.number_input("Alert if day move > (%)", min_value=0.0, value=0.0, step=0.5,
                                         help="0 = disabled")
    submitted = st.form_submit_button("Add to Watchlist", type="primary")

if submitted and new_ticker:
    existing = [w["ticker"] for w in watchlist]
    if new_ticker in existing:
        st.warning(f"{new_ticker} is already on your watchlist.")
    else:
        watchlist.append({
            "ticker": new_ticker,
            "notes": notes,
            "added_at": datetime.now().isoformat(),
            "alert_price_above": alert_price_above if alert_price_above > 0 else None,
            "alert_price_below": alert_price_below if alert_price_below > 0 else None,
            "alert_vol_spike": alert_vol_spike if alert_vol_spike > 0 else None,
            "alert_pct_move": alert_pct_move if alert_pct_move > 0 else None,
        })
        _save_watchlist(watchlist)
        st.success(f"Added {new_ticker} to watchlist.")
        st.rerun()

# ── Load live data for watchlist ──────────────────────────────────────────────
if not watchlist:
    st.info("Your watchlist is empty. Add tickers above.")
    st.stop()

wl_tickers = [w["ticker"] for w in watchlist]

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_watchlist_data(tickers: tuple) -> pd.DataFrame:
    rows = []
    try:
        close = alpaca.get_close_prices(list(tickers), period="10d")
        for t in tickers:
            try:
                s = close[t.upper()].dropna() if t.upper() in close.columns else pd.Series(dtype=float)
                cur  = float(s.iloc[-1]) if len(s) >= 1 else None
                prev = float(s.iloc[-2]) if len(s) >= 2 else cur
                chg  = ((cur - prev) / prev * 100) if (cur and prev and prev != 0) else None
                if len(s) >= 21:
                    log_rets = np.log(s / s.shift(1)).dropna()
                    vol_20 = float(log_rets.iloc[-20:].std() * np.sqrt(252) * 100)
                else:
                    vol_20 = None
                rows.append({"ticker": t, "price": cur, "prev_close": prev,
                             "day_chg_pct": chg, "vol_20d_ann": vol_20})
            except Exception:
                rows.append({"ticker": t, "price": None, "prev_close": None,
                             "day_chg_pct": None, "vol_20d_ann": None})
    except Exception:
        for t in tickers:
            rows.append({"ticker": t, "price": None, "prev_close": None,
                         "day_chg_pct": None, "vol_20d_ann": None})
    return pd.DataFrame(rows)

with st.spinner("Refreshing watchlist prices…"):
    live = _fetch_watchlist_data(tuple(wl_tickers))

# Merge live data with watchlist config
wl_df = pd.DataFrame(watchlist)
if not live.empty:
    wl_df = wl_df.merge(live, on="ticker", how="left")
else:
    for col in ["price", "prev_close", "day_chg_pct", "vol_20d_ann"]:
        wl_df[col] = None

# ── Triggered alerts ──────────────────────────────────────────────────────────
triggered: list[str] = []
for _, row in wl_df.iterrows():
    t = row["ticker"]
    p = row.get("price")
    chg = row.get("day_chg_pct")
    if p is not None:
        if row.get("alert_price_above") and p > row["alert_price_above"]:
            triggered.append(f"⬆ {t} crossed above ${row['alert_price_above']:.2f} (current ${p:.2f})")
        if row.get("alert_price_below") and p < row["alert_price_below"]:
            triggered.append(f"⬇ {t} crossed below ${row['alert_price_below']:.2f} (current ${p:.2f})")
    if chg is not None and row.get("alert_pct_move") and abs(chg) >= row["alert_pct_move"]:
        triggered.append(f"{t} moved {chg:+.2f}% today (threshold ±{row['alert_pct_move']:.1f}%)")

if triggered:
    for msg in triggered:
        st.markdown(
            f'<div style="background:rgba(245,158,11,0.10);border-left:3px solid {AMBER};'
            f'padding:0.5rem 1rem;margin-bottom:0.3rem;font-size:0.82rem;color:{AMBER};">'
            f'Alert: {msg}</div>',
            unsafe_allow_html=True,
        )

# ── Watchlist table ───────────────────────────────────────────────────────────
st.subheader("Watchlist")

header_cols = st.columns([1, 1, 1, 1, 1, 2, 1])
for col, hdr in zip(header_cols, ["Ticker", "Price", "Day Chg", "20d Ann. Vol", "Added", "Notes / Thesis", "Remove"]):
    col.markdown(
        f'<span style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;color:{DIM};">{hdr}</span>',
        unsafe_allow_html=True,
    )

to_remove = None
for idx, row in wl_df.iterrows():
    t    = row["ticker"]
    p    = row.get("price")
    chg  = row.get("day_chg_pct")
    vol  = row.get("vol_20d_ann")
    added = str(row.get("added_at", ""))[:10]
    notes_text = row.get("notes", "") or ""

    price_str = f"${p:.2f}" if p else "—"
    chg_str   = f"{chg:+.2f}%" if chg is not None else "—"
    vol_str   = f"{vol:.1f}%" if vol is not None else "—"
    chg_color = GAIN if (chg and chg >= 0) else LOSS

    c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1, 1, 2, 1])
    c1.markdown(f'<span style="color:{ACCENT};font-weight:500;">{t}</span>', unsafe_allow_html=True)
    c2.markdown(f'<span style="color:{FG};">{price_str}</span>', unsafe_allow_html=True)
    c3.markdown(f'<span style="color:{chg_color};">{chg_str}</span>', unsafe_allow_html=True)
    c4.markdown(f'<span style="color:{SUBTLE};">{vol_str}</span>', unsafe_allow_html=True)
    c5.markdown(f'<span style="color:{DIM};font-size:0.78rem;">{added}</span>', unsafe_allow_html=True)
    c6.markdown(f'<span style="color:{DIM};font-size:0.78rem;">{notes_text}</span>', unsafe_allow_html=True)
    if c7.button("✕", key=f"rm_{t}_{idx}"):
        to_remove = t

if to_remove:
    watchlist = [w for w in watchlist if w["ticker"] != to_remove]
    st.session_state["watchlist"] = watchlist
    _save_watchlist(watchlist)
    st.success(f"Removed {to_remove} from watchlist.")
    st.rerun()

# ── Mini price charts ─────────────────────────────────────────────────────────
st.subheader("7-Day Price Sparklines")

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_sparklines(tickers: tuple) -> dict:
    result = {}
    try:
        close = alpaca.get_close_prices(list(tickers), period="10d")
        for t in tickers:
            col = t.upper()
            if col in close.columns:
                result[t] = close[col].dropna().tail(7)
    except Exception:
        pass
    return result

spark_data = _fetch_sparklines(tuple(wl_tickers))

cols_per_row = 4
for i in range(0, len(wl_tickers), cols_per_row):
    row_tickers = wl_tickers[i:i + cols_per_row]
    spark_cols  = st.columns(cols_per_row)
    for col, t in zip(spark_cols, row_tickers):
        with col:
            s = spark_data.get(t)
            if s is not None and len(s) > 1:
                color = GAIN if s.iloc[-1] >= s.iloc[0] else LOSS
                fig_s = go.Figure(go.Scatter(
                    x=s.index, y=s.values,
                    line=dict(color=color, width=1.5),
                    fill="tozeroy",
                    fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.12)",
                ))
                fig_s.update_layout(**carbon_plotly_layout(height=130, margin=dict(l=8, r=8, t=24, b=8)))
                fig_s.update_xaxes(showticklabels=False)
                fig_s.update_layout(title=dict(text=t, font=dict(size=11, color=ACCENT), x=0.5))
                st.plotly_chart(fig_s, use_container_width=True)
            else:
                col.markdown(f'<div style="text-align:center;color:{DIM};padding:2rem;">{t}<br>No data</div>',
                             unsafe_allow_html=True)
