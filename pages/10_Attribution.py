"""
Regime-Aware Portfolio Manager
Section 17: Performance Attribution
Brinson-Hood-Beebower attribution: allocation, selection, interaction effects.
Benchmark: SPY (configurable). Compares portfolio segment weights & returns vs. benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import sys as _sys; _sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import data.alpaca_client as alpaca

sys.path.append(str(Path(__file__).parent.parent))

from utils.carbon_theme import (
    ACCENT, AMBER, BG, BORDER, CARD, DIM, FG, GAIN, LOSS, SUBTLE,
    apply_carbon_theme, carbon_plotly_layout, flex_table, page_header, top_nav,
)
from utils.portfolio_store import restore_portfolio_to_session

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Performance Attribution", page_icon="◈", layout="wide")
apply_carbon_theme()
top_nav("Attribution")

page_header(
    "Performance Attribution",
    "Brinson-Hood-Beebower attribution · allocation · selection · interaction",
)

# ── Auto-restore portfolio from disk if not in session ────────────────────────
restore_portfolio_to_session()

# ── Check portfolio ───────────────────────────────────────────────────────────
if "positions" not in st.session_state:
    st.warning("No portfolio loaded. Go to Home and load a portfolio first.")
    st.stop()

positions_df: pd.DataFrame = st.session_state["positions"]

# ── GICS sector mapping (static fallback) ────────────────────────────────────
_SECTOR_FALLBACK: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "GOOG": "Technology", "META": "Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "MS": "Financials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "JNJ": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
    "V": "Financials", "MA": "Financials",
    "HD": "Consumer Discretionary", "WMT": "Consumer Staples", "COST": "Consumer Staples",
    "SPY": "Blend", "QQQ": "Technology", "IWM": "Blend", "DIA": "Blend",
}

@st.cache_data(ttl=3600, show_spinner=False)
def _get_sector(ticker: str) -> str:
    return _SECTOR_FALLBACK.get(ticker, "Other")

# ── Two-column layout ─────────────────────────────────────────────────────────
_pa_lc, _pa_rc = st.columns([1, 3], gap="medium")

with _pa_lc:
    st.markdown(
        f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.12em;'
        f'color:{DIM};margin-bottom:0.4rem;">Controls</div>',
        unsafe_allow_html=True,
    )
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "DIA"], index=0)
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

# ── Data loading ──────────────────────────────────────────────────────────────
tickers = positions_df["ticker"].dropna().unique().tolist()
all_tickers = list(set(tickers + [benchmark]))

@st.cache_data(ttl=3600, show_spinner=False)
def _load_prices(tickers_key: tuple, period: str) -> pd.DataFrame:
    close = alpaca.get_close_prices(list(tickers_key), period=period)
    if close.empty:
        return pd.DataFrame()
    return close.dropna(how="all")

with st.spinner("Loading price data…"):
    prices = _load_prices(tuple(all_tickers), period)

if prices.empty:
    st.error("Could not load price data.")
    st.stop()

# ── Compute total returns ─────────────────────────────────────────────────────
def _total_return(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2:
        return 0.0
    return float((s.iloc[-1] / s.iloc[0]) - 1)

portfolio_total_return = 0.0
benchmark_total_return = 0.0

if benchmark in prices.columns:
    benchmark_total_return = _total_return(prices[benchmark])

# Current position values for weights
shares_map = dict(zip(positions_df["ticker"], positions_df["shares"].astype(float)))
cost_map   = dict(zip(positions_df["ticker"], positions_df["cost_basis"].astype(float)))

last_prices: dict[str, float] = {}
for t in tickers:
    if t in prices.columns:
        s = prices[t].dropna()
        last_prices[t] = float(s.iloc[-1]) if len(s) > 0 else cost_map.get(t, 1.0)
    else:
        last_prices[t] = cost_map.get(t, 1.0)

position_values = {t: shares_map.get(t, 0) * last_prices.get(t, 0) for t in tickers}
total_val = sum(position_values.values())
weights = {t: v / total_val for t, v in position_values.items()} if total_val > 0 else {}

# Position-level returns
pos_returns: dict[str, float] = {}
for t in tickers:
    if t in prices.columns:
        pos_returns[t] = _total_return(prices[t])
    else:
        pos_returns[t] = 0.0

portfolio_total_return = sum(weights.get(t, 0) * pos_returns.get(t, 0) for t in tickers)

# ── Sector grouping ───────────────────────────────────────────────────────────
with st.spinner("Fetching sector data…"):
    sectors = {t: _get_sector(t) for t in tickers}

sector_data: dict[str, dict] = {}
for t in tickers:
    sec = sectors.get(t, "Other")
    w   = weights.get(t, 0)
    r   = pos_returns.get(t, 0)
    if sec not in sector_data:
        sector_data[sec] = {"port_weight": 0.0, "port_ret_weighted": 0.0}
    sector_data[sec]["port_weight"]        += w
    sector_data[sec]["port_ret_weighted"]  += w * r

for sec in sector_data:
    pw = sector_data[sec]["port_weight"]
    sector_data[sec]["port_return"] = (
        sector_data[sec]["port_ret_weighted"] / pw if pw > 0 else 0.0
    )

# ── Benchmark sector weights (SPY approximate via GICS, simplified) ───────────
_SPY_SECTOR_WEIGHTS: dict[str, float] = {
    "Technology":             0.32,
    "Financials":             0.13,
    "Healthcare":             0.12,
    "Consumer Discretionary": 0.10,
    "Industrials":            0.09,
    "Consumer Staples":       0.06,
    "Energy":                 0.05,
    "Real Estate":            0.02,
    "Materials":              0.02,
    "Utilities":              0.02,
    "Communication Services": 0.07,
    "Other":                  0.0,
    "Blend":                  0.0,
}

_SPY_SECTOR_RETURNS: dict[str, float] = {}
# Use benchmark total return as approximate sector return (simplified)
for sec in _SPY_SECTOR_WEIGHTS:
    _SPY_SECTOR_RETURNS[sec] = benchmark_total_return  # simplified flat assumption

# ── BHB Attribution ───────────────────────────────────────────────────────────
all_sectors = sorted(set(list(sector_data.keys()) + list(_SPY_SECTOR_WEIGHTS.keys())))

attribution_rows = []
total_allocation  = 0.0
total_selection   = 0.0
total_interaction = 0.0

for sec in all_sectors:
    wp  = sector_data.get(sec, {}).get("port_weight", 0.0)
    wb  = _SPY_SECTOR_WEIGHTS.get(sec, 0.0)
    rp  = sector_data.get(sec, {}).get("port_return", 0.0)
    rb  = _SPY_SECTOR_RETURNS.get(sec, benchmark_total_return)
    rb_port = benchmark_total_return

    # BHB:
    allocation  = (wp - wb) * (rb - rb_port)
    selection   = wb * (rp - rb)
    interaction = (wp - wb) * (rp - rb)
    total       = allocation + selection + interaction

    total_allocation  += allocation
    total_selection   += selection
    total_interaction += interaction

    attribution_rows.append({
        "Sector":       sec,
        "Port Weight":  wp,
        "Bench Weight": wb,
        "Port Return":  rp,
        "Bench Return": rb,
        "Allocation":   allocation,
        "Selection":    selection,
        "Interaction":  interaction,
        "Total Effect": total,
    })

attr_df = pd.DataFrame(attribution_rows)
attr_df = attr_df[attr_df["Port Weight"] + attr_df["Bench Weight"] > 0].copy()

active_return = portfolio_total_return - benchmark_total_return

# ── Summary metrics → LEFT ───────────────────────────────────────────────────
summary_items = [
    ("Portfolio Return",   f"{portfolio_total_return*100:.2f}%", GAIN if portfolio_total_return >= 0 else LOSS),
    ("Benchmark Return",   f"{benchmark_total_return*100:.2f}%", GAIN if benchmark_total_return >= 0 else LOSS),
    ("Active Return",      f"{active_return*100:.2f}%",          GAIN if active_return >= 0 else LOSS),
    ("Allocation Effect",  f"{total_allocation*100:.2f}%",       GAIN if total_allocation >= 0 else LOSS),
    ("Selection Effect",   f"{total_selection*100:.2f}%",        GAIN if total_selection >= 0 else LOSS),
    ("Interaction Effect", f"{total_interaction*100:.2f}%",      AMBER),
]
with _pa_lc:
    st.markdown(
        f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.12em;'
        f'color:{DIM};margin:0.8rem 0 0.4rem;">Attribution Summary</div>',
        unsafe_allow_html=True,
    )
    for label, val, color in summary_items:
        st.markdown(
            f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:4px;'
            f'padding:0.4rem 0.65rem;margin-bottom:0.25rem;'
            f'display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;color:{DIM};">{label}</span>'
            f'<span style="font-size:0.9rem;font-weight:500;color:{color};">{val}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.caption(f"Period: {period} vs {benchmark}. BHB model.")

# ── Waterfall chart → RIGHT ───────────────────────────────────────────────────
with _pa_rc:
    st.subheader("Attribution Waterfall")

    waterfall_labels = ["Portfolio Return", "Benchmark Return", "Allocation", "Selection", "Interaction", "Active Return"]
    waterfall_values = [
        portfolio_total_return * 100,
        -benchmark_total_return * 100,
        total_allocation * 100,
        total_selection * 100,
        total_interaction * 100,
        0,
    ]
    waterfall_colors = [
        GAIN if portfolio_total_return >= 0 else LOSS,
        LOSS if benchmark_total_return >= 0 else GAIN,
        GAIN if total_allocation >= 0 else LOSS,
        GAIN if total_selection >= 0 else LOSS,
        AMBER,
        ACCENT,
    ]

    fig_wf = go.Figure(go.Bar(
        x=waterfall_labels,
        y=waterfall_values,
        marker_color=waterfall_colors,
        text=[f"{v:+.2f}%" for v in waterfall_values],
        textposition="outside",
    ))
    fig_wf.update_layout(**carbon_plotly_layout(
        title=f"Attribution Effects vs {benchmark} ({period})",
        yaxis_title="Return (%)",
        height=350,
    ))
    st.plotly_chart(fig_wf, use_container_width=True)

# ── Sector-level breakdown ────────────────────────────────────────────────────
with _pa_rc:
    st.subheader("Sector Attribution Breakdown")
    _l, _r = st.columns(2)
    with _l:
        fig_weights = go.Figure()
        sectors_sorted = attr_df.sort_values("Port Weight", ascending=True)
        fig_weights.add_trace(go.Bar(
            y=sectors_sorted["Sector"],
            x=sectors_sorted["Port Weight"] * 100,
            name="Portfolio",
            orientation="h",
            marker_color=ACCENT,
            opacity=0.85,
        ))
        fig_weights.add_trace(go.Bar(
            y=sectors_sorted["Sector"],
            x=sectors_sorted["Bench Weight"] * 100,
            name=benchmark,
            orientation="h",
            marker_color=SUBTLE,
            opacity=0.5,
        ))
        fig_weights.update_layout(**carbon_plotly_layout(
            barmode="overlay",
            title="Sector Weights: Portfolio vs Benchmark (%)",
            xaxis_title="Weight (%)",
            height=max(300, len(attr_df) * 30 + 80),
        ))
        st.plotly_chart(fig_weights, use_container_width=True)
    with _r:
        fig_effects = go.Figure()
        attr_sorted = attr_df.sort_values("Total Effect", ascending=True)
        fig_effects.add_trace(go.Bar(
            y=attr_sorted["Sector"],
            x=attr_sorted["Allocation"] * 100,
            name="Allocation",
            orientation="h",
            marker_color=GAIN,
        ))
        fig_effects.add_trace(go.Bar(
            y=attr_sorted["Sector"],
            x=attr_sorted["Selection"] * 100,
            name="Selection",
            orientation="h",
            marker_color=AMBER,
        ))
        fig_effects.add_trace(go.Bar(
            y=attr_sorted["Sector"],
            x=attr_sorted["Interaction"] * 100,
            name="Interaction",
            orientation="h",
            marker_color=LOSS,
            opacity=0.6,
        ))
        fig_effects.update_layout(**carbon_plotly_layout(
            barmode="relative",
            title="Attribution by Sector (%)",
            xaxis_title="Effect (%)",
            height=max(300, len(attr_df) * 30 + 80),
        ))
        st.plotly_chart(fig_effects, use_container_width=True)

# ── Detail table ──────────────────────────────────────────────────────────────
with _pa_rc:
    st.subheader("Detailed Attribution Table")

    _pct2 = lambda x: f"{x*100:.2f}%"
    _pct2cs = {"fmt": _pct2, "numeric": True, "color_scale": "rg"}
    _pct2n  = {"fmt": _pct2, "numeric": True}
    flex_table(attr_df, columns=[
        {"key": "Sector",       "label": "Sector",           "width": "15%", "align": "left"},
        {"key": "Port Weight",  "label": "Port Wt.",         "width": "10%", "align": "right", **_pct2n},
        {"key": "Bench Weight", "label": f"{benchmark} Wt.", "width": "11%", "align": "right", **_pct2n},
        {"key": "Port Return",  "label": "Port Ret.",        "width": "10%", "align": "right", **_pct2cs},
        {"key": "Bench Return", "label": f"{benchmark} Ret.","width": "11%", "align": "right", **_pct2cs},
        {"key": "Allocation",   "label": "Allocation",       "width": "11%", "align": "right", **_pct2cs},
        {"key": "Selection",    "label": "Selection",        "width": "11%", "align": "right", **_pct2cs},
        {"key": "Interaction",  "label": "Interaction",      "width": "11%", "align": "right", **_pct2cs},
        {"key": "Total Effect", "label": "Total Effect",     "width": "10%", "align": "right", **_pct2cs},
    ], key="attr_detail")

# ── Position-level contribution ───────────────────────────────────────────────
with _pa_rc:
    st.subheader("Position-Level Return Contribution")

    contrib_rows = []
    for t in tickers:
        w = weights.get(t, 0)
        r = pos_returns.get(t, 0)
        contrib_rows.append({
            "Ticker":       t,
            "Sector":       sectors.get(t, "Other"),
            "Weight":       w,
            "Return":       r,
            "Contribution": w * r,
        })

    contrib_df = pd.DataFrame(contrib_rows).sort_values("Contribution", ascending=False)

    fig_contrib = go.Figure(go.Bar(
        x=contrib_df["Ticker"],
        y=contrib_df["Contribution"] * 100,
        marker_color=[GAIN if v >= 0 else LOSS for v in contrib_df["Contribution"]],
        text=[f"{v*100:.2f}%" for v in contrib_df["Contribution"]],
        textposition="outside",
    ))
    fig_contrib.update_layout(**carbon_plotly_layout(
        title="Position Contribution to Portfolio Return (%)",
        yaxis_title="Contribution (%)",
        height=320,
    ))
    st.plotly_chart(fig_contrib, use_container_width=True)
