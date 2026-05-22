"""
Regime-Aware Portfolio Manager
Section 13: Multi-Portfolio Comparison
Side-by-side comparison of up to 3 portfolios — value, P&L, allocation,
regime scores, equity curves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from utils.carbon_theme import (
    ACCENT, AMBER, BG, BORDER, CARD, CARD2, DIM, FG, GAIN, LOSS, SUBTLE, PURPLE,
    apply_carbon_theme, carbon_plotly_layout, flex_table,
    page_header, section_header, top_nav, regime_color,
)
from utils.portfolio_store import (
    list_portfolios, load_portfolio, get_active_portfolio,
    load_options_positions,
)
from data.trade_journal import get_portfolio_history, compute_equity_curve

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Compare", page_icon="◈", layout="wide")
apply_carbon_theme()
top_nav("Compare")
page_header("Portfolio Comparison", "Side-by-side analysis of your portfolios")

# ── Portfolio selector ────────────────────────────────────────────────────────
_all = list_portfolios()
_names = [p["name"] for p in _all]

if len(_names) < 2:
    st.info("You need at least 2 portfolios to compare. Create more from the Command Center.")
    st.stop()

_active = get_active_portfolio()
_default_sel = _names[:3]

section_header("Select Portfolios to Compare")
_sel_cols = st.columns(3)
_selected: list[str | None] = []
_palette = [ACCENT, AMBER, PURPLE]

for i, col in enumerate(_sel_cols):
    with col:
        _options = ["— None —"] + _names
        _def_idx = _names.index(_active) + 1 if i == 0 and _active in _names else 0
        _def_idx = min(i + 1, len(_options) - 1) if _def_idx == 0 else _def_idx
        _pick = st.selectbox(
            f"Portfolio {i + 1}",
            _options,
            index=_def_idx,
            key=f"_cmp_sel_{i}",
        )
        _selected.append(_pick if _pick != "— None —" else None)

_compare = [n for n in _selected if n is not None]
if len(_compare) < 2:
    st.info("Select at least 2 portfolios above.")
    st.stop()

# ── Load data for selected portfolios ────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _load_for_compare(names: tuple) -> dict[str, dict]:
    """Load positions + live prices for each portfolio."""
    import data.alpaca_client as alpaca
    from data.portfolio_loader import PortfolioLoader
    loader = PortfolioLoader()
    result: dict[str, dict] = {}

    for name in names:
        df = load_portfolio(name)
        if df is None or df.empty:
            result[name] = {"df": pd.DataFrame(), "prices": {}, "summary": {}, "meta": {}}
            continue
        tickers = df["ticker"].tolist()
        try:
            prices = alpaca.get_latest_prices(tickers)
        except Exception:
            prices = {}
        metrics = loader.calculate_position_metrics(df, prices)
        summary = loader.get_portfolio_summary(metrics)
        result[name] = {
            "df":      df,
            "prices":  prices,
            "metrics": metrics,
            "summary": summary,
        }
    return result

@st.cache_data(ttl=1800, show_spinner=False)
def _regime_scores_for(name: str, tickers: tuple) -> dict[str, float]:
    if not tickers:
        return {}
    try:
        from calculations.scoring_engine import ScoringEngine
        from calculations.regime_detector import RegimeDetector
        from data.market_data import MarketDataLoader
        ml = MarketDataLoader(); det = RegimeDetector(); sc = ScoringEngine()
        spy = ml.load_index_data("SPY", "1y")
        vix = ml.load_vix_data("1y")
        spx, vxp = ml.align_data(spy, vix)
        reg, sigs = det.classify_regime(spx, vxp)
        stock_data = {t: ml.load_single_stock(t, "1y") for t in list(tickers)[:6]}
        scores = sc.score_portfolio(stock_data, reg, sigs)
        return {t: round(float(v.get("composite", 50)), 1) for t, v in scores.items()}
    except Exception:
        return {}

with st.spinner("Loading portfolio data…"):
    _pdata = _load_for_compare(tuple(_compare))

# ── Summary metrics ───────────────────────────────────────────────────────────
section_header("Summary Comparison")

_metric_rows: dict[str, list] = {
    "Total Value ($)":    [],
    "All-time P&L ($)":   [],
    "All-time P&L (%)":   [],
    "# Positions":        [],
    "Winners":            [],
    "Losers":             [],
    "Win Rate":           [],
    "Largest Position":   [],
}

for name in _compare:
    d = _pdata.get(name, {})
    s = d.get("summary", {})
    if s:
        _metric_rows["Total Value ($)"].append(f"${s.get('total_value', 0):,.0f}")
        _metric_rows["All-time P&L ($)"].append(f"${s.get('total_pnl', 0):+,.2f}")
        _metric_rows["All-time P&L (%)"].append(f"{s.get('total_pnl_pct', 0):+.2f}%")
        _metric_rows["# Positions"].append(str(s.get("num_positions", 0)))
        _metric_rows["Winners"].append(str(s.get("winners", 0)))
        _metric_rows["Losers"].append(str(s.get("losers", 0)))
        _wr = s.get("winners", 0) / max(s.get("num_positions", 1), 1) * 100
        _metric_rows["Win Rate"].append(f"{_wr:.0f}%")
        df = d.get("df", pd.DataFrame())
        if not df.empty and "ticker" in df.columns:
            prices = d.get("prices", {})
            _vals = df.apply(lambda r: float(r["shares"]) * prices.get(r["ticker"], 0), axis=1)
            _top_t = df.loc[_vals.idxmax(), "ticker"] if not _vals.empty else "—"
            _metric_rows["Largest Position"].append(str(_top_t))
        else:
            _metric_rows["Largest Position"].append("—")
    else:
        for k in _metric_rows:
            _metric_rows[k].append("—")

# Display as styled comparison grid
_n = len(_compare)
_hdr_cols = st.columns([2] + [2] * _n)
_hdr_cols[0].markdown(f'<span style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:{DIM};">Metric</span>', unsafe_allow_html=True)
for i, name in enumerate(_compare):
    _color = _palette[i % len(_palette)]
    _hdr_cols[i + 1].markdown(
        f'<span style="font-size:0.80rem;font-weight:600;color:{_color};">{name}</span>',
        unsafe_allow_html=True,
    )

st.markdown(f'<div style="height:1px;background:{BORDER};margin:4px 0 8px;"></div>', unsafe_allow_html=True)

for metric, values in _metric_rows.items():
    _row_cols = st.columns([2] + [2] * _n)
    _row_cols[0].markdown(f'<span style="font-size:0.78rem;color:{DIM};">{metric}</span>', unsafe_allow_html=True)
    for i, val in enumerate(values):
        _color = GAIN if ("+" in val and "$" in val or "+" in val) else LOSS if "-" in val else FG
        _color = FG if metric in ("# Positions", "Winners", "Losers", "Win Rate",
                                   "Largest Position", "Total Value ($)") else _color
        _row_cols[i + 1].markdown(f'<span style="font-size:0.85rem;color:{_color};font-weight:500;">{val}</span>',
                                   unsafe_allow_html=True)

# ── Allocation bar charts ─────────────────────────────────────────────────────
section_header("Position Weights")

_alloc_cols = st.columns(_n)
for i, name in enumerate(_compare):
    with _alloc_cols[i]:
        d = _pdata.get(name, {})
        df = d.get("df", pd.DataFrame())
        prices = d.get("prices", {})
        color = _palette[i % len(_palette)]

        if df.empty:
            st.caption(f"{name}: no positions")
            continue

        df = df.copy()
        df["value"] = df.apply(lambda r: float(r["shares"]) * prices.get(r["ticker"], float(r.get("cost_basis", 0))), axis=1)
        total = df["value"].sum()
        df["weight"] = df["value"] / total * 100 if total > 0 else 0
        df = df.sort_values("weight", ascending=True).tail(10)

        fig_a = go.Figure(go.Bar(
            x=df["weight"], y=df["ticker"],
            orientation="h",
            marker_color=color,
            text=df["weight"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        ))
        fig_a.update_layout(**carbon_plotly_layout(
            height=max(200, len(df) * 30 + 60),
            title=name,
            margin=dict(l=60, r=40, t=40, b=20),
        ))
        fig_a.update_layout(xaxis=dict(title="Weight (%)"), yaxis=dict(title=""))
        st.plotly_chart(fig_a, use_container_width=True)

# ── Regime scores ─────────────────────────────────────────────────────────────
section_header("Regime Scores by Portfolio")

_rs_rows = []
_all_tickers_seen: set[str] = set()
_score_cache: dict[str, dict[str, float]] = {}

for name in _compare:
    d    = _pdata.get(name, {})
    df   = d.get("df", pd.DataFrame())
    if df.empty:
        continue
    tickers = tuple(df["ticker"].unique())
    _all_tickers_seen.update(tickers)
    with st.spinner(f"Scoring {name}…"):
        _score_cache[name] = _regime_scores_for(name, tickers)

for t in sorted(_all_tickers_seen):
    row = {"Ticker": t}
    for name in _compare:
        row[name] = _score_cache.get(name, {}).get(t, None)
    _rs_rows.append(row)

if _rs_rows:
    _rs_df = pd.DataFrame(_rs_rows)
    flex_table(
        _rs_df,
        columns=[
            {"key": "Ticker", "label": "Ticker", "width": "15%", "align": "left"},
        ] + [
            {"key": name, "label": name, "width": f"{75 // _n}%", "align": "right",
             "numeric": True, "color_scale": "rg",
             "fmt": lambda v: f"{v:.1f}" if isinstance(v, (int, float)) and not pd.isna(v) else "—"}
            for name in _compare
        ],
        key="regime_scores_cmp",
    )

# ── Equity curves ─────────────────────────────────────────────────────────────
section_header("Equity Curves (Normalised)")
st.caption("Requires daily snapshots — visit the Command Center to populate.")

_history_available = False
fig_eq = go.Figure()

for i, name in enumerate(_compare):
    # Trade journal stores snapshots per active portfolio at snapshot time.
    # We surface what's available. In a future phase, journal will be per-portfolio.
    _hist = get_portfolio_history(days=365)
    if not _hist.empty:
        _eq = compute_equity_curve(_hist)
        if not _eq.empty:
            color = _palette[i % len(_palette)]
            fig_eq.add_trace(go.Scatter(
                x=_eq.index, y=_eq["portfolio_norm"],
                name=name, line=dict(color=color, width=2),
            ))
            if i == 0:
                fig_eq.add_trace(go.Scatter(
                    x=_eq.index, y=_eq["benchmark_norm"],
                    name="SPY", line=dict(color=DIM, width=1.5, dash="dash"),
                ))
            _history_available = True

if _history_available:
    fig_eq.add_hline(y=100, line_color=BORDER, line_dash="dot", line_width=1)
    fig_eq.update_layout(**carbon_plotly_layout(
        height=350, title="Indexed Portfolio Values (Base = 100)", hovermode="x unified",
    ))
    fig_eq.update_layout(yaxis=dict(title="Indexed (Base=100)"))
    st.plotly_chart(fig_eq, use_container_width=True)
else:
    st.info("No equity curve data yet. Open the Command Center daily — it auto-snapshots portfolio value on each visit.")
