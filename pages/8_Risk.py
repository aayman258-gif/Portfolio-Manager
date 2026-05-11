"""
Regime-Aware Portfolio Manager
Section 15: Risk Dashboard
VaR · CVaR · Max Drawdown · Factor Exposures · Stress Tests · Correlation Heatmap
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
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent))

from utils.carbon_theme import (
    ACCENT, AMBER, BG, BORDER, CARD, DIM, FG, GAIN, LOSS, SUBTLE,
    apply_carbon_theme, carbon_plotly_layout, page_header, top_nav,
)
from utils.portfolio_store import restore_portfolio_to_session

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Risk Dashboard", page_icon="◈", layout="wide")
apply_carbon_theme()
top_nav("Risk")

page_header("Risk Dashboard", "VaR · CVaR · drawdown · factor exposures · stress scenarios")

# ── Auto-restore portfolio from disk if not in session ────────────────────────
restore_portfolio_to_session()

# ── Check portfolio ───────────────────────────────────────────────────────────
if "positions" not in st.session_state:
    st.warning("No portfolio loaded. Go to Home and load a portfolio first.")
    st.stop()

positions_df: pd.DataFrame = st.session_state["positions"]

# ── Two-column layout ─────────────────────────────────────────────────────────
_lc, _rc = st.columns([1, 3], gap="medium")

with _lc:
    st.markdown(
        f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.12em;'
        f'color:{DIM};margin-bottom:0.5rem;">Controls</div>',
        unsafe_allow_html=True,
    )
    conf_level = st.selectbox("Confidence Level", [0.90, 0.95, 0.99],
                              index=1, format_func=lambda x: f"{x*100:.0f}%")
    lookback = st.selectbox("Lookback Period", ["6mo", "1y", "2y", "3y"], index=1)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _load_returns(tickers: tuple, period: str) -> pd.DataFrame:
    close = alpaca.get_close_prices(list(tickers), period=period)
    if close.empty:
        return pd.DataFrame()
    return close.pct_change().dropna()

@st.cache_data(ttl=3600, show_spinner=False)
def _load_factor_returns(period: str) -> pd.DataFrame:
    """Load SPY, QQQ, TLT, GLD as factor proxies."""
    factors = {"SPY": "S&P 500", "QQQ": "Nasdaq", "TLT": "Long Bonds", "GLD": "Gold"}
    close = alpaca.get_close_prices(list(factors.keys()), period=period)
    if close.empty:
        return pd.DataFrame()
    rets = close.pct_change().dropna()
    rets.columns = [factors.get(c, c) for c in rets.columns]
    return rets

tickers = tuple(positions_df["ticker"].dropna().unique().tolist())
if not tickers:
    st.error("No tickers found in portfolio.")
    st.stop()

with st.spinner("Loading market data…"):
    returns = _load_returns(tickers, lookback)
    factor_returns = _load_factor_returns(lookback)

if returns.empty:
    st.error("Could not load return data.")
    st.stop()

# ── Build weighted portfolio returns ─────────────────────────────────────────
shares_map = dict(zip(positions_df["ticker"], positions_df["shares"].astype(float)))
cost_map   = dict(zip(positions_df["ticker"], positions_df["cost_basis"].astype(float)))

# Use most recent prices for weights
last_prices = returns.iloc[-1] + 1  # approximate — use actual from last row of cumulative
try:
    _latest = alpaca.get_latest_prices(list(tickers))
    lp = pd.Series({t: _latest.get(t, cost_map.get(t, 1.0)) for t in tickers})
except Exception:
    lp = pd.Series({t: cost_map.get(t, 1.0) for t in tickers})

position_values = {t: shares_map.get(t, 0) * float(lp.get(t, cost_map.get(t, 1.0)))
                   for t in tickers if t in returns.columns}
total_val = sum(position_values.values())
weights = pd.Series({t: v / total_val for t, v in position_values.items()}) if total_val > 0 else pd.Series()

common_tickers = [t for t in tickers if t in returns.columns and t in weights.index]
if not common_tickers or weights.empty:
    st.error("Could not compute portfolio weights — no valid price data returned.")
    st.stop()
port_returns = (returns[common_tickers] * weights[common_tickers]).sum(axis=1)

# ── Risk metrics ──────────────────────────────────────────────────────────────
alpha = 1 - conf_level
var_hist  = float(np.percentile(port_returns, alpha * 100))
_tail = port_returns[port_returns <= var_hist]
cvar_hist = float(_tail.mean()) if not _tail.empty else var_hist

# Parametric VaR
mu, sigma = port_returns.mean(), port_returns.std()
var_param = float(stats.norm.ppf(alpha, mu, sigma))

# Max drawdown
cum = (1 + port_returns).cumprod()
rolling_max = cum.cummax()
drawdown = (cum - rolling_max) / rolling_max
max_dd = float(drawdown.min())

# Annualised metrics
ann_vol  = float(port_returns.std() * np.sqrt(252))
ann_ret  = float(port_returns.mean() * 252)
sharpe   = (ann_ret - 0.043) / ann_vol if ann_vol > 0 else 0.0
sortino_downside = float(port_returns[port_returns < 0].std() * np.sqrt(252))
sortino  = (ann_ret - 0.043) / sortino_downside if sortino_downside > 0 else 0.0

# ── Section 1: Key risk metrics → LEFT column ────────────────────────────────
metrics = [
    ("Daily VaR",        f"{var_hist*100:.2f}%",   "Historical", LOSS),
    ("Daily CVaR",       f"{cvar_hist*100:.2f}%",  "Expected shortfall", LOSS),
    ("Param VaR",        f"{var_param*100:.2f}%",  "Normal dist.", AMBER),
    ("Max Drawdown",     f"{max_dd*100:.1f}%",     "Peak to trough", LOSS),
    ("Ann. Volatility",  f"{ann_vol*100:.1f}%",    "1-year", SUBTLE),
    ("Sharpe",           f"{sharpe:.2f}",          "rf = 4.3%", GAIN if sharpe > 1 else SUBTLE),
    ("Sortino",          f"{sortino:.2f}",         "Downside σ", GAIN if sortino > 1 else SUBTLE),
]
with _lc:
    st.markdown(
        f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.12em;'
        f'color:{DIM};margin:0.8rem 0 0.4rem;">Risk Metrics</div>',
        unsafe_allow_html=True,
    )
    for label, val, sub, color in metrics:
        st.markdown(
            f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:4px;'
            f'padding:0.45rem 0.7rem;margin-bottom:0.3rem;'
            f'display:flex;justify-content:space-between;align-items:center;">'
            f'<div>'
            f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:{DIM};">{label}</div>'
            f'<div style="font-size:0.62rem;color:{DIM};margin-top:0.1rem;">{sub}</div>'
            f'</div>'
            f'<div style="font-size:1.05rem;font-weight:500;color:{color};">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Charts → RIGHT column ────────────────────────────────────────────────────
with _rc:
    _cd, _cdd = st.columns(2)

# ── Section 2: Return distribution + Drawdown chart ──────────────────────────
with _cd:
    st.subheader("Return Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=port_returns * 100,
        nbinsx=60,
        marker_color=ACCENT,
        opacity=0.7,
        name="Daily Returns",
    ))
    fig_dist.add_vline(x=var_hist * 100, line_color=LOSS, line_dash="dash",
                       annotation_text=f"VaR {conf_level*100:.0f}%", annotation_font_color=LOSS)
    fig_dist.add_vline(x=cvar_hist * 100, line_color=AMBER, line_dash="dot",
                       annotation_text="CVaR", annotation_font_color=AMBER)
    fig_dist.update_layout(**carbon_plotly_layout(
        title="Daily Portfolio Return Distribution (%)",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=320,
    ))
    st.plotly_chart(fig_dist, use_container_width=True)

with _cdd:
    st.subheader("Drawdown History")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown * 100,
        fill="tozeroy",
        fillcolor=f"rgba(251,113,133,0.20)",
        line=dict(color=LOSS, width=1.2),
        name="Drawdown",
    ))
    fig_dd.update_layout(**carbon_plotly_layout(
        title="Portfolio Drawdown (%)",
        yaxis_title="Drawdown (%)",
        height=320,
    ))
    st.plotly_chart(fig_dd, use_container_width=True)

# ── Sections 3-5: Correlation · Factor Exposures · Stress Tests → RIGHT ───────
# Factor computation (needed before display)
betas: dict = {}
r2s:   dict = {}
if not factor_returns.empty:
    aligned = port_returns.align(factor_returns, join="inner")
    pr_aligned, fr_aligned = aligned
    for factor in fr_aligned.columns:
        try:
            slope, _, r, _, _ = stats.linregress(fr_aligned[factor], pr_aligned)
            betas[factor] = slope
            r2s[factor]   = r ** 2
        except Exception:
            betas[factor] = 0.0
            r2s[factor]   = 0.0

with _rc:
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    if len(common_tickers) > 1:
        corr = returns[common_tickers].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0, LOSS], [0.5, CARD], [1, GAIN]],
            zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            showscale=True,
        ))
        fig_corr.update_layout(**carbon_plotly_layout(
            title="Position Correlation Matrix",
            height=max(280, len(common_tickers) * 42 + 80),
        ))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Need at least 2 positions for a correlation matrix.")

    # Factor exposures
    st.subheader("Factor Exposures")
    if betas:
        _fc = st.columns(len(betas))
        for col, (factor, beta) in zip(_fc, betas.items()):
            color = GAIN if abs(beta) < 0.5 else (AMBER if abs(beta) < 1.0 else LOSS)
            col.markdown(
                f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:4px;'
                f'padding:0.5rem;text-align:center;">'
                f'<div style="font-size:0.58rem;text-transform:uppercase;letter-spacing:0.1em;color:{DIM};">β {factor}</div>'
                f'<div style="font-size:1.1rem;color:{color};margin:0.2rem 0;">{beta:.2f}</div>'
                f'<div style="font-size:0.58rem;color:{DIM};">R² {r2s.get(factor, 0):.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Stress tests
    st.subheader("Stress Test Scenarios")
    st.caption("Estimated portfolio impact based on factor betas and historical analogue shocks.")

_SCENARIOS = {
    "2008 Financial Crisis":     {"S&P 500": -0.565, "Nasdaq": -0.498, "Long Bonds":  0.259, "Gold":  0.052},
    "COVID Crash (Mar 2020)":    {"S&P 500": -0.340, "Nasdaq": -0.305, "Long Bonds":  0.094, "Gold": -0.015},
    "2022 Rate Shock":           {"S&P 500": -0.193, "Nasdaq": -0.325, "Long Bonds": -0.288, "Gold": -0.018},
    "Dot-com Bust (2000-2002)":  {"S&P 500": -0.492, "Nasdaq": -0.778, "Long Bonds":  0.332, "Gold":  0.075},
    "Fed Tightening 50bps":      {"S&P 500": -0.030, "Nasdaq": -0.045, "Long Bonds": -0.040, "Gold": -0.005},
    "Flash Crash (Aug 2015)":    {"S&P 500": -0.115, "Nasdaq": -0.131, "Long Bonds":  0.012, "Gold":  0.027},
}

scenario_impacts = {}
for scenario, shocks in _SCENARIOS.items():
    impact = sum(betas.get(f, 0) * shock for f, shock in shocks.items())
    dollar_impact = impact * total_val
    scenario_impacts[scenario] = {"return": impact, "dollar": dollar_impact}

scenario_rows = []
for scenario, vals in scenario_impacts.items():
    color = LOSS if vals["return"] < -0.10 else (AMBER if vals["return"] < -0.05 else SUBTLE)
    scenario_rows.append({
        "Scenario": scenario,
        "Est. Return": f"{vals['return']*100:.1f}%",
        "Est. P&L": f"${vals['dollar']:+,.0f}",
        "_color": color,
        "_val": vals["return"],
    })

with _rc:
    for row in sorted(scenario_rows, key=lambda x: x["_val"]):
        color = row["_color"]
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'background:{CARD};border:1px solid {BORDER};border-radius:4px;'
            f'padding:0.35rem 0.9rem;margin-bottom:0.25rem;">'
            f'<span style="color:{SUBTLE};font-size:0.78rem;">{row["Scenario"]}</span>'
            f'<span style="color:{color};font-size:0.78rem;font-weight:500;margin-left:2rem;">'
            f'{row["Est. Return"]} &nbsp; {row["Est. P&L"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
