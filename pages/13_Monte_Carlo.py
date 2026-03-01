"""
Regime-Aware Portfolio Manager
Section 13: Monte Carlo Simulation
Geometric Brownian Motion price simulation with regime-conditioned drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

sys.path.append(str(Path(__file__).parent.parent))

from utils.carbon_theme import (
    ACCENT, AMBER, BG, BORDER, CARD, DIM, FG, GAIN, LOSS, SUBTLE,
    apply_carbon_theme, carbon_plotly_layout, page_header,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Monte Carlo Simulation",
    page_icon="🎲",
    layout="wide",
)
apply_carbon_theme()
page_header(
    "🎲 Monte Carlo Simulation",
    "Geometric Brownian Motion · regime-conditioned drift · percentile fan chart",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_TRADING_DAYS = 252


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_prices(ticker: str, period: str = "2y") -> pd.Series:
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw.empty or "Close" not in raw.columns:
        return pd.Series(dtype=float)
    return raw["Close"].dropna()


def _fit_gbm(prices: pd.Series) -> tuple[float, float, float]:
    """Return (mu_annual, sigma_annual, last_price) from a price series."""
    log_rets = np.log(prices / prices.shift(1)).dropna()
    mu_daily    = log_rets.mean()
    sigma_daily = log_rets.std(ddof=1)
    mu_annual    = mu_daily    * _TRADING_DAYS
    sigma_annual = sigma_daily * np.sqrt(_TRADING_DAYS)
    return float(mu_annual), float(sigma_annual), float(prices.iloc[-1])


def _run_simulation(
    S0: float,
    mu: float,
    sigma: float,
    horizon_days: int,
    n_sims: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate GBM paths.

    Returns ndarray of shape (horizon_days + 1, n_sims).
    S(t) = S0 * exp( (mu - 0.5*sigma^2)*t/T + sigma*sqrt(t/T)*Z )
    where T = trading days per year.
    """
    rng = np.random.default_rng(seed)
    dt  = 1.0 / _TRADING_DAYS                       # one day in year-fractions
    drift     = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # daily shocks: (horizon, n_sims)
    Z    = rng.standard_normal((horizon_days, n_sims))
    log_steps = drift + diffusion * Z                # (horizon, n_sims)
    log_paths = np.vstack([np.zeros(n_sims), log_steps.cumsum(axis=0)])
    return S0 * np.exp(log_paths)                    # (horizon+1, n_sims)


def _percentile_fan(
    paths: np.ndarray,
    pcts: list[float] = (5, 25, 50, 75, 95),
) -> pd.DataFrame:
    """Return DataFrame of percentile paths, columns named by percentile."""
    return pd.DataFrame(
        np.percentile(paths, pcts, axis=1).T,
        columns=[f"p{int(p)}" for p in pcts],
    )


def _risk_metrics(final_prices: np.ndarray, S0: float) -> dict:
    returns = (final_prices - S0) / S0
    var_95  = float(np.percentile(returns, 5))
    var_99  = float(np.percentile(returns, 1))
    cvar_95 = float(returns[returns <= var_95].mean())
    cvar_99 = float(returns[returns <= var_99].mean())
    return {
        "mean_return":    float(returns.mean()),
        "median_return":  float(np.median(returns)),
        "std_return":     float(returns.std()),
        "prob_gain":      float((final_prices > S0).mean()),
        "var_95":         var_95,
        "var_99":         var_99,
        "cvar_95":        cvar_95,
        "cvar_99":        cvar_99,
        "p5_price":       float(np.percentile(final_prices, 5)),
        "p25_price":      float(np.percentile(final_prices, 25)),
        "p50_price":      float(np.percentile(final_prices, 50)),
        "p75_price":      float(np.percentile(final_prices, 75)),
        "p95_price":      float(np.percentile(final_prices, 95)),
        "mean_price":     float(final_prices.mean()),
        "max_price":      float(final_prices.max()),
        "min_price":      float(final_prices.min()),
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Simulation Settings")

# Ticker selection
_portfolio_tickers: list[str] = []
if "positions" in st.session_state:
    _df = st.session_state["positions"]
    if isinstance(_df, pd.DataFrame) and "ticker" in _df.columns:
        _portfolio_tickers = sorted(
            t for t in _df["ticker"].astype(str).str.upper().unique()
            if t and t not in ("NAN", "NONE", "")
        )

if _portfolio_tickers:
    tickers_selected = st.sidebar.multiselect(
        "Tickers (from portfolio)",
        options=_portfolio_tickers,
        default=_portfolio_tickers[:3],
        help="Select one or more tickers from your loaded portfolio.",
    )
    extra_tickers = st.sidebar.text_input(
        "Additional tickers (comma-separated)",
        placeholder="e.g. NVDA, META",
    )
    if extra_tickers:
        for t in extra_tickers.upper().replace(" ", "").split(","):
            if t and t not in tickers_selected:
                tickers_selected.append(t)
else:
    raw_input = st.sidebar.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, NVDA",
        help="No portfolio loaded — enter tickers manually.",
    )
    tickers_selected = [
        t.strip().upper() for t in raw_input.split(",") if t.strip()
    ]

st.sidebar.divider()

n_sims = st.sidebar.select_slider(
    "Simulations",
    options=[100, 500, 1_000, 5_000, 10_000],
    value=1_000,
    help="More simulations = smoother distribution but slower to compute.",
)

horizon_days = st.sidebar.select_slider(
    "Horizon (trading days)",
    options=[21, 42, 63, 126, 252, 504],
    value=252,
    format_func=lambda d: {21:"1 month",42:"2 months",63:"3 months",
                            126:"6 months",252:"1 year",504:"2 years"}[d],
)

lookback = st.sidebar.selectbox(
    "Parameter lookback",
    options=["6mo", "1y", "2y", "5y"],
    index=2,
    help="Historical window used to estimate drift (μ) and volatility (σ).",
)

show_paths = st.sidebar.number_input(
    "Sample paths to plot",
    min_value=10, max_value=300, value=100, step=10,
    help="Number of individual simulated paths drawn on the fan chart.",
)

use_current_regime = st.sidebar.checkbox(
    "Apply regime drift adjustment",
    value=True,
    help=(
        "When enabled, drift is nudged toward the regime-implied equity premium:\n"
        "Low Vol → full CAPM drift · High Vol → risk-free rate only · "
        "Trending → CAPM + 1.5× premium · Mean Reversion → CAPM · "
        "Uncertain → average of CAPM and risk-free."
    ),
)

st.sidebar.divider()
st.sidebar.caption(
    f"Model: Geometric Brownian Motion  \n"
    f"dS = μS dt + σS dW  \n"
    f"Params estimated from {lookback} of daily log-returns."
)

# ── Regime drift multiplier ───────────────────────────────────────────────────

_REGIME_MU_BLEND: dict[str, float] = {
    # blend weight toward CAPM mu (vs. risk-free floor)
    "Low Vol":        1.00,
    "High Vol":       0.00,   # use risk-free only
    "Trending":       1.20,   # slight momentum boost (capped later)
    "Mean Reversion": 1.00,
    "Uncertain":      0.50,
}
_RF = 0.04   # annualised risk-free rate


def _regime_adjusted_mu(mu_hist: float, regime: str) -> float:
    """Blend historical drift toward risk-free based on regime."""
    blend = _REGIME_MU_BLEND.get(regime, 1.0)
    adjusted = _RF + blend * (mu_hist - _RF)
    return adjusted


# ── Run button ────────────────────────────────────────────────────────────────

if not tickers_selected:
    st.warning("Select at least one ticker in the sidebar to run the simulation.")
    st.stop()

run_clicked = st.button("▶ Run Simulation", type="primary")

if not run_clicked and "mc_results" not in st.session_state:
    st.info("Configure settings in the sidebar, then click **Run Simulation**.")
    st.stop()

# ── Compute simulations ───────────────────────────────────────────────────────

if run_clicked:
    # Detect current regime (optional)
    current_regime = "Unknown"
    if use_current_regime:
        try:
            from data.market_data import MarketDataLoader
            from calculations.regime_detector import RegimeDetector
            _ml = MarketDataLoader()
            _det = RegimeDetector()
            _spy, _vix = _ml.align_data(
                _ml.load_index_data("SPY", "2y"),
                _ml.load_vix_data("2y"),
            )
            current_regime = str(_det.classify_regime(_spy, _vix)[0].iloc[-1])
        except Exception:
            pass

    results: dict[str, dict] = {}
    errors:  dict[str, str]  = {}

    with st.spinner(f"Running {n_sims:,} × {len(tickers_selected)} simulations …"):
        for ticker in tickers_selected:
            prices = _fetch_prices(ticker, period=lookback)
            if prices.empty or len(prices) < 30:
                errors[ticker] = "Insufficient price history."
                continue
            mu, sigma, S0 = _fit_gbm(prices)
            if use_current_regime and current_regime not in ("Unknown", None):
                mu = _regime_adjusted_mu(mu, current_regime)
            paths = _run_simulation(S0, mu, sigma, horizon_days, n_sims)
            fan   = _percentile_fan(paths)
            risk  = _risk_metrics(paths[-1], S0)
            results[ticker] = {
                "paths":   paths,
                "fan":     fan,
                "risk":    risk,
                "S0":      S0,
                "mu":      mu,
                "sigma":   sigma,
                "prices":  prices,
            }

    st.session_state["mc_results"]       = results
    st.session_state["mc_errors"]        = errors
    st.session_state["mc_regime"]        = current_regime
    st.session_state["mc_horizon_days"]  = horizon_days
    st.session_state["mc_n_sims"]        = n_sims
    st.session_state["mc_show_paths"]    = show_paths

# ── Display results ───────────────────────────────────────────────────────────

results      = st.session_state.get("mc_results", {})
errors       = st.session_state.get("mc_errors", {})
current_regime = st.session_state.get("mc_regime", "Unknown")
horizon_days = st.session_state.get("mc_horizon_days", horizon_days)
n_sims       = st.session_state.get("mc_n_sims", n_sims)
show_paths   = st.session_state.get("mc_show_paths", show_paths)

if errors:
    for tkr, msg in errors.items():
        st.warning(f"**{tkr}**: {msg}")

if not results:
    st.error("No simulation results. Check tickers and try again.")
    st.stop()

# Regime banner
_regime_colors = {
    "Low Vol": GAIN, "High Vol": LOSS, "Trending": ACCENT,
    "Mean Reversion": AMBER, "Uncertain": SUBTLE, "Unknown": DIM,
}
_rc = _regime_colors.get(current_regime, DIM)
st.markdown(
    f'<div style="display:inline-block;padding:0.3rem 1rem;margin-bottom:1rem;'
    f'border:1px solid {_rc};border-radius:6px;font-size:0.78rem;'
    f'font-style:italic;color:{_rc};">'
    f'Regime: <strong>{current_regime}</strong>'
    f'{"  ·  drift adjusted" if use_current_regime and current_regime not in ("Unknown","") else ""}'
    f'</div>',
    unsafe_allow_html=True,
)

# Ticker selector (when >1 ticker)
if len(results) > 1:
    selected_ticker = st.selectbox(
        "View details for:", options=list(results.keys()), key="mc_detail_ticker"
    )
else:
    selected_ticker = list(results.keys())[0]

res    = results[selected_ticker]
paths  = res["paths"]
fan    = res["fan"]
risk   = res["risk"]
S0     = res["S0"]
sigma  = res["sigma"]
mu     = res["mu"]
prices = res["prices"]

# ── Section 1: Fan chart ──────────────────────────────────────────────────────

st.subheader(f"📈 Simulated Price Paths — {selected_ticker}")

x_axis = list(range(horizon_days + 1))

fig_paths = go.Figure()

# Sample paths (thin, very transparent)
rng_disp = np.random.default_rng(0)
path_idx = rng_disp.choice(n_sims, size=min(show_paths, n_sims), replace=False)
for i, idx in enumerate(path_idx):
    fig_paths.add_trace(go.Scatter(
        x=x_axis, y=paths[:, idx],
        mode="lines",
        line=dict(color=f"rgba(34,211,238,0.06)", width=0.6),
        showlegend=(i == 0),
        name="Individual paths",
        hoverinfo="skip",
    ))

# Percentile bands (p5–p95 fill, p25–p75 fill, median)
fig_paths.add_trace(go.Scatter(
    x=x_axis + x_axis[::-1],
    y=list(fan["p95"]) + list(fan["p5"])[::-1],
    fill="toself",
    fillcolor="rgba(34,211,238,0.06)",
    line=dict(color="rgba(0,0,0,0)"),
    name="5th–95th pct",
    hoverinfo="skip",
))
fig_paths.add_trace(go.Scatter(
    x=x_axis + x_axis[::-1],
    y=list(fan["p75"]) + list(fan["p25"])[::-1],
    fill="toself",
    fillcolor="rgba(34,211,238,0.14)",
    line=dict(color="rgba(0,0,0,0)"),
    name="25th–75th pct",
    hoverinfo="skip",
))
fig_paths.add_trace(go.Scatter(
    x=x_axis, y=fan["p50"],
    mode="lines",
    line=dict(color=ACCENT, width=2),
    name="Median (p50)",
))
fig_paths.add_trace(go.Scatter(
    x=x_axis, y=fan["p5"],
    mode="lines",
    line=dict(color=LOSS, width=1, dash="dot"),
    name="5th percentile",
))
fig_paths.add_trace(go.Scatter(
    x=x_axis, y=fan["p95"],
    mode="lines",
    line=dict(color="#4ade80", width=1, dash="dot"),
    name="95th percentile",
))
# Current price baseline
fig_paths.add_hline(
    y=S0, line_dash="dash", line_color=DIM, line_width=1,
    annotation_text=f"  Current ${S0:.2f}",
    annotation_font_color=SUBTLE,
)

fig_paths.update_layout(
    **carbon_plotly_layout(
        title=f"{selected_ticker} — {n_sims:,} GBM paths · {horizon_days}d horizon",
        xaxis_title="Trading days",
        yaxis_title="Price ($)",
        height=480,
    )
)
st.plotly_chart(fig_paths, use_container_width=True)

# ── Section 2: Key metrics ────────────────────────────────────────────────────

st.subheader("📊 Outcome Metrics")

_horizon_label = {
    21:"1 month", 42:"2 months", 63:"3 months",
    126:"6 months", 252:"1 year", 504:"2 years",
}.get(horizon_days, f"{horizon_days}d")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Current Price",  f"${S0:.2f}")
with c2:
    _med = risk["p50_price"]
    st.metric("Median Outcome", f"${_med:.2f}", f"{(_med/S0-1)*100:+.1f}%")
with c3:
    st.metric("P(Gain)", f"{risk['prob_gain']*100:.1f}%")
with c4:
    st.metric(f"VaR 95% ({_horizon_label})", f"{risk['var_95']*100:.2f}%",
              help="5th-percentile return — losses exceed this 5% of the time.")
with c5:
    st.metric(f"CVaR 95%", f"{risk['cvar_95']*100:.2f}%",
              help="Expected loss given that we are in the worst 5% of outcomes.")

# Annualised params
c6, c7, c8 = st.columns(3)
with c6:
    st.metric("Annual Drift (μ)", f"{mu*100:.2f}%",
              help="Estimated from historical log-returns, regime-adjusted if enabled.")
with c7:
    st.metric("Annual Volatility (σ)", f"{sigma*100:.2f}%")
with c8:
    st.metric("Sharpe (implied)", f"{(mu - _RF) / sigma:.2f}" if sigma > 0 else "—",
              help="(μ − rf) / σ using the simulated drift and 4% risk-free rate.")

# ── Section 3: Distribution + percentile table ────────────────────────────────

st.subheader("📉 Final Price Distribution")

col_hist, col_tbl = st.columns([3, 2])

final_prices = paths[-1]

with col_hist:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=final_prices,
        nbinsx=80,
        marker_color=ACCENT,
        opacity=0.7,
        name="Final price",
    ))
    # Vertical lines for key percentiles
    for pct_val, color, label in [
        (risk["p5_price"],  LOSS,    "p5"),
        (risk["p25_price"], AMBER,   "p25"),
        (risk["p50_price"], ACCENT,  "p50"),
        (risk["p75_price"], "#4ade80", "p75"),
        (risk["p95_price"], "#4ade80", "p95"),
        (S0,                DIM,     "current"),
    ]:
        fig_hist.add_vline(
            x=pct_val, line_dash="dot", line_color=color, line_width=1.5,
            annotation_text=f"  {label}",
            annotation_font_color=color,
            annotation_font_size=9,
        )
    fig_hist.update_layout(
        **carbon_plotly_layout(
            title=f"Distribution of {selected_ticker} prices at day {horizon_days}",
            xaxis_title="Final price ($)",
            yaxis_title="Simulations",
            height=380,
            showlegend=False,
        )
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_tbl:
    _pct_rows = [
        ("5th",  risk["p5_price"],  (risk["p5_price"]  / S0 - 1) * 100),
        ("10th", float(np.percentile(final_prices, 10)),
                 (float(np.percentile(final_prices, 10)) / S0 - 1) * 100),
        ("25th", risk["p25_price"], (risk["p25_price"] / S0 - 1) * 100),
        ("50th", risk["p50_price"], (risk["p50_price"] / S0 - 1) * 100),
        ("75th", risk["p75_price"], (risk["p75_price"] / S0 - 1) * 100),
        ("90th", float(np.percentile(final_prices, 90)),
                 (float(np.percentile(final_prices, 90)) / S0 - 1) * 100),
        ("95th", risk["p95_price"], (risk["p95_price"] / S0 - 1) * 100),
        ("Mean", risk["mean_price"], risk["mean_return"] * 100),
    ]
    _th_s = (
        f"padding:0.45rem 0.8rem;font-size:0.60rem;text-transform:uppercase;"
        f"letter-spacing:0.13em;color:{ACCENT};border-bottom:1px solid {BORDER};"
        f"text-align:{{align}};"
    )
    _td_s = (
        f"padding:0.5rem 0.8rem;font-size:0.84rem;color:{{color}};"
        f"border-bottom:1px solid {BORDER};text-align:{{align}};"
    )
    _hrow = (
        f'<th style="{_th_s.format(align="left")}">Percentile</th>'
        f'<th style="{_th_s.format(align="right")}">Price</th>'
        f'<th style="{_th_s.format(align="right")}">Return</th>'
    )
    _rows = ""
    for label, price, ret in _pct_rows:
        _col = GAIN if ret >= 0 else LOSS
        _rows += (
            f'<tr>'
            f'<td style="{_td_s.format(color=SUBTLE, align="left")}">{label}</td>'
            f'<td style="{_td_s.format(color=FG,     align="right")}">${price:.2f}</td>'
            f'<td style="{_td_s.format(color=_col,   align="right")}">{ret:+.1f}%</td>'
            f'</tr>'
        )
    st.markdown(
        f'<div style="border:1px solid {BORDER};border-radius:8px;overflow:hidden;margin-top:0.5rem;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>{_hrow}</tr></thead>'
        f'<tbody>{_rows}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

# ── Section 4: VaR / CVaR bar chart ──────────────────────────────────────────

st.subheader("🛡️ Risk Summary")

col_risk1, col_risk2 = st.columns(2)

with col_risk1:
    _var_labels = ["VaR 90%", "VaR 95%", "VaR 99%", "CVaR 95%", "CVaR 99%"]
    _var_values = [
        float(np.percentile((final_prices - S0) / S0, 10)) * 100,
        risk["var_95"] * 100,
        risk["var_99"] * 100,
        risk["cvar_95"] * 100,
        risk["cvar_99"] * 100,
    ]
    fig_var = go.Figure(go.Bar(
        x=_var_labels,
        y=_var_values,
        marker_color=[LOSS] * len(_var_labels),
        text=[f"{v:.2f}%" for v in _var_values],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig_var.update_layout(
        **carbon_plotly_layout(
            title=f"Value at Risk & CVaR — {_horizon_label} horizon",
            yaxis_title="Return (%)",
            height=340,
            showlegend=False,
        )
    )
    fig_var.add_hline(y=0, line_dash="dash", line_color=BORDER, line_width=1)
    st.plotly_chart(fig_var, use_container_width=True)

with col_risk2:
    # Return distribution: probability of gain buckets
    _buckets = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    _rets_pct = ((final_prices - S0) / S0) * 100
    _bucket_counts = []
    _bucket_labels = []
    for i in range(len(_buckets) - 1):
        lo, hi = _buckets[i], _buckets[i + 1]
        _bucket_counts.append(float(((lo <= _rets_pct) & (_rets_pct < hi)).mean() * 100))
        _bucket_labels.append(f"{lo:+d}% to {hi:+d}%")
    # Tails
    _bucket_labels.insert(0, f"< {_buckets[0]}%")
    _bucket_counts.insert(0, float((_rets_pct < _buckets[0]).mean() * 100))
    _bucket_labels.append(f"≥ {_buckets[-1]}%")
    _bucket_counts.append(float((_rets_pct >= _buckets[-1]).mean() * 100))

    _bar_colors = [
        LOSS if "−" in lbl or lbl.startswith("< ") or
        any(lbl.startswith(f"{n:+d}") for n in range(-40, 0, 10))
        else GAIN
        for lbl in _bucket_labels
    ]
    # simple colour: negative bucket → LOSS, else GAIN
    _bar_colors = [
        LOSS if i < (_bucket_labels.index("0% to +10%") if "0% to +10%" in _bucket_labels else len(_bucket_labels))
        else GAIN
        for i in range(len(_bucket_labels))
    ]
    # more readable: colour based on midpoint sign
    _bar_colors = []
    for lbl in _bucket_labels:
        if lbl.startswith("< ") or (
            any(f"{n:+d}%" in lbl for n in range(-99, 0))
        ):
            _bar_colors.append(LOSS)
        else:
            _bar_colors.append(GAIN)

    fig_prob = go.Figure(go.Bar(
        x=_bucket_labels,
        y=_bucket_counts,
        marker_color=_bar_colors,
        text=[f"{v:.1f}%" for v in _bucket_counts],
        textposition="outside",
        textfont=dict(size=9),
    ))
    fig_prob.update_layout(
        **carbon_plotly_layout(
            title="Probability by return bucket",
            yaxis_title="% of simulations",
            height=340,
            showlegend=False,
        )
    )
    fig_prob.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_prob, use_container_width=True)

# ── Section 5: Multi-ticker comparison ───────────────────────────────────────

if len(results) > 1:
    st.subheader("🔀 Multi-Ticker Comparison")

    summary_rows = []
    for tkr, r in results.items():
        rm = r["risk"]
        summary_rows.append({
            "Ticker":         tkr,
            "Current ($)":    f"${r['S0']:.2f}",
            "Median ($)":     f"${rm['p50_price']:.2f}",
            "Median Return":  f"{rm['median_return']*100:+.1f}%",
            "P(Gain)":        f"{rm['prob_gain']*100:.1f}%",
            "VaR 95%":        f"{rm['var_95']*100:.2f}%",
            "CVaR 95%":       f"{rm['cvar_95']*100:.2f}%",
            "σ (annual)":     f"{r['sigma']*100:.1f}%",
            "μ (annual)":     f"{r['mu']*100:.1f}%",
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Overlay median paths
    fig_multi = go.Figure()
    _colors_multi = [
        ACCENT, LOSS, AMBER, "#4ade80", "#a78bfa", "#f472b6",
        "#fb923c", "#34d399", "#60a5fa", "#e879f9",
    ]
    for i, (tkr, r) in enumerate(results.items()):
        fan_i = r["fan"]
        x_i   = list(range(len(fan_i)))
        color  = _colors_multi[i % len(_colors_multi)]
        # IQR band
        fig_multi.add_trace(go.Scatter(
            x=x_i + x_i[::-1],
            y=list(fan_i["p75"]) + list(fan_i["p25"])[::-1],
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_multi.add_trace(go.Scatter(
            x=x_i, y=fan_i["p50"],
            mode="lines",
            line=dict(color=color, width=1.8),
            name=f"{tkr} median",
        ))
    fig_multi.update_layout(
        **carbon_plotly_layout(
            title=f"Median price paths — {horizon_days}d horizon",
            xaxis_title="Trading days",
            yaxis_title="Price ($)",
            height=420,
        )
    )
    st.plotly_chart(fig_multi, use_container_width=True)

# ── Methodology note ─────────────────────────────────────────────────────────

with st.expander("📚 Methodology"):
    st.markdown(f"""
**Model: Geometric Brownian Motion (GBM)**

$$S(t) = S_0 \\; \\exp\\!\\left(\\left(\\mu - \\tfrac{{1}}{{2}}\\sigma^2\\right)t + \\sigma\\sqrt{{t}}\\,Z\\right), \\quad Z \\sim \\mathcal{{N}}(0,1)$$

| Parameter | Estimation |
|-----------|-----------|
| **μ** (drift) | Mean of daily log-returns × 252, regime-adjusted if enabled |
| **σ** (volatility) | Std-dev of daily log-returns × √252 |
| **S₀** | Last closing price from yfinance |

**Regime drift adjustment**

When enabled, drift is blended toward the risk-free rate based on the current regime:

| Regime | Blend weight |
|--------|-------------|
| Low Vol | 100% of historical μ |
| High Vol | 0% (risk-free floor only) |
| Trending | 120% of historical μ |
| Mean Reversion | 100% |
| Uncertain | 50% |

**Risk measures**

- **VaR p%**: the p-th percentile of the simulated return distribution (e.g. VaR 95% = 5th percentile)
- **CVaR p%**: conditional VaR — average of all returns below the VaR threshold (also called Expected Shortfall)

**Limitations**

- GBM assumes constant drift and volatility and log-normally distributed returns — real markets exhibit fat tails, volatility clustering, and jumps
- Historical parameter estimates are backward-looking and may not represent future conditions
- This is for educational and analytical purposes only — *not financial advice*
""")
