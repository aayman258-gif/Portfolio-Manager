"""
Regime-Aware Portfolio Manager
Section 9: Portfolio Hedge Analyzer — options-based downside protection.

Reads live portfolio from session state (positions + cash from Home page).
Provides both:
  - Portfolio-level SPY hedges (scaled by portfolio beta)
  - Individual position hedges (protective put, collar, covered call per stock)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from data.options_data import OptionsDataLoader
from calculations.options_analytics import OptionsAnalytics
from calculations.regime_detector import RegimeDetector
from data.market_data import MarketDataLoader
from utils.carbon_theme import apply_carbon_theme, carbon_plotly_layout, page_header

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Portfolio Hedges", page_icon="🛡️", layout="wide")
apply_carbon_theme()
page_header(
    "🛡️ Portfolio Hedge Analyzer",
    "Options-based downside protection — portfolio-aware, regime-driven",
)

# ── Singletons ────────────────────────────────────────────────────────────────
options_loader  = OptionsDataLoader()
options_calc    = OptionsAnalytics()
market_loader   = MarketDataLoader()
regime_detector = RegimeDetector()


# ── Cached data helpers ───────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner="Detecting regime…")
def _get_regime() -> str:
    try:
        spy_data = market_loader.load_index_data("SPY", "1y")
        vix_data = market_loader.load_vix_data("1y")
        spy_px, vix_px = market_loader.align_data(spy_data, vix_data)
        regime, _ = regime_detector.classify_regime(spy_px, vix_px)
        return str(regime.iloc[-1])
    except Exception:
        return "Unknown"


@st.cache_data(ttl=300, show_spinner=False)
def _spy_expirations() -> list:
    try:
        return list(yf.Ticker("SPY").options)
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner="Loading SPY chain…")
def _spy_chain(expiration: str):
    calls, puts, price = options_loader.get_options_chain("SPY", expiration)
    return calls, puts, float(price)


@st.cache_data(ttl=120, show_spinner=False)
def _spot(ticker: str) -> float:
    try:
        h = yf.Ticker(ticker).history(period="2d")
        return float(h["Close"].iloc[-1]) if not h.empty else 0.0
    except Exception:
        return 0.0


@st.cache_data(ttl=300, show_spinner=False)
def _stock_chain(ticker: str, expiration: str):
    try:
        calls, puts, price = options_loader.get_options_chain(ticker, expiration)
        return calls, puts, float(price)
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), 0.0


@st.cache_data(ttl=300, show_spinner=False)
def _stock_expirations(ticker: str) -> list:
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []


@st.cache_data(ttl=900, show_spinner="Computing portfolio beta…")
def _calc_betas(tickers_tuple: tuple) -> dict:
    """Batch OLS beta calculation vs SPY over 1 year."""
    tickers = list(tickers_tuple)
    all_tks = tickers + ["SPY"]
    try:
        raw = yf.download(all_tks, period="1y", progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw.rename(columns={"Close": all_tks[0]})[all_tks]
        rets = close.pct_change().dropna()
    except Exception:
        return {t: 1.0 for t in tickers}

    if "SPY" not in rets.columns:
        return {t: 1.0 for t in tickers}

    spy_r = rets["SPY"].dropna()
    betas = {}
    for tk in tickers:
        if tk not in rets.columns:
            betas[tk] = 1.0
            continue
        stk_r = rets[tk].dropna()
        common = spy_r.index.intersection(stk_r.index)
        if len(common) < 30:
            betas[tk] = 1.0
            continue
        cov = np.cov(stk_r.loc[common].values, spy_r.loc[common].values, ddof=1)
        betas[tk] = float(cov[0, 1] / (cov[1, 1] + 1e-12))
    return betas


# ── Option chain helpers ──────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if not (np.isnan(v) or np.isinf(v)) else default
    except Exception:
        return default


def _nearest_put(chain: pd.DataFrame, target_K: float):
    """Return (strike, premium) of the chain row nearest target_K."""
    diff = (chain["strike"] - target_K).abs()
    row  = chain.loc[diff.idxmin()]
    prem = _safe_float(row.get("lastPrice"))
    if prem < 0.01:
        prem = _safe_float(row.get("midPrice"))
    if prem < 0.01:
        bid = _safe_float(row.get("bid"))
        ask = _safe_float(row.get("ask"))
        prem = (bid + ask) / 2 if ask > 0 else 0.0
    return float(row["strike"]), max(prem, 0.01)


def _nearest_call(chain: pd.DataFrame, target_K: float):
    diff = (chain["strike"] - target_K).abs()
    row  = chain.loc[diff.idxmin()]
    prem = _safe_float(row.get("lastPrice"))
    if prem < 0.01:
        prem = _safe_float(row.get("midPrice"))
    if prem < 0.01:
        bid = _safe_float(row.get("bid"))
        ask = _safe_float(row.get("ask"))
        prem = (bid + ask) / 2 if ask > 0 else 0.0
    return float(row["strike"]), max(prem, 0.01)


def _get_iv(chain: pd.DataFrame, strike: float) -> float:
    diff = (chain["strike"] - strike).abs()
    row  = chain.loc[diff.idxmin()]
    return max(_safe_float(row.get("impliedVolatility"), 0.20), 0.05)


# ── Regime ────────────────────────────────────────────────────────────────────
current_regime = _get_regime()

_REGIME_COLORS = {
    "Low Vol":        "#22d3ee",
    "High Vol":       "#fb7185",
    "Trending":       "#22d3ee",
    "Mean Reversion": "#f59e0b",
    "Uncertain":      "#a78bfa",
    "Unknown":        "#888888",
}
rc = _REGIME_COLORS.get(current_regime, "#888888")
st.markdown(
    f"""<div style="background:{rc}22;border:1px solid {rc}55;border-radius:8px;
    padding:10px 18px;margin-bottom:16px;font-size:15px;">
    🌍 <strong>Current Regime:</strong> {current_regime}
    — adjust hedge sizing accordingly.</div>""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PORTFOLIO OVERVIEW (auto-read from session state)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Portfolio Overview")

_positions  = st.session_state.get("positions", pd.DataFrame())
_cash       = float(st.session_state.get("cash_balance", 0.0))
_sv_prices  = st.session_state.get("current_prices", {}) or {}
_sv_total   = float(st.session_state.get("total_value", 0.0))

_no_portfolio = (
    _positions is None
    or not isinstance(_positions, pd.DataFrame)
    or _positions.empty
)

pos_df = pd.DataFrame()  # always defined

if _no_portfolio:
    st.info("No portfolio loaded. Go to **Home** to enter positions, or use the manual inputs below.")
    col_mv, col_mb = st.columns(2)
    with col_mv:
        portfolio_value = st.number_input(
            "Portfolio Value ($)", min_value=1_000.0, value=100_000.0,
            step=5_000.0, format="%.0f",
        )
    with col_mb:
        portfolio_beta = st.slider(
            "Portfolio Beta vs SPY", min_value=0.3, max_value=2.5,
            value=1.0, step=0.05,
            help="Beta = 1 means portfolio moves 1:1 with SPY",
        )
    spy_price    = _spot("SPY")
    hedge_tickers: list = []

else:
    # Build per-position table with live prices
    rows = []
    for _, pos in _positions.iterrows():
        ticker = str(pos.get("ticker", ""))
        shares = float(pos.get("shares", 0))
        price  = _safe_float(_sv_prices.get(ticker)) or _spot(ticker)
        value  = shares * price
        rows.append({"Ticker": ticker, "Shares": shares, "Price": price, "Value": value})

    pos_df = pd.DataFrame(rows)
    equity_value = float(pos_df["Value"].sum())
    total_value  = _sv_total if _sv_total > 0 else equity_value + _cash

    # Compute betas via batch download
    tickers_tuple = tuple(pos_df["Ticker"].tolist())
    beta_map = _calc_betas(tickers_tuple)

    pos_df["Beta"]   = pos_df["Ticker"].map(beta_map).fillna(1.0)
    pos_df["Weight"] = pos_df["Value"] / equity_value if equity_value > 0 else 0.0
    portfolio_beta   = float((pos_df["Beta"] * pos_df["Weight"]).sum()) if not pos_df.empty else 1.0
    portfolio_value  = total_value
    spy_price        = _spot("SPY")

    col_tv, col_eq, col_cash, col_beta, col_spy = st.columns(5)
    col_tv.metric("Total Portfolio",  f"${portfolio_value:,.0f}")
    col_eq.metric("Equities Value",   f"${equity_value:,.0f}")
    col_cash.metric("Cash Balance",   f"${_cash:,.0f}")
    col_beta.metric(
        "Portfolio Beta", f"{portfolio_beta:.2f}",
        help="Weighted-average beta vs SPY, estimated via OLS over 1 year",
    )
    col_spy.metric("SPY Price", f"${spy_price:.2f}")

    with st.expander("📊 Position Detail", expanded=False):
        st.dataframe(
            pos_df[["Ticker", "Shares", "Price", "Value", "Beta", "Weight"]].style.format(
                {"Price": "${:.2f}", "Value": "${:,.0f}", "Beta": "{:.2f}", "Weight": "{:.1%}"}
            ),
            use_container_width=True, hide_index=True,
        )

    hedge_tickers = list(pos_df["Ticker"].unique())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — HEDGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("⚙️ Hedge Configuration")

col_exp, col_prot, col_call = st.columns(3)

with col_exp:
    all_exps    = _spy_expirations()
    exp_options = []
    for exp in all_exps[:12]:
        try:
            dte = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days
            if dte >= 14:
                exp_options.append(f"{exp} ({dte}d)")
        except Exception:
            continue

    if not exp_options:
        st.error("No SPY expirations with DTE ≥ 14 are currently available. Try refreshing.")
        st.stop()

    selected_exp_str = st.selectbox(
        "Hedge Expiration", exp_options, index=min(2, len(exp_options) - 1)
    )
    if not selected_exp_str:
        st.error("Could not resolve an expiration. Please refresh.")
        st.stop()

    selected_exp = selected_exp_str.split(" ")[0]
    exp_dte      = int(selected_exp_str.split("(")[1].replace("d)", ""))
    T_hedge      = max(exp_dte / 365, 0.003)

with col_prot:
    protection_pct = st.slider(
        "Put Strike (% below spot)", 2.0, 20.0, 5.0, 0.5,
        help="e.g. 5% → buy put at SPY × 0.95",
    )

with col_call:
    collar_call_pct = st.slider(
        "Collar Call Strike (% above spot)", 2.0, 20.0, 5.0, 0.5,
        help="Sell this far OTM to offset put cost",
    )

# ── Load SPY chain ─────────────────────────────────────────────────────────────
calls_spy, puts_spy, spy_live = _spy_chain(selected_exp)

if puts_spy.empty or calls_spy.empty:
    st.error("Could not load SPY options chain — try a different expiration.")
    st.stop()

if spy_live > 0:
    spy_price = spy_live

contracts_needed = max(1, int(np.ceil(portfolio_value * portfolio_beta / (spy_price * 100))))

# ── SPY hedge calculations ─────────────────────────────────────────────────────

# Protective Put
put_K, put_prem  = _nearest_put(puts_spy, spy_price * (1 - protection_pct / 100))
put_iv           = _get_iv(puts_spy, put_K)
if put_prem <= 0.01:
    put_prem = options_calc.black_scholes(spy_price, put_K, T_hedge, put_iv, "put")
put_total        = put_prem * contracts_needed * 100
put_annual       = put_total * (365 / exp_dte)

# Put Spread (buy put_K, sell deeper OTM put)
ps_short_K, ps_short_prem = _nearest_put(puts_spy, spy_price * (1 - 2 * protection_pct / 100))
ps_iv        = _get_iv(puts_spy, ps_short_K)
if ps_short_prem <= 0.01:
    ps_short_prem = options_calc.black_scholes(spy_price, ps_short_K, T_hedge, ps_iv, "put")
ps_net       = put_prem - ps_short_prem
ps_total     = ps_net * contracts_needed * 100
ps_annual    = ps_total * (365 / exp_dte)

# Collar (buy put_K, sell call)
call_K, call_prem = _nearest_call(calls_spy, spy_price * (1 + collar_call_pct / 100))
call_iv      = _get_iv(calls_spy, call_K)
if call_prem <= 0.01:
    call_prem = options_calc.black_scholes(spy_price, call_K, T_hedge, call_iv, "call")
collar_net   = put_prem - call_prem
collar_total = collar_net * contracts_needed * 100
collar_annual = collar_total * (365 / exp_dte)

# Put Ratio Spread (buy 1× put_K, sell 2× ps_short_K)
ratio_net    = put_prem - 2 * ps_short_prem
ratio_total  = ratio_net * contracts_needed * 100
ratio_annual = ratio_total * (365 / exp_dte)
ratio_max_gain = (put_K - ps_short_K) * contracts_needed * 100

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PORTFOLIO-LEVEL HEDGE (SPY)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🛡️ Portfolio-Level Hedge Strategies (via SPY)")

_URGENCY = {
    "High Vol":       ("⚠️ HIGH URGENCY",  "ef4444", "Volatility is elevated — full protection recommended."),
    "Trending":       ("📈 MODERATE",       "4a9eff", "Trend in place — collar or spread is efficient."),
    "Low Vol":        ("🟢 LOW URGENCY",    "22c55e", "Calm markets — light hedge or no hedge needed."),
    "Mean Reversion": ("🟡 MODERATE",       "f59e0b", "Choppy conditions — put spread recommended."),
    "Uncertain":      ("🔵 CAUTIOUS",       "a78bfa", "Ambiguous regime — maintain moderate protection."),
    "Unknown":        ("❓ UNKNOWN",         "6b7a8f", "Insufficient data — maintain baseline protection."),
}
urg_label, urg_col, urg_msg = _URGENCY.get(current_regime, _URGENCY["Unknown"])
st.markdown(
    f"""<div style="background:#{urg_col}22;border:1px solid #{urg_col}55;
    border-radius:8px;padding:10px 18px;margin-bottom:16px;">
    <strong>{urg_label}</strong> — {urg_msg}</div>""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("#### 🔴 Protective Put")
    st.markdown(f"Buy **{contracts_needed}×** ${put_K:.0f} put")
    st.metric("Premium", f"${put_prem:.2f}/contract")
    st.metric("Total Cost", f"${put_total:,.0f}")
    st.metric("Annualized", f"${put_annual:,.0f}",
              f"{put_annual / portfolio_value * 100:.2f}% of portfolio")
    st.markdown(f"Protected below **${put_K:.0f}** ({protection_pct:.1f}% OTM)")

with c2:
    st.markdown("#### 🟡 Put Spread")
    st.markdown(f"Buy **${put_K:.0f}p**, Sell **${ps_short_K:.0f}p**")
    st.metric("Net Premium", f"${ps_net:.2f}/contract")
    st.metric("Total Cost", f"${ps_total:,.0f}")
    st.metric("Annualized", f"${ps_annual:,.0f}",
              f"{ps_annual / portfolio_value * 100:.2f}% of portfolio")
    st.markdown(
        f"Zone: **${ps_short_K:.0f} – ${put_K:.0f}**  \n"
        f"Max protection: **${(put_K - ps_short_K) * contracts_needed * 100:,.0f}**"
    )

with c3:
    st.markdown("#### 🔵 Collar")
    st.markdown(f"Buy **${put_K:.0f}p**, Sell **${call_K:.0f}c**")
    col_lbl = "Net Credit" if collar_net < 0 else "Net Debit"
    st.metric(col_lbl, f"${abs(collar_net):.2f}/contract")
    st.metric("Total Cost", f"${abs(collar_total):,.0f}",
              "credit received" if collar_net < 0 else "debit paid")
    st.metric("Annualized", f"${abs(collar_annual):,.0f}",
              f"{abs(collar_annual) / portfolio_value * 100:.2f}% of portfolio")
    st.markdown(
        f"Floor: **${put_K:.0f}**, Upside capped: **${call_K:.0f}**  \n"
        f"{'Slight income' if collar_net < 0 else 'Small net cost'}"
    )

with c4:
    st.markdown("#### 🟣 Put Ratio Spread")
    st.markdown(f"Buy **1×** ${put_K:.0f}p, Sell **2×** ${ps_short_K:.0f}p")
    r_lbl = "Net Credit" if ratio_net < 0 else "Net Debit"
    st.metric(r_lbl, f"${abs(ratio_net):.2f}/contract")
    st.metric("Total Cost", f"${abs(ratio_total):,.0f}",
              "credit" if ratio_net < 0 else "debit")
    st.metric("Annualized", f"${abs(ratio_annual):,.0f}",
              f"{abs(ratio_annual) / portfolio_value * 100:.2f}% of portfolio")
    st.markdown(
        f"Max gain: **${ratio_max_gain:,.0f}** at ${ps_short_K:.0f}  \n"
        f"⚠️ Extra short exposure below **${ps_short_K:.0f}**"
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — INDIVIDUAL POSITION HEDGES
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📌 Individual Position Hedges")

if not hedge_tickers:
    st.info("Load a portfolio on **Home** to enable per-position hedges.")
else:
    hedge_sel = st.multiselect(
        "Select positions to hedge individually",
        options=hedge_tickers,
        default=hedge_tickers[:min(3, len(hedge_tickers))],
    )

    if hedge_sel:
        col_iexp, col_iput, col_icall = st.columns(3)
        with col_iexp:
            ind_exp_str = st.selectbox(
                "Expiration for individual hedges", exp_options,
                index=min(1, len(exp_options) - 1), key="ind_exp",
            )
            ind_exp = ind_exp_str.split(" ")[0]
            ind_dte = int(ind_exp_str.split("(")[1].replace("d)", ""))
            T_ind   = max(ind_dte / 365, 0.003)
        with col_iput:
            ind_put_pct  = st.slider("Put Strike (% below spot)", 2.0, 20.0, 5.0, 0.5, key="ind_put")
        with col_icall:
            ind_call_pct = st.slider("Call Strike (% above spot)", 2.0, 20.0, 5.0, 0.5, key="ind_call")

        for ticker in hedge_sel:
            with st.expander(f"📍 {ticker}", expanded=True):
                # Position size from pos_df (already built above)
                match = pos_df[pos_df["Ticker"] == ticker]
                if not match.empty:
                    row_        = match.iloc[0]
                    pos_shares  = int(row_["Shares"])
                    spot        = float(row_["Price"])
                    pos_val     = float(row_["Value"])
                else:
                    pos_shares  = 100
                    spot        = _spot(ticker)
                    pos_val     = pos_shares * spot

                contracts_t = max(1, int(np.ceil(pos_shares / 100)))

                # Fetch option chain — try requested expiry, fall back to nearest available
                calls_t, puts_t, spot_live = _stock_chain(ticker, ind_exp)

                if puts_t.empty:
                    st_exps = _stock_expirations(ticker)
                    near_exps = [
                        e for e in st_exps
                        if (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days >= 7
                    ]
                    if near_exps:
                        alt_exp = min(
                            near_exps,
                            key=lambda e: abs(
                                (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days - ind_dte
                            ),
                        )
                        calls_t, puts_t, spot_live = _stock_chain(ticker, alt_exp)
                        if not puts_t.empty:
                            st.caption(f"ℹ️ Nearest available expiry used: {alt_exp}")

                if spot_live > 0:
                    spot = spot_live

                # Compute strikes and premiums
                if not puts_t.empty and not calls_t.empty:
                    put_K_t,  put_prem_t  = _nearest_put(puts_t,  spot * (1 - ind_put_pct  / 100))
                    call_K_t, call_prem_t = _nearest_call(calls_t, spot * (1 + ind_call_pct / 100))
                else:
                    # Black-Scholes fallback with a default IV of 30%
                    st.caption("⚠️ No live chain — using Black-Scholes estimate (σ = 30%)")
                    stk_iv    = 0.30
                    put_K_t   = round(spot * (1 - ind_put_pct  / 100), 0)
                    call_K_t  = round(spot * (1 + ind_call_pct / 100), 0)
                    put_prem_t  = max(options_calc.black_scholes(spot, put_K_t,  T_ind, stk_iv, "put"),  0.01)
                    call_prem_t = max(options_calc.black_scholes(spot, call_K_t, T_ind, stk_iv, "call"), 0.01)

                put_cost_t     = put_prem_t * contracts_t * 100
                collar_net_t   = put_prem_t - call_prem_t
                collar_cost_t  = collar_net_t * contracts_t * 100
                cov_income_t   = call_prem_t * contracts_t * 100

                st.markdown(
                    f"**{pos_shares} shares @ ${spot:.2f}** — "
                    f"${pos_val:,.0f} position value — "
                    f"{contracts_t} contract{'s' if contracts_t > 1 else ''}"
                )

                h1, h2, h3 = st.columns(3)

                with h1:
                    st.markdown("**🔴 Protective Put**")
                    st.write(f"Buy {contracts_t}× ${put_K_t:.0f} put")
                    st.metric("Premium", f"${put_prem_t:.2f}/contract")
                    st.metric(
                        "Total Cost", f"${put_cost_t:,.0f}",
                        f"{put_cost_t / pos_val * 100:.2f}% of position" if pos_val > 0 else "",
                    )
                    st.write(f"Protected below **${put_K_t:.0f}**")

                with h2:
                    st.markdown("**🔵 Collar**")
                    st.write(f"Buy ${put_K_t:.0f}p / Sell ${call_K_t:.0f}c")
                    c_lbl = "Net Credit" if collar_net_t < 0 else "Net Debit"
                    st.metric(c_lbl, f"${abs(collar_net_t):.2f}/contract")
                    st.metric(
                        "Total Cost", f"${abs(collar_cost_t):,.0f}",
                        "credit" if collar_net_t < 0 else "debit",
                    )
                    st.write(f"Floor: **${put_K_t:.0f}**, Cap: **${call_K_t:.0f}**")

                with h3:
                    st.markdown("**🟢 Covered Call**")
                    st.write(f"Sell {contracts_t}× ${call_K_t:.0f} call")
                    st.metric("Premium Received", f"${call_prem_t:.2f}/contract")
                    st.metric(
                        "Total Income", f"${cov_income_t:,.0f}",
                        f"{cov_income_t / pos_val * 100:.2f}% of position" if pos_val > 0 else "",
                    )
                    st.write(f"Upside capped at **${call_K_t:.0f}** — offsets put cost")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Scenario Analysis — Portfolio P&L vs SPY Move")
st.markdown(
    "Shows how your portfolio P&L changes across market moves "
    "with and without each SPY hedge strategy applied."
)

moves        = np.linspace(-0.30, 0.30, 61)
unhedged_pnl = portfolio_value * portfolio_beta * moves

# Protective Put
put_payoff   = np.array([max(put_K - spy_price * (1 + m), 0) for m in moves])
pp_pnl       = (put_payoff - put_prem) * contracts_needed * 100
pp_total_pnl = unhedged_pnl + pp_pnl

# Put Spread
ps_long_pay  = np.array([max(put_K        - spy_price * (1 + m), 0) for m in moves])
ps_short_pay = np.array([max(ps_short_K   - spy_price * (1 + m), 0) for m in moves])
ps_pnl       = (ps_long_pay - ps_short_pay - ps_net) * contracts_needed * 100
ps_total_pnl = unhedged_pnl + ps_pnl

# Collar
cl_put_pay   = np.array([max(put_K - spy_price * (1 + m), 0) for m in moves])
cl_call_pay  = np.array([max(spy_price * (1 + m) - call_K, 0) for m in moves])
cl_pnl       = (cl_put_pay - cl_call_pay - collar_net) * contracts_needed * 100
cl_total_pnl = unhedged_pnl + cl_pnl

# Put Ratio Spread
rs_long_pay  = np.array([max(put_K      - spy_price * (1 + m), 0) for m in moves])
rs_short_pay = np.array([max(ps_short_K - spy_price * (1 + m), 0) for m in moves])
rs_pnl       = (rs_long_pay - 2 * rs_short_pay - ratio_net) * contracts_needed * 100
rs_total_pnl = unhedged_pnl + rs_pnl

move_pcts = moves * 100

fig_scen = go.Figure()
fig_scen.add_hrect(y0=-portfolio_value * 0.4, y1=0,
                   fillcolor="rgba(251,113,133,0.04)", line_width=0)

fig_scen.add_trace(go.Scatter(
    x=move_pcts, y=unhedged_pnl, name="Unhedged", mode="lines",
    line=dict(color="#888888", width=2, dash="dot"),
))
fig_scen.add_trace(go.Scatter(
    x=move_pcts, y=pp_total_pnl,
    name=f"Protective Put (${put_K:.0f})", mode="lines",
    line=dict(color="#fb7185", width=2.5),
))
fig_scen.add_trace(go.Scatter(
    x=move_pcts, y=ps_total_pnl,
    name=f"Put Spread ({put_K:.0f}/{ps_short_K:.0f})", mode="lines",
    line=dict(color="#f59e0b", width=2.5),
))
fig_scen.add_trace(go.Scatter(
    x=move_pcts, y=cl_total_pnl,
    name=f"Collar ({put_K:.0f}p/{call_K:.0f}c)", mode="lines",
    line=dict(color="#22d3ee", width=2.5),
))
fig_scen.add_trace(go.Scatter(
    x=move_pcts, y=rs_total_pnl,
    name=f"Ratio Spread ({put_K:.0f}/2×{ps_short_K:.0f})", mode="lines",
    line=dict(color="#a78bfa", width=2.5),
))

fig_scen.add_hline(y=0, line_color="#ffffff", line_width=1, line_dash="dash")
fig_scen.add_vline(x=0, line_color="#888888", line_width=1, line_dash="dot")

fig_scen.update_layout(**carbon_plotly_layout(
    height=520,
    title="Portfolio P&L vs SPY Move — All Hedge Strategies",
    xaxis_title="SPY Move (%)",
    yaxis_title="Portfolio P&L ($)",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.15),
))
fig_scen.update_xaxes(tickformat="+.0f", ticksuffix="%", gridcolor="rgba(107,122,143,0.15)")
fig_scen.update_yaxes(tickprefix="$", tickformat=",.0f",  gridcolor="rgba(107,122,143,0.15)")
st.plotly_chart(fig_scen, use_container_width=True)

# ── Scenario table ─────────────────────────────────────────────────────────────
st.markdown("### Scenario Table")

key_moves = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]
key_idx   = [int(np.argmin(np.abs(moves - m))) for m in key_moves]

scen_rows = []
for i in key_idx:
    m = moves[i]
    scen_rows.append({
        "Market Move":    f"{m:+.0%}",
        "SPY Target":     f"${spy_price * (1 + m):.0f}",
        "Unhedged P&L":   unhedged_pnl[i],
        "Protective Put": pp_total_pnl[i],
        "Put Spread":     ps_total_pnl[i],
        "Collar":         cl_total_pnl[i],
        "Ratio Spread":   rs_total_pnl[i],
    })

scen_df  = pd.DataFrame(scen_rows)
pnl_cols = ["Unhedged P&L", "Protective Put", "Put Spread", "Collar", "Ratio Spread"]


def _color_pnl(val):
    try:
        v = float(str(val).replace("$", "").replace(",", ""))
        return f'color: {"#22d3ee" if v >= 0 else "#fb7185"}'
    except Exception:
        return ""


styled = (
    scen_df.style
    .format({c: "${:+,.0f}" for c in pnl_cols})
    .applymap(_color_pnl, subset=pnl_cols)
)
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Effectiveness summary ──────────────────────────────────────────────────────
st.subheader("📈 Hedge Effectiveness — −20% Scenario")

wi = int(np.argmin(np.abs(moves - (-0.20))))
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Unhedged",       f"${unhedged_pnl[wi]:+,.0f}", f"{unhedged_pnl[wi] / portfolio_value * 100:+.1f}%")
c2.metric("Protective Put", f"${pp_total_pnl[wi]:+,.0f}", f"Saved ${pp_total_pnl[wi] - unhedged_pnl[wi]:+,.0f}")
c3.metric("Put Spread",     f"${ps_total_pnl[wi]:+,.0f}", f"Saved ${ps_total_pnl[wi] - unhedged_pnl[wi]:+,.0f}")
c4.metric("Collar",         f"${cl_total_pnl[wi]:+,.0f}", f"Saved ${cl_total_pnl[wi] - unhedged_pnl[wi]:+,.0f}")
c5.metric("Ratio Spread",   f"${rs_total_pnl[wi]:+,.0f}", f"Saved ${rs_total_pnl[wi] - unhedged_pnl[wi]:+,.0f}")

# ── Regime-based recommendation ────────────────────────────────────────────────
with st.expander("💡 Regime-Based Hedge Recommendation"):
    _RECS = {
        "High Vol": {
            "rec":    "Protective Put",
            "reason": "Volatility is elevated and tail risk is real — full put coverage justified.",
            "sizing": "100% of calculated contracts",
        },
        "Trending": {
            "rec":    "Put Spread",
            "reason": "Strong directional trend lowers crash risk. A spread gives affordable protection in the most likely drawdown zone.",
            "sizing": "75–100% of contracts",
        },
        "Low Vol": {
            "rec":    "Collar (light)",
            "reason": "Calm environment — a collar costs almost nothing and provides a safety net without meaningful upside drag.",
            "sizing": "50–75% of contracts",
        },
        "Mean Reversion": {
            "rec":    "Put Spread",
            "reason": "Choppy markets can produce sudden drops. A put spread balances cost and protection in range-bound conditions.",
            "sizing": "75% of contracts",
        },
        "Uncertain": {
            "rec":    "Put Ratio Spread",
            "reason": "Ambiguous regime — a ratio spread can be near zero-cost while still providing meaningful protection in the most likely drawdown range.",
            "sizing": "75% of contracts",
        },
        "Unknown": {
            "rec":    "Protective Put (reduced size)",
            "reason": "Regime signal is insufficient — maintain baseline protection at reduced notional.",
            "sizing": "50% of contracts",
        },
    }
    rec = _RECS.get(current_regime, _RECS["Unknown"])

    st.markdown(f"""
**Regime:** {current_regime}

**Recommended Strategy:** {rec['rec']}

**Why:** {rec['reason']}

**Suggested Sizing:** {rec['sizing']} ({contracts_needed} contracts at full size)

---

**Cost comparison for {exp_dte}-day horizon ({selected_exp}):**

| Strategy | Cost | Annualized | Portfolio % |
|---|---|---|---|
| Protective Put   | ${put_total:,.0f}         | ${put_annual:,.0f}         | {put_annual / portfolio_value * 100:.2f}% |
| Put Spread       | ${ps_total:,.0f}          | ${ps_annual:,.0f}          | {ps_annual / portfolio_value * 100:.2f}% |
| Collar           | ${abs(collar_total):,.0f} | ${abs(collar_annual):,.0f} | {abs(collar_annual) / portfolio_value * 100:.2f}% |
| Put Ratio Spread | ${abs(ratio_total):,.0f}  | ${abs(ratio_annual):,.0f}  | {abs(ratio_annual) / portfolio_value * 100:.2f}% |
""")
