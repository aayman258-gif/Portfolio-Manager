"""
Regime-Aware Portfolio Manager
Section 6: Options Analytics & Recommendations
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))

from utils.carbon_theme import (
    ACCENT, AMBER, BG, BORDER, CARD, DIM, FG, GAIN, LOSS, SUBTLE,
    apply_carbon_theme, carbon_plotly_layout, page_header,
)
from calculations.options_analytics import OptionsAnalytics
from calculations.options_recommender import OptionsRecommender
from calculations.regime_detector import RegimeDetector
from data.market_data import MarketDataLoader

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Options Analytics", page_icon="📊", layout="wide")
apply_carbon_theme()
page_header(
    "📊 Options Analytics & Trading",
    "Analyze positions with live Greeks · regime-aware recommendations · portfolio integration",
)

# ── Init ──────────────────────────────────────────────────────────────────────

options_calc = OptionsAnalytics(risk_free_rate=0.045)
recommender  = OptionsRecommender()
detector     = RegimeDetector()
market_loader = MarketDataLoader()

_RF = 0.045


@st.cache_data(ttl=1800, show_spinner=False)
def _get_regime():
    spy_data = market_loader.load_index_data("SPY", "2y")
    vix_data = market_loader.load_vix_data("2y")
    spy_prices, vix_prices = market_loader.align_data(spy_data, vix_data)
    regime, signals = detector.classify_regime(spy_prices, vix_prices)
    return str(regime.iloc[-1]), float(signals["realized_vol"].iloc[-1])


current_regime, market_volatility = _get_regime()

# ── Shared helpers ────────────────────────────────────────────────────────────

def _dte_years(expiration: str) -> float:
    """Convert expiration string (YYYY-MM-DD) to time-in-years (min 0.001)."""
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        days = (exp_date - datetime.now().date()).days
        return max(days / 365.0, 0.001)
    except Exception:
        return 0.001


@st.cache_data(ttl=300, show_spinner=False)
def _live_option_row(underlying: str, option_type: str, strike: float, expiration: str):
    """
    Fetch the live option row from yfinance for (underlying, type, strike, expiration).
    Returns (mid_price, implied_vol) or (None, None).
    """
    try:
        chain = yf.Ticker(underlying).option_chain(expiration)
        df = chain.calls if option_type == "call" else chain.puts
        if df.empty:
            return None, None
        df = df.copy()
        df["_dist"] = (df["strike"] - strike).abs()
        row = df.loc[df["_dist"].idxmin()]
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        mid = (bid + ask) / 2 if (bid + ask) > 0 else float(row.get("lastPrice", 0) or 0)
        iv  = float(row.get("impliedVolatility", 0) or 0)
        return (mid if mid > 0 else None), (iv if iv > 0 else None)
    except Exception:
        return None, None


@st.cache_data(ttl=60, show_spinner=False)
def _spot_price(ticker: str) -> Optional[float]:
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        return float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception:
        return None


def _pnl_color(v: float) -> str:
    return GAIN if v >= 0 else LOSS


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "📈 Recommendations",
    "🔢 Analyze Position",
    "📊 Portfolio Greeks",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDATIONS (portfolio-aware)
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader(f"🎯 Regime-Aware Strategies — {current_regime}")
    st.info(f"**Regime:** {current_regime}  ·  **Realized Vol:** {market_volatility*100:.1f}%")

    # ── Portfolio context ─────────────────────────────────────────────────────
    _positions_df   = st.session_state.get("positions")
    _opts           = st.session_state.get("options_positions", [])
    _current_prices = st.session_state.get("current_prices", {})
    _total_value    = st.session_state.get("total_value")

    _equity_delta = 0.0
    _equity_tickers: list[str] = []
    if isinstance(_positions_df, pd.DataFrame) and not _positions_df.empty:
        for _, _row in _positions_df.iterrows():
            _equity_tickers.append(str(_row["ticker"]))
            _equity_delta += float(_row["shares"])   # shares ≈ delta units

    # Quick net options Greeks from stored iv_at_entry (fast path for context bar)
    _opt_net = {"delta": 0.0, "theta": 0.0, "vega": 0.0}
    for _p in _opts:
        if _p.get("status") != "open":
            continue
        if _p.get("type") == "strategy":
            for _lg in _p.get("legs", []):
                _S0 = _spot_price(_p.get("underlying", "")) or 100.0
                _T  = _dte_years(_lg.get("expiration", ""))
                _iv = float(_lg.get("iv_at_entry", 0.3) or 0.3)
                _g  = options_calc.calculate_greeks(_S0, _lg["strike"], _T, _iv, _lg["option_type"])
                _sign = 1 if _lg["position"] == "long" else -1
                _ct   = int(_lg.get("contracts", 1))
                for _k in ("delta", "theta", "vega"):
                    _opt_net[_k] += _g[_k] * _sign * _ct * 100
        else:
            _S0 = _spot_price(_p.get("underlying", "")) or 100.0
            _T  = _dte_years(_p.get("expiration", ""))
            _iv = float(_p.get("iv_at_entry", 0.3) or 0.3)
            _g  = options_calc.calculate_greeks(_S0, _p["strike"], _T, _iv, _p["option_type"])
            _ct = int(_p.get("contracts", 1))
            for _k in ("delta", "theta", "vega"):
                _opt_net[_k] += _g[_k] * _ct * 100

    _net_delta = _equity_delta + _opt_net["delta"]

    if _equity_tickers or _opts:
        with st.expander("📋 Your Portfolio Context (drives recommendations)", expanded=True):
            _pc1, _pc2, _pc3, _pc4 = st.columns(4)
            with _pc1:
                st.metric("Equity Δ Exposure", f"{_equity_delta:+,.0f} shares",
                          help="Sum of all share positions — each share ≈ +1 delta.")
            with _pc2:
                st.metric("Options Net Δ", f"{_opt_net['delta']:+,.1f}",
                          help="Net delta from all open options positions.")
            with _pc3:
                st.metric("Options Net Θ", f"${_opt_net['theta']:+,.2f}/day",
                          help="Daily time decay across all open options.")
            with _pc4:
                st.metric("Options Net Vega", f"${_opt_net['vega']:+,.2f}",
                          help="Sensitivity to 1% IV move across all open options.")

            # Contextual nudge
            if _equity_delta > 500 and _opt_net["delta"] >= 0:
                st.warning(
                    "**Long-heavy portfolio** — consider protective puts or collars on your "
                    "largest equity positions to reduce downside risk."
                )
            elif _opt_net["theta"] < -50:
                st.warning(
                    "**High negative theta** — your long options are decaying significantly. "
                    "Consider closing or rolling near-expiry positions."
                )
            elif _opt_net["vega"] > 500:
                st.info(
                    "**Long vega** — your portfolio benefits if IV rises. "
                    "In a High Vol regime, consider taking some vega off the table."
                )

            if _equity_tickers:
                st.caption(f"Equity holdings: {', '.join(_equity_tickers[:10])}" +
                           ("…" if len(_equity_tickers) > 10 else ""))

    # ── Strategy recommendations ──────────────────────────────────────────────
    strategies = recommender.get_strategies_for_regime(current_regime)

    for idx, strategy in enumerate(strategies):
        with st.expander(f"**{strategy['name']}** — {strategy['type']}", expanded=(idx == 0)):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Description:** {strategy['description']}")
                st.markdown(f"**Structure:** {strategy['structure']}")
                st.markdown(f"**Best When:** {strategy['best_when']}")
            with col2:
                st.metric("Risk Level", strategy["risk"])
                st.metric("Complexity", strategy["complexity"])

            st.markdown("### 📍 Strike Recommendations")

            # Pre-fill with an owned ticker if relevant
            _default_ul = _equity_tickers[0] if _equity_tickers else ""
            _ul_hint    = f" (e.g. {_default_ul})" if _default_ul else ""

            underlying_price = st.number_input(
                f"Underlying Price{_ul_hint} ({strategy['name']})",
                min_value=1.0, value=450.0, step=1.0, key=f"price_{idx}",
            )

            col_a, col_b = st.columns(2)
            with col_a:
                days_to_expiry = st.slider(
                    "Days to Expiration", 7, 90, 30, key=f"dte_{idx}",
                )
            with col_b:
                trend = st.selectbox(
                    "Market Trend",
                    options=[0, 1, -1],
                    format_func=lambda x: "Sideways" if x == 0 else "Uptrend" if x == 1 else "Downtrend",
                    key=f"trend_{idx}",
                )

            strike_recs = recommender.calculate_strike_recommendations(
                current_price=underlying_price,
                regime=current_regime,
                volatility=market_volatility,
                trend_direction=trend,
                days_to_expiry=days_to_expiry,
            )

            if strategy["name"] in strike_recs:
                strikes = strike_recs[strategy["name"]]
                st.success(f"**Rationale:** {strikes.get('rationale', 'N/A')}")
                strike_data = {k: v for k, v in strikes.items() if k != "rationale"}
                if strike_data:
                    st.dataframe(pd.DataFrame([strike_data]), use_container_width=True, hide_index=True)

    # ── Position sizing (auto account size) ──────────────────────────────────
    st.subheader("💰 Position Sizing Guide")

    _auto_acct = float(_total_value) if _total_value else 50_000.0
    account_size = st.number_input(
        "Account Size ($)",
        min_value=1_000, max_value=10_000_000,
        value=int(_auto_acct), step=1_000,
        help="Auto-populated from your loaded portfolio total value.",
    )
    if _total_value:
        st.caption(f"Auto-read from portfolio: ${_total_value:,.0f}")

    sizing = recommender.get_position_sizing_guide(current_regime, account_size)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Max Risk Per Trade", f"${sizing['max_dollar_risk']:,.0f}",
                  f"{sizing['max_risk_per_trade']*100:.1f}%")
    with c2:
        st.metric("Max Positions", sizing["max_positions"])
    with c3:
        _tot_risk = sizing["max_dollar_risk"] * sizing["max_positions"]
        st.metric("Max Total Risk", f"${_tot_risk:,.0f}",
                  f"{_tot_risk/account_size*100:.1f}%")
    st.info(f"**Note:** {sizing['notes']}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYZE POSITION (with live-chain toggle)
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("🔍 Analyze Individual Option Position")

    # ── Input mode toggle ─────────────────────────────────────────────────────
    _input_mode = st.radio(
        "Input mode",
        ["Manual Input", "From Live Chain"],
        horizontal=True,
        key="_a2_mode",
    )

    # ── Pre-fill state (populated by live chain selector) ─────────────────────
    _pre = st.session_state.get("_a2_prefill", {})

    # ── LIVE CHAIN SELECTOR ───────────────────────────────────────────────────
    if _input_mode == "From Live Chain":
        st.markdown("##### Select from live options chain")
        _lc1, _lc2, _lc3 = st.columns([2, 2, 1])

        with _lc1:
            _chain_ticker = st.text_input(
                "Ticker", value=st.session_state.get("_a2_chain_ticker", ""),
                placeholder="e.g. AAPL", key="_a2_chain_ticker_input",
            ).upper().strip()
        with _lc2:
            _chain_type = st.radio(
                "Type", ["call", "put"], horizontal=True, key="_a2_chain_type",
            )
        with _lc3:
            _load_chain = st.button("🔄 Load Chain", key="_a2_load_chain")

        if _load_chain and _chain_ticker:
            with st.spinner(f"Fetching {_chain_ticker} options chain…"):
                try:
                    _tk = yf.Ticker(_chain_ticker)
                    _exps = list(_tk.options)
                    _hist = _tk.history(period="1d")
                    _S_live = float(_hist["Close"].iloc[-1]) if not _hist.empty else None
                    if _exps and _S_live:
                        st.session_state["_a2_chain_ticker"] = _chain_ticker
                        st.session_state["_a2_chain_exps"]   = _exps
                        st.session_state["_a2_chain_spot"]   = _S_live
                        st.session_state["_a2_chain_loaded"] = True
                    else:
                        st.error(f"No options data found for {_chain_ticker}.")
                        st.session_state.pop("_a2_chain_loaded", None)
                except Exception as _e:
                    st.error(f"Error loading chain: {_e}")
                    st.session_state.pop("_a2_chain_loaded", None)

        if st.session_state.get("_a2_chain_loaded"):
            _chain_ticker_loaded = st.session_state["_a2_chain_ticker"]
            _exps_avail = st.session_state["_a2_chain_exps"]
            _S_live     = st.session_state["_a2_chain_spot"]

            st.caption(f"**{_chain_ticker_loaded}** — spot ${_S_live:.2f} · "
                       f"{len(_exps_avail)} expirations available")

            _sel_exp = st.selectbox(
                "Expiration", _exps_avail, key="_a2_sel_exp",
            )

            # Load strikes for selected expiry + type
            with st.spinner("Loading strikes…"):
                try:
                    _chain_data = yf.Ticker(_chain_ticker_loaded).option_chain(_sel_exp)
                    _chain_df   = (_chain_data.calls if _chain_type == "call"
                                   else _chain_data.puts).copy()
                    _chain_df   = _chain_df[_chain_df["strike"] > 0].reset_index(drop=True)
                except Exception as _e:
                    st.error(f"Could not load strikes: {_e}")
                    _chain_df = pd.DataFrame()

            if not _chain_df.empty:
                # Format strike labels
                def _strike_label(i: int, df=_chain_df) -> str:
                    r   = df.iloc[i]
                    bid = float(r.get("bid", 0) or 0)
                    ask = float(r.get("ask", 0) or 0)
                    mid = (bid + ask) / 2 if (bid + ask) > 0 else float(r.get("lastPrice", 0) or 0)
                    iv  = float(r.get("impliedVolatility", 0) or 0) * 100
                    oi  = int(r.get("openInterest", 0) or 0)
                    return f"${r['strike']:.2f}  mid ${mid:.2f}  IV {iv:.0f}%  OI {oi:,}"

                # Default to nearest ATM
                _atm_idx = int((_chain_df["strike"] - _S_live).abs().argsort().iloc[0])
                _sel_strike_idx = st.selectbox(
                    "Strike", range(len(_chain_df)),
                    index=_atm_idx,
                    format_func=_strike_label,
                    key="_a2_sel_strike_idx",
                )
                _sel_row = _chain_df.iloc[_sel_strike_idx]
                _sel_strike = float(_sel_row["strike"])
                _sel_bid    = float(_sel_row.get("bid", 0) or 0)
                _sel_ask    = float(_sel_row.get("ask", 0) or 0)
                _sel_mid    = (_sel_bid + _sel_ask) / 2 if (_sel_bid + _sel_ask) > 0 else float(_sel_row.get("lastPrice", 0) or 0)
                _sel_iv     = float(_sel_row.get("impliedVolatility", 0) or 0.3)
                _sel_oi     = int(_sel_row.get("openInterest", 0) or 0)
                _sel_vol    = int(_sel_row.get("volume", 0) or 0)

                st.info(
                    f"Selected: **{_chain_ticker_loaded}** {_chain_type.upper()} "
                    f"${_sel_strike:.2f} exp {_sel_exp}  ·  "
                    f"Mid **${_sel_mid:.2f}**  ·  IV **{_sel_iv*100:.1f}%**  ·  "
                    f"OI {_sel_oi:,}  ·  Vol {_sel_vol:,}"
                )

                if st.button("⬇ Populate Analysis Form", type="primary", key="_a2_populate"):
                    st.session_state["_a2_prefill"] = {
                        "S":    _S_live,
                        "K":    _sel_strike,
                        "type": _chain_type,
                        "iv":   _sel_iv * 100,
                        "exp":  _sel_exp,
                        "mid":  _sel_mid,
                    }
                    _pre = st.session_state["_a2_prefill"]
                    st.success("Form populated — see inputs below.")

        st.divider()

    # ── Analysis form ─────────────────────────────────────────────────────────
    st.markdown("##### Position Details")
    col1, col2 = st.columns(2)

    with col1:
        underlying_price = st.number_input(
            "Current Underlying Price ($)",
            min_value=1.0,
            value=float(_pre.get("S", 450.0)),
            step=0.50,
        )
        strike_price = st.number_input(
            "Strike Price ($)",
            min_value=1.0,
            value=float(_pre.get("K", 455.0)),
            step=1.0,
        )
        option_type = st.selectbox(
            "Option Type",
            options=["call", "put"],
            index=0 if _pre.get("type", "call") == "call" else 1,
        )
        contracts = st.number_input(
            "Contracts (negative for short)",
            min_value=-1000, max_value=1000, value=1, step=1,
        )

    with col2:
        st.markdown("##### Time & Volatility")

        _pre_exp_date = None
        if _pre.get("exp"):
            try:
                _pre_exp_date = datetime.strptime(_pre["exp"], "%Y-%m-%d").date()
            except Exception:
                pass

        expiration_date = st.date_input(
            "Expiration Date",
            value=_pre_exp_date or (datetime.now() + timedelta(days=30)).date(),
        )
        days_to_exp  = (expiration_date - datetime.now().date()).days
        time_to_exp  = max(days_to_exp / 365, 0.001)
        st.metric("Days to Expiration", days_to_exp)

        implied_vol = st.slider(
            "Implied Volatility (%)",
            min_value=5.0, max_value=150.0,
            value=float(_pre.get("iv", 30.0)),
            step=1.0,
        ) / 100

        risk_free = st.slider(
            "Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.5, step=0.1,
        ) / 100

        dividend_yield = st.slider(
            "Dividend Yield (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1,
        ) / 100

    st.markdown("##### Optional: P&L Analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        premium_paid = st.number_input(
            "Premium Paid per Contract ($/share)",
            min_value=0.0, value=0.0, step=0.10,
        )
    with col_b:
        current_market_price = st.number_input(
            "Current Market Price ($/share)",
            min_value=0.0,
            value=float(_pre.get("mid", 0.0)),
            step=0.10,
        )

    premium_paid         = premium_paid         if premium_paid         > 0 else None
    current_market_price = current_market_price if current_market_price > 0 else None

    # ── Run analysis ──────────────────────────────────────────────────────────
    if st.button("🔍 Analyze Position", type="primary"):
        options_calc.risk_free_rate = risk_free

        analysis = options_calc.analyze_option_position(
            S=underlying_price,
            K=strike_price,
            T=time_to_exp,
            sigma=implied_vol,
            option_type=option_type,
            contracts=contracts,
            premium_paid=premium_paid,
            current_price=current_market_price,
            q=dividend_yield,
        )

        st.success("✅ Analysis complete")

        # Key metrics
        st.subheader("📊 Key Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Theoretical Value", f"${analysis['theoretical_value']:.2f}")
        with c2: st.metric("Intrinsic Value",   f"${analysis['intrinsic_value']:.2f}")
        with c3: st.metric("Time Value",        f"${analysis['time_value']:.2f}")
        with c4: st.metric("Moneyness",          analysis["moneyness"])

        # P&L
        if analysis["pnl"] is not None:
            st.subheader("💰 Profit & Loss")
            _pcol1, _pcol2 = st.columns(2)
            with _pcol1:
                st.markdown(
                    f"<h2 style='color:{_pnl_color(analysis['pnl'])};'>"
                    f"${analysis['pnl']:,.2f}</h2>",
                    unsafe_allow_html=True,
                )
                st.caption(f"P&L: {analysis['pnl_pct']:+.2f}%")
            with _pcol2:
                _pos_val = abs(contracts) * current_market_price * 100 if current_market_price else 0
                st.metric("Position Value", f"${_pos_val:,.0f}")

        # Fair value
        if analysis["fair_value_analysis"]:
            st.subheader("⚖️ Fair Value Analysis")
            fv = analysis["fair_value_analysis"]
            _fc1, _fc2, _fc3 = st.columns(3)
            with _fc1: st.metric("Market Price",       f"${fv['market_price']:.2f}")
            with _fc2: st.metric("Theoretical Value",  f"${fv['theoretical_value']:.2f}")
            with _fc3:
                st.metric("Difference", f"${fv['difference']:.2f}", f"{fv['difference_pct']:+.2f}%",
                          delta_color="inverse" if fv["difference"] > 0 else "normal")
            (st.warning if fv["rating"] in ("Overpriced", "Bad Short") else
             st.success if fv["rating"] in ("Underpriced", "Good Short") else st.info)(
                f"**Rating:** {fv['rating']}"
            )

        # Greeks
        st.subheader("🔢 The Greeks")
        greeks   = analysis["greeks"]
        pos_g    = analysis["position_greeks"]

        st.markdown("**Per Contract**")
        _gc = st.columns(5)
        for _col, _name, _fmt in zip(
            _gc,
            ["delta", "gamma", "theta", "vega", "rho"],
            [".4f", ".4f", "$.2f", "$.2f", "$.2f"],
        ):
            with _col:
                _v = greeks[_name]
                st.metric(_name.capitalize(), f"{_v:{_fmt.lstrip('$')}}" if "$" not in _fmt else f"${_v:.2f}")
                st.caption(options_calc.get_greek_interpretation(_name, _v)["description"])

        st.markdown("**Total Position**")
        _gp = st.columns(5)
        _pos_items = [
            ("Position Δ", pos_g["delta"],  "{:+.2f}"),
            ("Position Γ", pos_g["gamma"],  "{:+.4f}"),
            ("Daily Θ",    pos_g["theta"],  "${:+.2f}"),
            ("Position V", pos_g["vega"],   "${:+.2f}"),
            ("Position ρ", pos_g["rho"],    "${:+.2f}"),
        ]
        for _col, (_lbl, _val, _fmt) in zip(_gp, _pos_items):
            with _col:
                _disp = _fmt.format(_val) if "$" not in _fmt else f"${_val:+.2f}"
                _color = _pnl_color(_val)
                st.markdown(
                    f"<div style='font-size:1.4rem;font-weight:300;color:{_color};'>{_disp}</div>"
                    f"<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:.13em;"
                    f"color:{DIM};'>{_lbl}</div>",
                    unsafe_allow_html=True,
                )

        with st.expander("📚 Understanding Your Greeks"):
            for _gn in ["delta", "gamma", "theta", "vega", "rho"]:
                st.markdown(f"**{_gn.capitalize()}**")
                st.write(options_calc.get_greek_interpretation(_gn, pos_g[_gn])["interpretation"])

        # Sensitivity chart
        st.subheader("📈 Price Sensitivity")
        _pr = np.linspace(underlying_price * 0.80, underlying_price * 1.20, 60)
        _pv, _pd = [], []
        for _p in _pr:
            _pv.append(options_calc.black_scholes(_p, strike_price, time_to_exp, implied_vol,
                                                   option_type, risk_free, dividend_yield) * contracts * 100)
            _pd.append(options_calc.calculate_greeks(_p, strike_price, time_to_exp, implied_vol,
                                                      option_type, risk_free, dividend_yield)["delta"])

        fig_sens = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Position Value vs Underlying", "Delta vs Underlying"),
            vertical_spacing=0.1,
        )
        fig_sens.add_trace(go.Scatter(x=_pr, y=_pv, name="Position Value",
                                       line=dict(color=ACCENT, width=2)), row=1, col=1)
        fig_sens.add_trace(go.Scatter(x=_pr, y=_pd, name="Delta",
                                       line=dict(color=AMBER, width=2)), row=2, col=1)
        for _row in (1, 2):
            fig_sens.add_vline(x=underlying_price, line_dash="dash",
                                line_color=SUBTLE, row=_row, col=1)
            fig_sens.add_vline(x=strike_price, line_dash="dot",
                                line_color=LOSS, row=_row, col=1)
        fig_sens.update_layout(**carbon_plotly_layout(height=560, showlegend=False))
        fig_sens.update_xaxes(title_text="Underlying Price ($)", row=2, col=1)
        fig_sens.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig_sens.update_yaxes(title_text="Delta", row=2, col=1)
        st.plotly_chart(fig_sens, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO GREEKS (live, from session state positions)
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("📊 Portfolio-Level Greeks")

    _opts_all = st.session_state.get("options_positions", [])
    _open_opts = [p for p in _opts_all if p.get("status") == "open"]

    if not _open_opts:
        st.info(
            "No open options positions found.  \n"
            "Add positions on the **Home** page using the sidebar options chain, "
            "or build multi-leg strategies with the Strategy Builder."
        )
        st.stop()

    _singles    = [p for p in _open_opts if p.get("type", "single") != "strategy"]
    _strategies = [p for p in _open_opts if p.get("type") == "strategy"]

    st.caption(
        f"{len(_singles)} single-leg position(s) · "
        f"{len(_strategies)} multi-leg strategy/ies · "
        f"{sum(len(p.get('legs',[])) for p in _strategies)} strategy legs"
    )

    if st.button("🔄 Calculate Live Greeks", type="primary", key="_pg_run"):
        with st.spinner("Fetching live prices and IV from options chain…"):
            _rows = []

            # ── Single-leg positions ──────────────────────────────────────────
            for _pos in _singles:
                _ul   = _pos.get("underlying", "")
                _ot   = _pos.get("option_type", "call")
                _K    = float(_pos.get("strike", 0))
                _exp  = _pos.get("expiration", "")
                _ct   = int(_pos.get("contracts", 1))
                _cb   = float(_pos.get("cost_basis", 0))

                _S    = _spot_price(_ul) or 0.0
                _T    = _dte_years(_exp)

                # Live IV — fall back to iv_at_entry
                _, _live_iv = _live_option_row(_ul, _ot, _K, _exp)
                _iv = _live_iv or float(_pos.get("iv_at_entry", 0.3) or 0.3)
                _iv_src = "live" if _live_iv else "entry"

                _g = options_calc.calculate_greeks(_S, _K, _T, _iv, _ot) if _S > 0 else {
                    k: 0.0 for k in ("delta","gamma","theta","vega","rho")
                }

                # Single-leg from Home sidebar = always long
                _rows.append({
                    "Position":    f"{_ul} {_ot.upper()} ${_K:.0f}",
                    "Underlying":  _ul,
                    "Type":        _ot,
                    "L/S":         "Long",
                    "Strike":      _K,
                    "Expiration":  _exp,
                    "DTE":         max(int(_T * 365), 0),
                    "Contracts":   _ct,
                    "Spot":        _S,
                    "IV%":         _iv * 100,
                    "IV Source":   _iv_src,
                    "Cost Basis":  _cb,
                    "Δ Delta":     _g["delta"] * _ct * 100,
                    "Γ Gamma":     _g["gamma"] * _ct * 100,
                    "Θ Theta":     _g["theta"] * _ct * 100,
                    "V Vega":      _g["vega"]  * _ct * 100,
                    "ρ Rho":       _g["rho"]   * _ct * 100,
                })

            # ── Strategy legs ─────────────────────────────────────────────────
            for _strat in _strategies:
                _ul = _strat.get("underlying", "")
                _S  = _spot_price(_ul) or 0.0
                for _lg in _strat.get("legs", []):
                    _ot   = _lg.get("option_type", "call")
                    _K    = float(_lg.get("strike", 0))
                    _exp  = _lg.get("expiration", "")
                    _pos  = _lg.get("position", "long")
                    _ct   = int(_lg.get("contracts", 1))
                    _cb   = float(_lg.get("cost_basis", 0))
                    _sign = 1 if _pos == "long" else -1
                    _T    = _dte_years(_exp)

                    _, _live_iv = _live_option_row(_ul, _ot, _K, _exp)
                    _iv = _live_iv or float(_lg.get("iv_at_entry", 0.3) or 0.3)
                    _iv_src = "live" if _live_iv else "entry"

                    _g = options_calc.calculate_greeks(_S, _K, _T, _iv, _ot) if _S > 0 else {
                        k: 0.0 for k in ("delta","gamma","theta","vega","rho")
                    }

                    _label = f"{_strat.get('name','Strategy')} · {_lg.get('label', _ot.upper())}"
                    _rows.append({
                        "Position":    _label,
                        "Underlying":  _ul,
                        "Type":        _ot,
                        "L/S":         _pos.capitalize(),
                        "Strike":      _K,
                        "Expiration":  _exp,
                        "DTE":         max(int(_T * 365), 0),
                        "Contracts":   _ct,
                        "Spot":        _S,
                        "IV%":         _iv * 100,
                        "IV Source":   _iv_src,
                        "Cost Basis":  _cb,
                        "Δ Delta":     _g["delta"] * _ct * 100 * _sign,
                        "Γ Gamma":     _g["gamma"] * _ct * 100 * _sign,
                        "Θ Theta":     _g["theta"] * _ct * 100 * _sign,
                        "V Vega":      _g["vega"]  * _ct * 100 * _sign,
                        "ρ Rho":       _g["rho"]   * _ct * 100 * _sign,
                    })

            st.session_state["_pg_rows"] = _rows

    # ── Display results ───────────────────────────────────────────────────────

    _rows = st.session_state.get("_pg_rows")

    if not _rows:
        st.info("Click **Calculate Live Greeks** to fetch live IV and compute Greeks for all positions.")
    else:
        _df = pd.DataFrame(_rows)

        # ── Aggregate row ─────────────────────────────────────────────────────
        _net_delta = _df["Δ Delta"].sum()
        _net_gamma = _df["Γ Gamma"].sum()
        _net_theta = _df["Θ Theta"].sum()
        _net_vega  = _df["V Vega"].sum()
        _net_rho   = _df["ρ Rho"].sum()

        st.subheader("📐 Aggregate Portfolio Greeks")
        _m1, _m2, _m3, _m4, _m5 = st.columns(5)
        with _m1:
            st.metric("Net Delta", f"{_net_delta:+.1f}",
                      help="Total directional exposure — like holding this many shares.")
        with _m2:
            st.metric("Net Gamma", f"{_net_gamma:+.4f}",
                      help="Rate of delta change per $1 move.")
        with _m3:
            st.metric("Daily Theta", f"${_net_theta:+.2f}",
                      help="Daily P&L from time decay.")
        with _m4:
            st.metric("Net Vega", f"${_net_vega:+.2f}",
                      help="P&L per 1% IV move.")
        with _m5:
            st.metric("Net Rho", f"${_net_rho:+.2f}",
                      help="P&L per 1% rate move.")

        # Risk flags
        if _net_delta > 500:
            st.warning("**Long-biased** — net delta > 500. Portfolio benefits from rising prices but exposed to downside.")
        elif _net_delta < -500:
            st.warning("**Short-biased** — net delta < −500. Portfolio benefits from falling prices but exposed to rallies.")
        if _net_theta < -100:
            st.error(f"**High theta burn** — losing ${abs(_net_theta):.2f}/day from time decay on long options.")
        elif _net_theta > 100:
            st.success(f"**Theta positive** — earning ${_net_theta:.2f}/day from short options.")
        if abs(_net_vega) > 1000:
            st.info(f"**High vega exposure** — {'benefits' if _net_vega > 0 else 'hurt'} by rising IV.")

        # ── Per-position table ────────────────────────────────────────────────
        st.subheader("📋 Position Detail")

        _display_df = _df[[
            "Position", "L/S", "DTE", "Contracts", "Spot", "IV%", "IV Source",
            "Δ Delta", "Γ Gamma", "Θ Theta", "V Vega",
        ]].copy()

        _styled = _display_df.style.format({
            "Spot":    "${:.2f}",
            "IV%":     "{:.1f}%",
            "Δ Delta": "{:+.2f}",
            "Γ Gamma": "{:+.4f}",
            "Θ Theta": "${:+.2f}",
            "V Vega":  "${:+.2f}",
        }).applymap(
            lambda v: f"color: {GAIN}" if (isinstance(v, (int, float)) and v > 0)
                      else (f"color: {LOSS}" if (isinstance(v, (int, float)) and v < 0) else ""),
            subset=["Δ Delta", "Θ Theta", "V Vega"],
        ).applymap(
            lambda v: f"color: {GAIN}" if v == "live" else f"color: {AMBER}",
            subset=["IV Source"],
        )
        st.dataframe(_styled, use_container_width=True, hide_index=True)
        st.caption("IV Source: *live* = fetched from options chain  ·  *amber* = using IV at entry as fallback")

        # ── Charts ────────────────────────────────────────────────────────────
        st.subheader("📊 Greeks Breakdown")
        _ch1, _ch2 = st.columns(2)

        with _ch1:
            # Delta by underlying
            _delta_by_ul = _df.groupby("Underlying")["Δ Delta"].sum().reset_index()
            fig_delta = go.Figure(go.Bar(
                x=_delta_by_ul["Underlying"],
                y=_delta_by_ul["Δ Delta"],
                marker_color=[GAIN if v >= 0 else LOSS for v in _delta_by_ul["Δ Delta"]],
                text=[f"{v:+.1f}" for v in _delta_by_ul["Δ Delta"]],
                textposition="outside",
            ))
            fig_delta.update_layout(**carbon_plotly_layout(
                title="Net Delta by Underlying",
                xaxis_title="Ticker",
                yaxis_title="Net Delta",
                height=340,
                showlegend=False,
            ))
            fig_delta.add_hline(y=0, line_dash="dash", line_color=BORDER)
            st.plotly_chart(fig_delta, use_container_width=True)

        with _ch2:
            # Theta + Vega per position (grouped bar)
            fig_tv = go.Figure()
            fig_tv.add_trace(go.Bar(
                name="Daily Θ ($)",
                x=_df["Position"],
                y=_df["Θ Theta"],
                marker_color=[GAIN if v >= 0 else LOSS for v in _df["Θ Theta"]],
            ))
            fig_tv.add_trace(go.Bar(
                name="Vega ($)",
                x=_df["Position"],
                y=_df["V Vega"],
                marker_color=[ACCENT if v >= 0 else AMBER for v in _df["V Vega"]],
            ))
            fig_tv.update_layout(**carbon_plotly_layout(
                title="Theta & Vega per Position",
                xaxis_title="",
                yaxis_title="$",
                height=340,
                barmode="group",
            ))
            fig_tv.update_xaxes(tickangle=-30)
            st.plotly_chart(fig_tv, use_container_width=True)

        # Greeks waterfall (aggregate)
        fig_wf = go.Figure(go.Bar(
            x=["Net Delta", "Net Gamma ×100", "Daily Theta ($)", "Net Vega ($)", "Net Rho ($)"],
            y=[_net_delta, _net_gamma * 100, _net_theta, _net_vega, _net_rho],
            marker_color=[
                GAIN if v >= 0 else LOSS
                for v in [_net_delta, _net_gamma * 100, _net_theta, _net_vega, _net_rho]
            ],
            text=[f"{v:+.2f}" for v in [_net_delta, _net_gamma * 100, _net_theta, _net_vega, _net_rho]],
            textposition="outside",
        ))
        fig_wf.update_layout(**carbon_plotly_layout(
            title="Portfolio Greeks Summary",
            height=300,
            showlegend=False,
        ))
        fig_wf.add_hline(y=0, line_dash="dash", line_color=BORDER)
        st.plotly_chart(fig_wf, use_container_width=True)

        # Download
        _csv = _df.to_csv(index=False)
        st.download_button(
            "📥 Download Greeks Table (CSV)",
            data=_csv,
            file_name=f"portfolio_greeks_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


# ── Educational expander ──────────────────────────────────────────────────────

with st.expander("📚 Options Trading Guide"):
    st.markdown(f"""
## Options Trading in {current_regime} Regime

**Regime:** {current_regime}  ·  **Realized Vol:** {market_volatility*100:.1f}%

---

### The Greeks Explained

**Delta (Δ):** Price sensitivity — moves $Δ for each $1 change in underlying.
**Gamma (Γ):** Rate of delta change — highest for ATM options near expiry.
**Theta (Θ):** Daily time decay — always negative for long options, positive for short.
**Vega (V):** IV sensitivity — long options benefit from rising IV, short options from falling IV.
**Rho (ρ):** Rate sensitivity — usually small; more relevant for long-dated LEAPS.

---

### Black-Scholes Model

Theoretical price derived from: spot price · strike · time · implied volatility · risk-free rate.
Compare market price to theoretical value to assess if an option is over- or under-priced.

---

### Risk Management

- Never risk more than 2–3% of account on a single trade
- Reduce size in High Vol regimes
- Take profits at ~50% of max gain on credit spreads
- Watch total portfolio delta and theta daily

---

### Strategy Selection by Regime

Use the **Recommendations** tab for regime-appropriate strategies tailored to your portfolio!
""")
