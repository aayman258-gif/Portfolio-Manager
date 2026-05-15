"""
Options Suite — consolidated 5-in-1 options intelligence page.
Tabs: Analytics | Live Chain | Portfolio Hedges | Vol Surface | Options Flow
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
    apply_carbon_theme, carbon_plotly_layout, flex_table, page_header, section_header,
    top_nav,
)
from calculations.options_analytics import OptionsAnalytics
from calculations.options_recommender import OptionsRecommender
from calculations.regime_detector import RegimeDetector
from calculations.strategy_builder import StrategyBuilder
from calculations.probability_utils import probability_of_profit, expected_value
from data.market_data import MarketDataLoader
from data.options_data import OptionsDataLoader

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Options Suite", page_icon="◈", layout="wide")
apply_carbon_theme()
top_nav("Options Suite")
page_header("Options Suite",
            "Analytics · Live Chain · Portfolio Hedges · Vol Surface · Options Flow")

# ── Shared singletons ─────────────────────────────────────────────────────────
_options_calc    = OptionsAnalytics(risk_free_rate=0.045)
_recommender     = OptionsRecommender()
_regime_detector = RegimeDetector()
_market_loader   = MarketDataLoader()
_options_loader  = OptionsDataLoader()
_RF = 0.045

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _dte_years(expiration: str) -> float:
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        return max((exp_date - datetime.now().date()).days / 365.0, 0.001)
    except Exception:
        return 0.001


@st.cache_data(ttl=300, show_spinner=False)
def _spot_price(ticker: str) -> Optional[float]:
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        return float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _live_option_row(underlying: str, option_type: str, strike: float, expiration: str):
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


@st.cache_data(ttl=1800, show_spinner=False)
def _get_regime():
    spy_data = _market_loader.load_index_data("SPY", "2y")
    vix_data = _market_loader.load_vix_data("2y")
    spy_prices, vix_prices = _market_loader.align_data(spy_data, vix_data)
    regime, signals = _regime_detector.classify_regime(spy_prices, vix_prices)
    return str(regime.iloc[-1]), float(signals["realized_vol"].iloc[-1])


def _pnl_color(v: float) -> str:
    return GAIN if v >= 0 else LOSS


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_a, tab_c, tab_h, tab_v, tab_f = st.tabs([
    "Analytics",
    "Live Chain",
    "Portfolio Hedges",
    "Vol Surface",
    "Options Flow",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB A — ANALYTICS (from 51_Options_Analytics)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_a:
    current_regime, market_volatility = _get_regime()

    a_tab1, a_tab2, a_tab3 = st.tabs([
        "Recommendations",
        "Analyze Position",
        "Portfolio Greeks",
    ])

    # ── A-TAB 1: Recommendations ──────────────────────────────────────────────
    with a_tab1:
        st.subheader(f"Regime-Aware Strategies — {current_regime}")
        st.info(f"**Regime:** {current_regime}  ·  **Realized Vol:** {market_volatility*100:.1f}%")

        _positions_df   = st.session_state.get("positions")
        _opts           = st.session_state.get("options_positions", [])
        _current_prices = st.session_state.get("current_prices", {})
        _total_value    = st.session_state.get("total_value")

        _equity_delta = 0.0
        _equity_tickers: list[str] = []
        if isinstance(_positions_df, pd.DataFrame) and not _positions_df.empty:
            for _, _row in _positions_df.iterrows():
                _equity_tickers.append(str(_row["ticker"]))
                _equity_delta += float(_row["shares"])

        _opt_net = {"delta": 0.0, "theta": 0.0, "vega": 0.0}
        for _p in _opts:
            if _p.get("status") != "open":
                continue
            if _p.get("type") == "strategy":
                for _lg in _p.get("legs", []):
                    _S0 = _spot_price(_p.get("underlying", "")) or 100.0
                    _T  = _dte_years(_lg.get("expiration", ""))
                    _iv = float(_lg.get("iv_at_entry", 0.3) or 0.3)
                    _g  = _options_calc.calculate_greeks(
                        _S0, _lg["strike"], _T, _iv, _lg["option_type"]
                    )
                    _sign = 1 if _lg["position"] == "long" else -1
                    _ct   = int(_lg.get("contracts", 1))
                    for _k in ("delta", "theta", "vega"):
                        _opt_net[_k] += _g[_k] * _sign * _ct * 100
            else:
                _S0 = _spot_price(_p.get("underlying", "")) or 100.0
                _T  = _dte_years(_p.get("expiration", ""))
                _iv = float(_p.get("iv_at_entry", 0.3) or 0.3)
                _g  = _options_calc.calculate_greeks(
                    _S0, _p["strike"], _T, _iv, _p["option_type"]
                )
                _ct = int(_p.get("contracts", 1))
                for _k in ("delta", "theta", "vega"):
                    _opt_net[_k] += _g[_k] * _ct * 100

        if _equity_tickers or _opts:
            with st.expander("Portfolio Context", expanded=True):
                _pc1, _pc2, _pc3, _pc4 = st.columns(4)
                with _pc1:
                    st.metric("Equity Δ Exposure", f"{_equity_delta:+,.0f} shares")
                with _pc2:
                    st.metric("Options Net Δ", f"{_opt_net['delta']:+,.1f}")
                with _pc3:
                    st.metric("Options Net Θ", f"${_opt_net['theta']:+,.2f}/day")
                with _pc4:
                    st.metric("Options Net Vega", f"${_opt_net['vega']:+,.2f}")

                if _equity_delta > 500 and _opt_net["delta"] >= 0:
                    st.warning("**Long-heavy portfolio** — consider protective puts or collars.")
                elif _opt_net["theta"] < -50:
                    st.warning("**High negative theta** — long options decaying. Consider rolling near-expiry positions.")
                elif _opt_net["vega"] > 500:
                    st.info("**Long vega** — portfolio benefits if IV rises.")

                if _equity_tickers:
                    st.caption(f"Holdings: {', '.join(_equity_tickers[:10])}" +
                               ("…" if len(_equity_tickers) > 10 else ""))

        strategies = _recommender.get_strategies_for_regime(current_regime)
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

                st.markdown("### Strike Recommendations")
                _default_ul = _equity_tickers[0] if _equity_tickers else ""
                _ul_hint    = f" (e.g. {_default_ul})" if _default_ul else ""
                underlying_price = st.number_input(
                    f"Underlying Price{_ul_hint} ({strategy['name']})",
                    min_value=1.0, value=450.0, step=1.0, key=f"a_price_{idx}",
                )
                col_a2, col_b2 = st.columns(2)
                with col_a2:
                    days_to_expiry = st.slider("Days to Expiration", 7, 90, 30, key=f"a_dte_{idx}")
                with col_b2:
                    trend = st.selectbox(
                        "Market Trend",
                        options=[0, 1, -1],
                        format_func=lambda x: "Sideways" if x == 0 else "Uptrend" if x == 1 else "Downtrend",
                        key=f"a_trend_{idx}",
                    )

                strike_recs = _recommender.calculate_strike_recommendations(
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

        st.subheader("Position Sizing Guide")
        _auto_acct = float(_total_value) if _total_value else 50_000.0
        account_size = st.number_input(
            "Account Size ($)",
            min_value=1_000, max_value=10_000_000,
            value=int(_auto_acct), step=1_000,
            key="a_acct_size",
        )
        if _total_value:
            st.caption(f"Auto-read from portfolio: ${_total_value:,.0f}")

        sizing = _recommender.get_position_sizing_guide(current_regime, account_size)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Max Risk Per Trade", f"${sizing['max_dollar_risk']:,.0f}",
                      f"{sizing['max_risk_per_trade']*100:.1f}%")
        with c2:
            st.metric("Max Positions", sizing["max_positions"])
        with c3:
            _tot_risk = sizing["max_dollar_risk"] * sizing["max_positions"]
            st.metric("Max Total Risk", f"${_tot_risk:,.0f}", f"{_tot_risk/account_size*100:.1f}%")
        st.info(f"**Note:** {sizing['notes']}")

    # ── A-TAB 2: Analyze Position ─────────────────────────────────────────────
    with a_tab2:
        st.subheader("Analyze Individual Option Position")

        _input_mode = st.radio(
            "Input mode", ["Manual Input", "From Live Chain"],
            horizontal=True, key="_a2_mode",
        )
        _pre = st.session_state.get("_a2_prefill", {})

        if _input_mode == "From Live Chain":
            st.markdown("##### Select from live options chain")
            _lc1, _lc2, _lc3 = st.columns([2, 2, 1])
            with _lc1:
                _chain_ticker = st.text_input(
                    "Ticker", value=st.session_state.get("_a2_chain_ticker", ""),
                    placeholder="e.g. AAPL", key="_a2_chain_ticker_input",
                ).upper().strip()
            with _lc2:
                _chain_type = st.radio("Type", ["call", "put"], horizontal=True, key="_a2_chain_type")
            with _lc3:
                _load_chain = st.button("Load Chain", key="_a2_load_chain")

            if _load_chain and _chain_ticker:
                with st.spinner(f"Fetching {_chain_ticker} options chain…"):
                    try:
                        _tk   = yf.Ticker(_chain_ticker)
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
                st.caption(f"**{_chain_ticker_loaded}** — spot ${_S_live:.2f} · {len(_exps_avail)} expirations")

                _sel_exp = st.selectbox("Expiration", _exps_avail, key="_a2_sel_exp")

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
                    def _strike_label(i: int, df=_chain_df) -> str:
                        r   = df.iloc[i]
                        bid = float(r.get("bid", 0) or 0)
                        ask = float(r.get("ask", 0) or 0)
                        mid = (bid + ask) / 2 if (bid + ask) > 0 else float(r.get("lastPrice", 0) or 0)
                        iv  = float(r.get("impliedVolatility", 0) or 0) * 100
                        oi  = int(r.get("openInterest", 0) or 0)
                        return f"${r['strike']:.2f}  mid ${mid:.2f}  IV {iv:.0f}%  OI {oi:,}"

                    _atm_idx = int((_chain_df["strike"] - _S_live).abs().argsort().iloc[0])
                    _sel_strike_idx = st.selectbox(
                        "Strike", range(len(_chain_df)),
                        index=_atm_idx, format_func=_strike_label, key="_a2_sel_strike_idx",
                    )
                    _sel_row    = _chain_df.iloc[_sel_strike_idx]
                    _sel_strike = float(_sel_row["strike"])
                    _sel_bid    = float(_sel_row.get("bid", 0) or 0)
                    _sel_ask    = float(_sel_row.get("ask", 0) or 0)
                    _sel_mid    = (_sel_bid + _sel_ask) / 2 if (_sel_bid + _sel_ask) > 0 else float(_sel_row.get("lastPrice", 0) or 0)
                    _sel_iv     = float(_sel_row.get("impliedVolatility", 0) or 0.3)
                    _sel_oi     = int(_sel_row.get("openInterest", 0) or 0)
                    _sel_vol    = int(_sel_row.get("volume", 0) or 0)

                    st.info(
                        f"**{_chain_ticker_loaded}** {_chain_type.upper()} "
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

        st.markdown("##### Position Details")
        col1, col2 = st.columns(2)
        with col1:
            underlying_price = st.number_input(
                "Current Underlying Price ($)",
                min_value=1.0, value=float(_pre.get("S", 450.0)), step=0.50, key="a2_S",
            )
            strike_price = st.number_input(
                "Strike Price ($)",
                min_value=1.0, value=float(_pre.get("K", 455.0)), step=1.0, key="a2_K",
            )
            option_type = st.selectbox(
                "Option Type", options=["call", "put"],
                index=0 if _pre.get("type", "call") == "call" else 1, key="a2_type",
            )
            contracts = st.number_input(
                "Contracts (negative for short)",
                min_value=-1000, max_value=1000, value=1, step=1, key="a2_ct",
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
                key="a2_exp",
            )
            days_to_exp = (expiration_date - datetime.now().date()).days
            time_to_exp = max(days_to_exp / 365, 0.001)
            st.metric("Days to Expiration", days_to_exp)

            implied_vol = st.slider(
                "Implied Volatility (%)",
                min_value=5.0, max_value=150.0,
                value=float(_pre.get("iv", 30.0)), step=1.0, key="a2_iv",
            ) / 100
            risk_free = st.slider(
                "Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.5, step=0.1, key="a2_rf",
            ) / 100
            dividend_yield = st.slider(
                "Dividend Yield (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="a2_q",
            ) / 100

        st.markdown("##### Optional: P&L Analysis")
        col_a2, col_b2 = st.columns(2)
        with col_a2:
            premium_paid = st.number_input(
                "Premium Paid per Contract ($/share)", min_value=0.0, value=0.0, step=0.10, key="a2_prem",
            )
        with col_b2:
            current_market_price = st.number_input(
                "Current Market Price ($/share)",
                min_value=0.0, value=float(_pre.get("mid", 0.0)), step=0.10, key="a2_mktpx",
            )

        premium_paid         = premium_paid         if premium_paid         > 0 else None
        current_market_price = current_market_price if current_market_price > 0 else None

        if st.button("Analyze Position", type="primary", key="a2_analyze"):
            _options_calc.risk_free_rate = risk_free
            analysis = _options_calc.analyze_option_position(
                S=underlying_price, K=strike_price, T=time_to_exp, sigma=implied_vol,
                option_type=option_type, contracts=contracts,
                premium_paid=premium_paid, current_price=current_market_price,
                q=dividend_yield,
            )
            st.success("Analysis complete")

            st.subheader("Key Metrics")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Theoretical Value", f"${analysis['theoretical_value']:.2f}")
            with c2: st.metric("Intrinsic Value",   f"${analysis['intrinsic_value']:.2f}")
            with c3: st.metric("Time Value",        f"${analysis['time_value']:.2f}")
            with c4: st.metric("Moneyness",          analysis["moneyness"])

            if analysis["pnl"] is not None:
                st.subheader("Profit & Loss")
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

            if analysis.get("fair_value_analysis"):
                st.subheader("Fair Value Analysis")
                fv = analysis["fair_value_analysis"]
                _fc1, _fc2, _fc3 = st.columns(3)
                with _fc1: st.metric("Market Price",      f"${fv['market_price']:.2f}")
                with _fc2: st.metric("Theoretical Value", f"${fv['theoretical_value']:.2f}")
                with _fc3:
                    st.metric("Difference", f"${fv['difference']:.2f}", f"{fv['difference_pct']:+.2f}%",
                              delta_color="inverse" if fv["difference"] > 0 else "normal")
                (st.warning if fv["rating"] in ("Overpriced", "Bad Short") else
                 st.success if fv["rating"] in ("Underpriced", "Good Short") else st.info)(
                    f"**Rating:** {fv['rating']}"
                )

            st.subheader("The Greeks")
            greeks = analysis["greeks"]
            pos_g  = analysis["position_greeks"]

            st.markdown("**Per Contract**")
            _gc = st.columns(5)
            for _col, _name, _fmt in zip(
                _gc,
                ["delta", "gamma", "theta", "vega", "rho"],
                [".4f", ".4f", "$.2f", "$.2f", "$.2f"],
            ):
                with _col:
                    _v = greeks[_name]
                    st.metric(_name.capitalize(),
                              f"${_v:.2f}" if "$" in _fmt else f"{_v:{_fmt}}")
                    st.caption(_options_calc.get_greek_interpretation(_name, _v)["description"])

            st.markdown("**Total Position**")
            _gp = st.columns(5)
            for _col, (_lbl, _val) in zip(_gp, [
                ("Position Δ", pos_g["delta"]),
                ("Position Γ", pos_g["gamma"]),
                ("Daily Θ",    pos_g["theta"]),
                ("Position V", pos_g["vega"]),
                ("Position ρ", pos_g["rho"]),
            ]):
                with _col:
                    _color = _pnl_color(_val)
                    st.markdown(
                        f"<div style='font-size:1.4rem;font-weight:300;color:{_color};'>{_val:+.4f}</div>"
                        f"<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:.13em;"
                        f"color:{DIM};'>{_lbl}</div>",
                        unsafe_allow_html=True,
                    )

            st.subheader("Price Sensitivity")
            _pr = np.linspace(underlying_price * 0.80, underlying_price * 1.20, 60)
            _pv, _pd_vals = [], []
            for _p in _pr:
                _pv.append(_options_calc.black_scholes(
                    _p, strike_price, time_to_exp, implied_vol, option_type, risk_free, dividend_yield,
                ) * contracts * 100)
                _pd_vals.append(_options_calc.calculate_greeks(
                    _p, strike_price, time_to_exp, implied_vol, option_type, risk_free, dividend_yield,
                )["delta"])

            fig_sens = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                subplot_titles=("Position Value vs Underlying", "Delta vs Underlying"),
                vertical_spacing=0.1,
            )
            fig_sens.add_trace(go.Scatter(x=_pr, y=_pv, name="Value",
                                          line=dict(color=ACCENT, width=2)), row=1, col=1)
            fig_sens.add_trace(go.Scatter(x=_pr, y=_pd_vals, name="Delta",
                                          line=dict(color=AMBER, width=2)), row=2, col=1)
            for _row in (1, 2):
                fig_sens.add_vline(x=underlying_price, line_dash="dash", line_color=SUBTLE, row=_row, col=1)
                fig_sens.add_vline(x=strike_price,     line_dash="dot",  line_color=LOSS,   row=_row, col=1)
            fig_sens.update_layout(**carbon_plotly_layout(height=560, showlegend=False))
            fig_sens.update_xaxes(title_text="Underlying Price ($)", row=2, col=1)
            fig_sens.update_yaxes(title_text="Value ($)",  row=1, col=1)
            fig_sens.update_yaxes(title_text="Delta",      row=2, col=1)
            st.plotly_chart(fig_sens, use_container_width=True)

    # ── A-TAB 3: Portfolio Greeks ─────────────────────────────────────────────
    with a_tab3:
        st.subheader("Portfolio-Level Greeks")
        _opts_all  = st.session_state.get("options_positions", [])
        _open_opts = [p for p in _opts_all if p.get("status") == "open"]

        if not _open_opts:
            st.info("No open options positions. Add positions on the **Home** page.")
        else:
            _singles    = [p for p in _open_opts if p.get("type", "single") != "strategy"]
            _strategies = [p for p in _open_opts if p.get("type") == "strategy"]
            st.caption(
                f"{len(_singles)} single-leg · {len(_strategies)} strategies · "
                f"{sum(len(p.get('legs',[])) for p in _strategies)} strategy legs"
            )

            if st.button("Calculate Live Greeks", type="primary", key="_pg_run"):
                with st.spinner("Fetching live prices and IV…"):
                    _rows = []
                    for _pos in _singles:
                        _ul = _pos.get("underlying", "")
                        _ot = _pos.get("option_type", "call")
                        _K  = float(_pos.get("strike", 0))
                        _exp = _pos.get("expiration", "")
                        _ct = int(_pos.get("contracts", 1))
                        _cb = float(_pos.get("cost_basis", 0))
                        _S  = _spot_price(_ul) or 0.0
                        _T  = _dte_years(_exp)
                        _, _live_iv = _live_option_row(_ul, _ot, _K, _exp)
                        _iv = _live_iv or float(_pos.get("iv_at_entry", 0.3) or 0.3)
                        _iv_src = "live" if _live_iv else "entry"
                        _g = _options_calc.calculate_greeks(_S, _K, _T, _iv, _ot) if _S > 0 else {
                            k: 0.0 for k in ("delta","gamma","theta","vega","rho")
                        }
                        _rows.append({
                            "Position":   f"{_ul} {_ot.upper()} ${_K:.0f}",
                            "Underlying": _ul, "Type": _ot, "L/S": "Long",
                            "DTE": max(int(_T * 365), 0), "Contracts": _ct,
                            "Spot": _S, "IV%": _iv * 100, "IV Source": _iv_src,
                            "Δ Delta": _g["delta"] * _ct * 100,
                            "Γ Gamma": _g["gamma"] * _ct * 100,
                            "Θ Theta": _g["theta"] * _ct * 100,
                            "V Vega":  _g["vega"]  * _ct * 100,
                            "ρ Rho":   _g["rho"]   * _ct * 100,
                        })
                    for _strat in _strategies:
                        _ul = _strat.get("underlying", "")
                        _S  = _spot_price(_ul) or 0.0
                        for _lg in _strat.get("legs", []):
                            _ot  = _lg.get("option_type", "call")
                            _K   = float(_lg.get("strike", 0))
                            _exp = _lg.get("expiration", "")
                            _pos_dir = _lg.get("position", "long")
                            _ct  = int(_lg.get("contracts", 1))
                            _sign = 1 if _pos_dir == "long" else -1
                            _T   = _dte_years(_exp)
                            _, _live_iv = _live_option_row(_ul, _ot, _K, _exp)
                            _iv = _live_iv or float(_lg.get("iv_at_entry", 0.3) or 0.3)
                            _iv_src = "live" if _live_iv else "entry"
                            _g = _options_calc.calculate_greeks(_S, _K, _T, _iv, _ot) if _S > 0 else {
                                k: 0.0 for k in ("delta","gamma","theta","vega","rho")
                            }
                            _rows.append({
                                "Position":   f"{_strat.get('name','Strategy')} · {_lg.get('label', _ot.upper())}",
                                "Underlying": _ul, "Type": _ot, "L/S": _pos_dir.capitalize(),
                                "DTE": max(int(_T * 365), 0), "Contracts": _ct,
                                "Spot": _S, "IV%": _iv * 100, "IV Source": _iv_src,
                                "Δ Delta": _g["delta"] * _ct * 100 * _sign,
                                "Γ Gamma": _g["gamma"] * _ct * 100 * _sign,
                                "Θ Theta": _g["theta"] * _ct * 100 * _sign,
                                "V Vega":  _g["vega"]  * _ct * 100 * _sign,
                                "ρ Rho":   _g["rho"]   * _ct * 100 * _sign,
                            })
                    st.session_state["_pg_rows"] = _rows

            _pg_rows = st.session_state.get("_pg_rows")
            if not _pg_rows:
                st.info("Click **Calculate Live Greeks** to fetch live IV and compute Greeks.")
            else:
                _df = pd.DataFrame(_pg_rows)
                _net_delta = _df["Δ Delta"].sum()
                _net_gamma = _df["Γ Gamma"].sum()
                _net_theta = _df["Θ Theta"].sum()
                _net_vega  = _df["V Vega"].sum()
                _net_rho   = _df["ρ Rho"].sum()

                st.subheader("Aggregate Portfolio Greeks")
                _m1, _m2, _m3, _m4, _m5 = st.columns(5)
                with _m1: st.metric("Net Delta", f"{_net_delta:+.1f}")
                with _m2: st.metric("Net Gamma", f"{_net_gamma:+.4f}")
                with _m3: st.metric("Daily Theta", f"${_net_theta:+.2f}")
                with _m4: st.metric("Net Vega", f"${_net_vega:+.2f}")
                with _m5: st.metric("Net Rho", f"${_net_rho:+.2f}")

                if _net_delta > 500:
                    st.warning("**Long-biased** — net delta > 500.")
                elif _net_delta < -500:
                    st.warning("**Short-biased** — net delta < −500.")
                if _net_theta < -100:
                    st.error(f"**High theta burn** — losing ${abs(_net_theta):.2f}/day.")
                elif _net_theta > 100:
                    st.success(f"**Theta positive** — earning ${_net_theta:.2f}/day.")
                if abs(_net_vega) > 1000:
                    st.info(f"**High vega exposure** — {'benefits' if _net_vega > 0 else 'hurt'} by rising IV.")

                st.subheader("Position Detail")
                _display_df = _df[[
                    "Position", "L/S", "DTE", "Contracts", "Spot", "IV%", "IV Source",
                    "Δ Delta", "Γ Gamma", "Θ Theta", "V Vega",
                ]].copy()
                flex_table(_display_df, columns=[
                    {"key": "Position",  "label": "Position", "width": "16%", "align": "left"},
                    {"key": "L/S",       "label": "L/S",      "width": "7%",  "align": "center"},
                    {"key": "DTE",       "label": "DTE",      "width": "7%",  "align": "right", "numeric": True},
                    {"key": "Contracts", "label": "Qty",      "width": "7%",  "align": "right", "numeric": True},
                    {"key": "Spot",      "label": "Spot",     "width": "10%", "align": "right", "fmt": lambda x: f"${x:.2f}", "numeric": True},
                    {"key": "IV%",       "label": "IV%",      "width": "9%",  "align": "right", "fmt": lambda x: f"{x:.1f}%", "numeric": True},
                    {"key": "IV Source", "label": "Source",   "width": "8%",  "align": "center"},
                    {"key": "Δ Delta",   "label": "Δ Delta",  "width": "10%", "align": "right", "fmt": lambda x: f"{x:+.2f}", "numeric": True},
                    {"key": "Γ Gamma",   "label": "Γ Gamma",  "width": "9%",  "align": "right", "fmt": lambda x: f"{x:+.4f}", "numeric": True},
                    {"key": "Θ Theta",   "label": "Θ Theta",  "width": "9%",  "align": "right", "fmt": lambda x: f"${x:+.2f}", "numeric": True},
                    {"key": "V Vega",    "label": "V Vega",   "width": "8%",  "align": "right", "fmt": lambda x: f"${x:+.2f}", "numeric": True},
                ], key="opts_positions")

                st.subheader("Greeks Breakdown")
                _ch1, _ch2 = st.columns(2)
                with _ch1:
                    _delta_by_ul = _df.groupby("Underlying")["Δ Delta"].sum().reset_index()
                    fig_delta = go.Figure(go.Bar(
                        x=_delta_by_ul["Underlying"],
                        y=_delta_by_ul["Δ Delta"],
                        marker_color=[GAIN if v >= 0 else LOSS for v in _delta_by_ul["Δ Delta"]],
                        text=[f"{v:+.1f}" for v in _delta_by_ul["Δ Delta"]],
                        textposition="outside",
                    ))
                    fig_delta.update_layout(**carbon_plotly_layout(
                        title="Net Delta by Underlying", height=340, showlegend=False,
                    ))
                    fig_delta.add_hline(y=0, line_dash="dash", line_color=BORDER)
                    st.plotly_chart(fig_delta, use_container_width=True)
                with _ch2:
                    fig_greeks = go.Figure()
                    _greek_cols = ["Θ Theta", "V Vega", "ρ Rho"]
                    _greek_colors = [LOSS, ACCENT, AMBER]
                    for _gcol, _gcolor in zip(_greek_cols, _greek_colors):
                        _by_ul = _df.groupby("Underlying")[_gcol].sum().reset_index()
                        fig_greeks.add_trace(go.Bar(
                            name=_gcol, x=_by_ul["Underlying"], y=_by_ul[_gcol],
                            marker_color=_gcolor, opacity=0.85,
                        ))
                    fig_greeks.update_layout(**carbon_plotly_layout(
                        title="Θ / V / ρ by Underlying", height=340, barmode="group",
                    ))
                    st.plotly_chart(fig_greeks, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB C — LIVE CHAIN (from 52_Live_Options_Chain)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_c:
    section_header("Live Options Chain", ACCENT)

    # Inline controls (replaces sidebar)
    _c_ctrl1, _c_ctrl2, _c_ctrl3 = st.columns([2, 1, 1])
    with _c_ctrl1:
        c_ticker = st.text_input(
            "Ticker Symbol", value=st.session_state.get("c_ticker", "SPY"),
            key="c_ticker_input", placeholder="SPY, AAPL, TSLA…",
        ).upper().strip()
    with _c_ctrl2:
        st.write("")
        st.write("")
        c_load_btn = st.button("Load Options Chain", type="primary", key="c_load_btn")
    with _c_ctrl3:
        st.write("")

    if c_load_btn and c_ticker:
        st.session_state["c_ticker"]  = c_ticker
        st.session_state["c_loaded"]  = True

    if st.session_state.get("c_loaded", False):
        c_ticker = st.session_state.get("c_ticker", "SPY")

        with st.spinner(f"Loading options data for {c_ticker}…"):
            summary = _options_loader.get_options_summary(c_ticker)

        if "error" in summary:
            st.error(f"Error: {summary['error']}")
        else:
            st.subheader(f"{c_ticker} Options Overview")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("Current Price", f"${summary['current_price']:.2f}")
            with col2: st.metric("Expirations", summary['num_expirations'])
            with col3: st.metric("Call Volume", f"{summary['total_call_volume']:,.0f}")
            with col4: st.metric("Put Volume",  f"{summary['total_put_volume']:,.0f}")
            with col5: st.metric("Put/Call Ratio", f"{summary['put_call_ratio']:.2f}")

            expirations = _options_loader.get_options_expirations(c_ticker)
            if not expirations:
                st.error(f"No options available for {c_ticker}")
            else:
                exp_with_dte = []
                for exp in expirations:
                    exp_date = datetime.strptime(exp, '%Y-%m-%d')
                    dte = (exp_date - datetime.now()).days
                    exp_with_dte.append(f"{exp} ({dte} DTE)")

                selected_exp_display = st.selectbox(
                    "Expiration Date", options=exp_with_dte, index=0, key="c_exp_select",
                )
                selected_expiration = selected_exp_display.split(' ')[0]

                with st.spinner("Loading options chain…"):
                    calls, puts, underlying_price = _options_loader.get_options_chain(
                        c_ticker, selected_expiration
                    )

                if calls.empty and puts.empty:
                    st.error("No options data available for this expiration")
                else:
                    exp_date = datetime.strptime(selected_expiration, '%Y-%m-%d')
                    days_to_exp = (exp_date - datetime.now()).days
                    time_to_exp = max(days_to_exp / 365, 0.001)
                    st.info(
                        f"**Expiration:** {selected_expiration} | "
                        f"**DTE:** {days_to_exp} | "
                        f"**Underlying:** ${underlying_price:.2f}"
                    )

                    c_tab1, c_tab2, c_tab3, c_tab4 = st.tabs([
                        "Options Chain",
                        "High Volume",
                        "IV Smile",
                        "Quick Analyze",
                    ])

                    with c_tab1:
                        st.subheader("Full Options Chain with Greeks")
                        with st.spinner("Calculating Greeks…"):
                            calls_g = calls.copy()
                            for idx, row in calls_g.iterrows():
                                if row.get('impliedVolatility', 0) > 0:
                                    g = _options_calc.calculate_greeks(
                                        S=underlying_price,
                                        K=row['strike'],
                                        T=time_to_exp,
                                        sigma=row['impliedVolatility'],
                                        option_type='call',
                                    )
                                    for gk in ('delta','gamma','theta','vega'):
                                        calls_g.at[idx, gk] = round(g[gk], 4)

                            puts_g = puts.copy()
                            for idx, row in puts_g.iterrows():
                                if row.get('impliedVolatility', 0) > 0:
                                    g = _options_calc.calculate_greeks(
                                        S=underlying_price,
                                        K=row['strike'],
                                        T=time_to_exp,
                                        sigma=row['impliedVolatility'],
                                        option_type='put',
                                    )
                                    for gk in ('delta','gamma','theta','vega'):
                                        puts_g.at[idx, gk] = round(g[gk], 4)

                        chain_cols = [
                            'strike', 'lastPrice', 'bid', 'ask', 'volume',
                            'openInterest', 'impliedVolatility',
                            'delta', 'gamma', 'theta', 'vega',
                        ]
                        existing_calls = [c for c in chain_cols if c in calls_g.columns]
                        existing_puts  = [c for c in chain_cols if c in puts_g.columns]

                        _chain_col_specs = {
                            'strike':            {"label": "Strike", "width": "12%", "align": "right", "fmt": lambda x: f"${x:.0f}", "numeric": True},
                            'lastPrice':         {"label": "Last",   "width": "12%", "align": "right", "fmt": lambda x: f"${x:.2f}", "numeric": True},
                            'bid':               {"label": "Bid",    "width": "11%", "align": "right", "fmt": lambda x: f"${x:.2f}", "numeric": True},
                            'ask':               {"label": "Ask",    "width": "11%", "align": "right", "fmt": lambda x: f"${x:.2f}", "numeric": True},
                            'volume':            {"label": "Vol",    "width": "10%", "align": "right", "numeric": True},
                            'openInterest':      {"label": "OI",     "width": "10%", "align": "right", "numeric": True},
                            'impliedVolatility': {"label": "IV",     "width": "10%", "align": "right", "fmt": lambda x: f"{x:.1%}", "numeric": True},
                            'delta':             {"label": "Δ",      "width": "9%",  "align": "right", "fmt": lambda x: f"{x:+.2f}", "numeric": True},
                            'gamma':             {"label": "Γ",      "width": "8%",  "align": "right", "fmt": lambda x: f"{x:.4f}", "numeric": True},
                            'theta':             {"label": "Θ",      "width": "4%",  "align": "right", "fmt": lambda x: f"{x:.2f}", "numeric": True},
                            'vega':              {"label": "V",      "width": "3%",  "align": "right", "fmt": lambda x: f"{x:.2f}", "numeric": True},
                        }
                        calls_cols = [{"key": k, **v} for k, v in _chain_col_specs.items() if k in existing_calls]
                        puts_cols  = [{"key": k, **v} for k, v in _chain_col_specs.items() if k in existing_puts]
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.markdown("**Calls**")
                            flex_table(calls_g[existing_calls], columns=calls_cols, key="chain_calls", row_height=32)
                        with cc2:
                            st.markdown("**Puts**")
                            flex_table(puts_g[existing_puts], columns=puts_cols, key="chain_puts", row_height=32)

                    with c_tab2:
                        st.subheader("High Volume Options")
                        all_opts = pd.concat([
                            calls.assign(type='Call'),
                            puts.assign(type='Put'),
                        ]).copy()
                        if 'volume' in all_opts.columns:
                            high_vol = all_opts[all_opts['volume'] > 0].nlargest(20, 'volume')
                            cols_show = [c for c in ['type','strike','volume','openInterest',
                                                      'lastPrice','bid','ask','impliedVolatility']
                                        if c in high_vol.columns]
                            _hv_specs = {
                                'type':            {"label": "Type",   "width": "10%", "align": "center"},
                                'strike':          {"label": "Strike", "width": "12%", "align": "right", "fmt": lambda x: f"${x:.0f}", "numeric": True},
                                'volume':          {"label": "Volume", "width": "13%", "align": "right", "numeric": True},
                                'openInterest':    {"label": "OI",     "width": "13%", "align": "right", "numeric": True},
                                'lastPrice':       {"label": "Last",   "width": "12%", "align": "right", "fmt": lambda x: f"${x:.2f}", "numeric": True},
                                'bid':             {"label": "Bid",    "width": "12%", "align": "right", "fmt": lambda x: f"${x:.2f}", "numeric": True},
                                'ask':             {"label": "Ask",    "width": "12%", "align": "right", "fmt": lambda x: f"${x:.2f}", "numeric": True},
                                'impliedVolatility': {"label": "IV",   "width": "16%", "align": "right", "fmt": lambda x: f"{x:.1%}", "numeric": True},
                            }
                            hv_cols = [{"key": k, **v} for k, v in _hv_specs.items() if k in cols_show]
                            flex_table(high_vol[cols_show], columns=hv_cols, key="chain_highvol", row_height=32)
                        else:
                            st.info("Volume data not available for this expiration.")

                    with c_tab3:
                        st.subheader("IV Smile")
                        if 'impliedVolatility' in calls.columns and 'strike' in calls.columns:
                            c_smile = calls[calls['impliedVolatility'] > 0][['strike','impliedVolatility']].dropna()
                            p_smile = puts[puts['impliedVolatility'] > 0][['strike','impliedVolatility']].dropna()
                            fig_smile = go.Figure()
                            if not c_smile.empty:
                                fig_smile.add_trace(go.Scatter(
                                    x=c_smile['strike'], y=c_smile['impliedVolatility'] * 100,
                                    name="Calls IV", mode="lines+markers",
                                    line=dict(color=ACCENT, width=2),
                                ))
                            if not p_smile.empty:
                                fig_smile.add_trace(go.Scatter(
                                    x=p_smile['strike'], y=p_smile['impliedVolatility'] * 100,
                                    name="Puts IV", mode="lines+markers",
                                    line=dict(color=LOSS, width=2),
                                ))
                            fig_smile.add_vline(
                                x=underlying_price, line_dash="dash",
                                line_color=AMBER, annotation_text=f"Spot ${underlying_price:.0f}",
                            )
                            fig_smile.update_layout(**carbon_plotly_layout(
                                title=f"{c_ticker} IV Smile — {selected_expiration}",
                                xaxis_title="Strike",
                                yaxis_title="Implied Volatility (%)",
                                height=400,
                            ))
                            st.plotly_chart(fig_smile, use_container_width=True)
                        else:
                            st.info("IV data not available.")

                    with c_tab4:
                        st.subheader("Quick Analyze — Pick any Strike")
                        _qa_type = st.radio("Option Type", ["call", "put"], horizontal=True, key="c_qa_type")
                        _qa_df   = calls_g if _qa_type == "call" else puts_g
                        if not _qa_df.empty and 'strike' in _qa_df.columns:
                            _qa_atm = int((_qa_df["strike"] - underlying_price).abs().argsort().iloc[0])
                            _qa_strike = st.selectbox(
                                "Strike", _qa_df["strike"].tolist(),
                                index=_qa_atm, key="c_qa_strike",
                            )
                            _qa_row = _qa_df[_qa_df["strike"] == _qa_strike].iloc[0]
                            _qa_iv  = float(_qa_row.get("impliedVolatility", 0.3) or 0.3)
                            _qa_last = float(_qa_row.get("lastPrice", 0) or 0)

                            qa_g = _options_calc.calculate_greeks(
                                underlying_price, _qa_strike, time_to_exp, _qa_iv, _qa_type
                            )
                            qa_pnl = probability_of_profit(
                                underlying_price, _qa_strike, _qa_iv, time_to_exp, _qa_type
                            )

                            _qc1, _qc2, _qc3, _qc4, _qc5 = st.columns(5)
                            with _qc1: st.metric("Last Price", f"${_qa_last:.2f}")
                            with _qc2: st.metric("IV", f"{_qa_iv*100:.1f}%")
                            with _qc3: st.metric("Delta", f"{qa_g['delta']:.3f}")
                            with _qc4: st.metric("Theta", f"${qa_g['theta']:.2f}/day")
                            with _qc5: st.metric("P(Profit)", f"{qa_pnl:.1%}")

    else:
        st.info("Enter a ticker and click **Load Options Chain** to begin.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB H — PORTFOLIO HEDGES (from 54_Portfolio_Hedges)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_h:
    section_header("Portfolio Hedge Analyzer", LOSS)

    @st.cache_data(ttl=900, show_spinner="Detecting regime…")
    def _hedge_get_regime() -> str:
        try:
            spy_data = _market_loader.load_index_data("SPY", "1y")
            vix_data = _market_loader.load_vix_data("1y")
            spy_px, vix_px = _market_loader.align_data(spy_data, vix_data)
            regime, _ = _regime_detector.classify_regime(spy_px, vix_px)
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
        calls, puts, price = _options_loader.get_options_chain("SPY", expiration)
        return calls, puts, float(price)

    @st.cache_data(ttl=120, show_spinner=False)
    def _hedge_spot(ticker: str) -> float:
        try:
            h = yf.Ticker(ticker).history(period="2d")
            return float(h["Close"].iloc[-1]) if not h.empty else 0.0
        except Exception:
            return 0.0

    @st.cache_data(ttl=300, show_spinner=False)
    def _stock_chain(ticker: str, expiration: str):
        try:
            calls, puts, price = _options_loader.get_options_chain(ticker, expiration)
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

    def _nearest_put(chain: pd.DataFrame, target_K: float):
        diff = (chain["strike"] - target_K).abs()
        row  = chain.loc[diff.idxmin()]
        prem = _safe_float(row.get("lastPrice"))
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
            bid = _safe_float(row.get("bid"))
            ask = _safe_float(row.get("ask"))
            prem = (bid + ask) / 2 if ask > 0 else 0.0
        return float(row["strike"]), max(prem, 0.01)

    _REGIME_COLORS = {
        "Risk-On":         "#22d3ee",
        "Caution":         "#f59e0b",
        "High Volatility": "#fb7185",
        "Stagflation":     "#f97316",
        "Recession":       "#dc2626",
        "Mean Reversion":  "#3b82f6",
        "Uncertain":       "#a78bfa",
        "Unknown":         "#888888",
        # legacy
        "Low Vol":         "#22d3ee",
        "High Vol":        "#fb7185",
        "Trending":        "#4ade80",
    }

    hedge_regime = _hedge_get_regime()
    rcolor = _REGIME_COLORS.get(hedge_regime, "#888888")
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.03);border-left:4px solid {rcolor};'
        f'border-radius:6px;padding:10px 18px;margin-bottom:16px;">'
        f'<span style="color:{rcolor};font-weight:700;">Regime: {hedge_regime}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Portfolio summary from session state ──────────────────────────────────
    _h_positions = st.session_state.get("positions")
    _h_prices    = st.session_state.get("current_prices", {})
    _h_total_val = st.session_state.get("total_value", 0.0)

    if not isinstance(_h_positions, pd.DataFrame) or _h_positions.empty:
        st.info("No portfolio loaded. Add positions on the **Home** page to get hedge recommendations.")
    else:
        _h_tickers = [str(r["ticker"]) for _, r in _h_positions.iterrows()]
        _h_shares  = {str(r["ticker"]): float(r["shares"]) for _, r in _h_positions.iterrows()}

        with st.spinner("Computing betas…"):
            betas = _calc_betas(tuple(sorted(_h_tickers)))

        portfolio_beta = sum(
            betas.get(tk, 1.0) * _h_shares[tk] * float(_h_prices.get(tk, 0))
            for tk in _h_tickers
        ) / (_h_total_val if _h_total_val else 1.0)

        h1, h2, h3 = st.columns(3)
        with h1: st.metric("Portfolio Value",   f"${_h_total_val:,.0f}")
        with h2: st.metric("Portfolio Beta",    f"{portfolio_beta:.2f}")
        with h3: st.metric("Beta-Adj. Exposure", f"${_h_total_val * portfolio_beta:,.0f}")

        h_tab1, h_tab2 = st.tabs(["Portfolio-Level Hedge (SPY Puts)", "Per-Position Hedges"])

        with h_tab1:
            st.subheader("SPY Put Hedge — Beta-Weighted")
            exps = _spy_expirations()
            if not exps:
                st.error("Could not fetch SPY options.")
            else:
                exp_opts = []
                for e in exps:
                    try:
                        dte = (datetime.strptime(e, '%Y-%m-%d') - datetime.now()).days
                        if 14 <= dte <= 90:
                            exp_opts.append(f"{e} ({dte}d)")
                    except ValueError:
                        pass

                if not exp_opts:
                    exp_opts = [f"{exps[0]} (near-term)"]

                sel_exp_str = st.selectbox("Expiration", exp_opts, key="h_spy_exp")
                sel_exp     = sel_exp_str.split(" ")[0]
                protect_pct = st.slider(
                    "Protection Level (% below spot)", 2, 15, 5, key="h_protect_pct",
                )

                _, spy_puts, spy_spot = _spy_chain(sel_exp)

                if not spy_puts.empty and spy_spot > 0:
                    target_K    = spy_spot * (1 - protect_pct / 100)
                    strike, prem = _nearest_put(spy_puts, target_K)
                    contracts_needed = int(np.ceil(
                        (_h_total_val * portfolio_beta) / (spy_spot * 100)
                    ))
                    total_cost = contracts_needed * prem * 100
                    protection_value = contracts_needed * (strike - spy_spot * 0.90) * 100 if strike > spy_spot * 0.90 else 0

                    pc1, pc2, pc3, pc4 = st.columns(4)
                    with pc1: st.metric("SPY Spot",         f"${spy_spot:.2f}")
                    with pc2: st.metric("Put Strike",       f"${strike:.2f}")
                    with pc3: st.metric("Contracts Needed", contracts_needed)
                    with pc4: st.metric("Total Premium",    f"${total_cost:,.0f}",
                                        f"{total_cost/_h_total_val*100:.2f}% of portfolio")

                    pct_protection = protect_pct
                    st.info(
                        f"**Hedge Summary:** {contracts_needed} SPY ${strike:.0f} puts expiring "
                        f"{sel_exp} for ${total_cost:,.0f} ({total_cost/_h_total_val*100:.2f}% of NAV). "
                        f"Covers a {pct_protection}% SPY drawdown on your beta-weighted exposure."
                    )

                    # P&L at expiry
                    spy_range = np.linspace(spy_spot * 0.75, spy_spot * 1.10, 60)
                    unhedged  = [_h_total_val * portfolio_beta * (s / spy_spot - 1) for s in spy_range]
                    hedge_pnl = [max(strike - s, 0) * contracts_needed * 100 - total_cost for s in spy_range]
                    hedged    = [u + h for u, h in zip(unhedged, hedge_pnl)]

                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(
                        x=spy_range, y=unhedged, name="Unhedged",
                        line=dict(color=LOSS, width=2, dash="dash"),
                    ))
                    fig_h.add_trace(go.Scatter(
                        x=spy_range, y=hedged, name="Hedged",
                        line=dict(color=GAIN, width=2),
                    ))
                    fig_h.add_vline(x=spy_spot, line_dash="dash", line_color=AMBER,
                                    annotation_text="Current")
                    fig_h.add_hline(y=0, line_color=SUBTLE, line_width=1)
                    fig_h.update_layout(**carbon_plotly_layout(
                        title="Portfolio P&L at SPY Expiry",
                        xaxis_title="SPY Price at Expiry",
                        yaxis_title="Portfolio P&L ($)",
                        height=400,
                    ))
                    st.plotly_chart(fig_h, use_container_width=True)

        with h_tab2:
            st.subheader("Individual Position Hedges")
            selected_ticker = st.selectbox(
                "Select Position", _h_tickers, key="h_pos_select",
            )
            pos_shares = _h_shares.get(selected_ticker, 0)
            pos_price  = float(_h_prices.get(selected_ticker, _hedge_spot(selected_ticker)))
            pos_value  = pos_shares * pos_price
            pos_beta   = betas.get(selected_ticker, 1.0)

            st.markdown(
                f"**{selected_ticker}** — {pos_shares:.0f} shares @ ${pos_price:.2f} = "
                f"${pos_value:,.0f}  ·  β={pos_beta:.2f}"
            )

            stock_exps = _stock_expirations(selected_ticker)
            if not stock_exps:
                st.warning(f"No options available for {selected_ticker}.")
            else:
                _se_opts = []
                for e in stock_exps:
                    try:
                        dte = (datetime.strptime(e, '%Y-%m-%d') - datetime.now()).days
                        if 14 <= dte <= 90:
                            _se_opts.append(f"{e} ({dte}d)")
                    except ValueError:
                        pass
                if not _se_opts:
                    _se_opts = [f"{stock_exps[0]} (near-term)"]

                sel_stock_exp = st.selectbox("Expiration", _se_opts, key="h_stock_exp").split(" ")[0]
                stock_calls, stock_puts, s_price = _stock_chain(selected_ticker, sel_stock_exp)

                if not stock_puts.empty:
                    pp1, pp2, pp3 = st.columns(3)
                    with pp1:
                        st.subheader("Protective Put")
                        target = s_price * 0.95
                        put_k, put_prem = _nearest_put(stock_puts, target)
                        put_cost = put_prem * pos_shares
                        st.metric("Strike",     f"${put_k:.2f}")
                        st.metric("Premium",    f"${put_prem:.2f}/share")
                        st.metric("Total Cost", f"${put_cost:,.0f}",
                                  f"{put_cost/pos_value*100:.2f}% of position")
                        st.caption("Max loss locked at strike − premium")

                    with pp2:
                        st.subheader("Covered Call")
                        if not stock_calls.empty:
                            call_target = s_price * 1.05
                            call_k, call_prem = _nearest_call(stock_calls, call_target)
                            call_credit = call_prem * pos_shares
                            st.metric("Strike",    f"${call_k:.2f}")
                            st.metric("Premium",   f"${call_prem:.2f}/share")
                            st.metric("Credit",    f"${call_credit:,.0f}",
                                      f"{call_credit/pos_value*100:.2f}% yield")
                            st.caption("Caps upside at strike, earns premium income")

                    with pp3:
                        st.subheader("Collar")
                        if not stock_calls.empty:
                            net_cost = put_prem - call_prem
                            collar_desc = "Credit" if net_cost < 0 else "Debit"
                            st.metric("Put Strike",  f"${put_k:.2f}")
                            st.metric("Call Strike", f"${call_k:.2f}")
                            st.metric(f"Net {collar_desc}", f"${abs(net_cost):.2f}/share")
                            st.caption("Protected range: bounded profit & loss")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB V — VOL SURFACE (from 55_Vol_Surface)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_v:
    section_header("Volatility Surface", ACCENT)

    _STRIKE_PCTS   = np.array([0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975,
                                1.00, 1.025, 1.05, 1.075, 1.10, 1.125, 1.15, 1.20])
    _STRIKE_LABELS = [f"{p*100:.1f}%" for p in _STRIKE_PCTS]

    # Inline controls (replaces sidebar)
    vc1, vc2, vc3, vc4, vc5 = st.columns([2, 1, 1, 1, 1])
    with vc1:
        v_ticker = st.text_input("Ticker", value="SPY", key="v_ticker_input").upper().strip()
    with vc2:
        v_max_exp = st.slider("Max Expirations", 4, 12, 8, key="v_max_exp")
    with vc3:
        v_min_dte = st.number_input("Min DTE", min_value=1, max_value=30, value=7, key="v_min_dte")
    with vc4:
        v_max_dte = st.number_input("Max DTE", min_value=30, max_value=730, value=180, key="v_max_dte")
    with vc5:
        st.write("")
        st.write("")
        v_build = st.button("Build Surface", type="primary", key="v_build_btn", use_container_width=True)

    if v_build:
        st.session_state["vs_ticker"] = v_ticker
        st.session_state["vs_params"] = (int(v_max_exp), int(v_min_dte), int(v_max_dte))
        st.session_state["vs_loaded"] = True

    if not st.session_state.get("vs_loaded", False):
        st.info("Configure settings above and click **Build Surface** to generate the volatility surface.")
        st.markdown("""
        - **IV Surface** — 3D or heatmap across all strikes and expirations
        - **Term Structure** — ATM IV vs DTE (Contango or Backwardation)
        - **Skew Analysis** — Put-call skew per expiration
        """)
    else:
        vs_ticker   = st.session_state["vs_ticker"]
        vs_max_exp, vs_min_dte, vs_max_dte = st.session_state["vs_params"]

        @st.cache_data(ttl=300, show_spinner="Building volatility surface…")
        def _build_iv_matrix(ticker: str, max_exp: int, min_dte: int, max_dte: int):
            loader = OptionsDataLoader()
            expirations = loader.get_options_expirations(ticker)
            if not expirations:
                return None, None, None, None, []
            now = datetime.now()
            dtes_list, iv_rows, exp_labels = [], [], []
            spot = None
            scanned = 0
            for exp in expirations:
                if scanned >= max_exp:
                    break
                try:
                    exp_date = datetime.strptime(exp, '%Y-%m-%d')
                except ValueError:
                    continue
                dte = (exp_date - now).days
                if not (min_dte <= dte <= max_dte):
                    scanned += 1
                    continue
                calls, puts, underlying = loader.get_options_chain(ticker, exp)
                if calls.empty or puts.empty:
                    scanned += 1
                    continue
                if spot is None:
                    spot = float(underlying)
                iv_row = []
                for pct in _STRIKE_PCTS:
                    target = underlying * pct
                    df = puts if pct <= 1.0 else calls
                    if df.empty:
                        iv_row.append(np.nan)
                        continue
                    diff = (df['strike'] - target).abs()
                    nearest = df.loc[diff.idxmin()]
                    iv_val = float(nearest.get('impliedVolatility', np.nan) or np.nan)
                    iv_row.append(iv_val if iv_val and iv_val > 0.005 else np.nan)
                valid = [v for v in iv_row if not np.isnan(v)]
                if not valid:
                    scanned += 1
                    continue
                chain_avg = float(np.mean(valid))
                iv_row = [v if not np.isnan(v) else chain_avg for v in iv_row]
                dtes_list.append(dte)
                iv_rows.append(iv_row)
                exp_labels.append(f"{exp} ({dte}d)")
                scanned += 1
            if len(dtes_list) < 2:
                return None, None, None, None, []
            sort_idx    = np.argsort(dtes_list)
            dtes_arr    = np.array(dtes_list)[sort_idx]
            iv_matrix   = np.array(iv_rows)[sort_idx]
            labels_sorted = [exp_labels[i] for i in sort_idx]
            return dtes_arr, _STRIKE_PCTS, iv_matrix, spot, labels_sorted

        with st.spinner(f"Fetching options data for {vs_ticker}…"):
            dtes, strike_pcts, iv_matrix, spot, exp_labels = _build_iv_matrix(
                vs_ticker, vs_max_exp, vs_min_dte, vs_max_dte
            )

        if dtes is None or len(dtes) < 2:
            st.error("Not enough data to build a surface. Try widening the DTE range or use a liquid ticker.")
        else:
            iv_pct = iv_matrix * 100
            atm_idx  = np.argmin(np.abs(strike_pcts - 1.0))
            atm_ivs  = iv_pct[:, atm_idx]
            front_atm = float(atm_ivs[0])
            back_atm  = float(atm_ivs[-1])

            vc_m1, vc_m2, vc_m3, vc_m4, vc_m5 = st.columns(5)
            vc_m1.metric("Spot Price",      f"${spot:.2f}")
            vc_m2.metric("Front ATM IV",    f"{front_atm:.1f}%")
            vc_m3.metric("Back ATM IV",     f"{back_atm:.1f}%")
            vc_m4.metric("Surface Min IV",  f"{float(np.nanmin(iv_pct)):.1f}%")
            vc_m5.metric("Surface Max IV",  f"{float(np.nanmax(iv_pct)):.1f}%")

            v_tab1, v_tab2, v_tab3 = st.tabs(["IV Surface", "Term Structure", "Skew Analysis"])

            with v_tab1:
                view_mode = st.radio("View Mode", ["3D Surface", "2D Heatmap"], horizontal=True, key="v_view")
                if view_mode == "3D Surface":
                    fig_v = go.Figure(go.Surface(
                        x=strike_pcts * 100, y=dtes, z=iv_pct,
                        colorscale='Viridis',
                        colorbar=dict(title="IV %", ticksuffix="%", tickfont=dict(color='#ffffff')),
                        hovertemplate="Strike: %{x:.1f}%<br>DTE: %{y}d<br>IV: %{z:.1f}%<extra></extra>",
                    ))
                    fig_v.update_layout(**carbon_plotly_layout(
                        height=620, title=f"{vs_ticker} Implied Volatility Surface",
                        scene=dict(
                            xaxis=dict(title="Strike %", ticksuffix="%", color='#ffffff'),
                            yaxis=dict(title="Days to Expiry", color='#ffffff'),
                            zaxis=dict(title="IV %", ticksuffix="%", color='#ffffff'),
                            bgcolor='#111111',
                        ),
                        margin=dict(l=0, r=0, t=50, b=0),
                    ))
                else:
                    fig_v = go.Figure(go.Heatmap(
                        x=_STRIKE_LABELS, y=[f"{d}d" for d in dtes], z=iv_pct,
                        colorscale='Viridis',
                        colorbar=dict(title="IV %", ticksuffix="%", tickfont=dict(color='#ffffff')),
                        hovertemplate="Strike: %{x}<br>DTE: %{y}<br>IV: %{z:.1f}%<extra></extra>",
                        text=np.round(iv_pct, 1),
                        texttemplate="%{text:.1f}%",
                        textfont=dict(size=10),
                    ))
                    fig_v.update_layout(**carbon_plotly_layout(
                        height=max(300, len(dtes) * 45 + 120),
                        title=f"{vs_ticker} IV Heatmap",
                    ))
                st.plotly_chart(fig_v, use_container_width=True)

            with v_tab2:
                # ATM IV term structure
                fig_ts = go.Figure(go.Scatter(
                    x=dtes, y=atm_ivs, mode="lines+markers",
                    line=dict(color=ACCENT, width=2),
                    marker=dict(size=8, color=ACCENT),
                    text=exp_labels, hovertemplate="%{text}<br>ATM IV: %{y:.1f}%<extra></extra>",
                ))
                ts_label = "Contango (Far > Near)" if back_atm > front_atm else "Backwardation (Near > Far)"
                fig_ts.update_layout(**carbon_plotly_layout(
                    title=f"{vs_ticker} ATM IV Term Structure — {ts_label}",
                    xaxis_title="Days to Expiry",
                    yaxis_title="ATM IV (%)",
                    height=400,
                ))
                st.plotly_chart(fig_ts, use_container_width=True)

            with v_tab3:
                # Skew: OTM put IV minus OTM call IV per expiration
                put_5pct_idx  = np.argmin(np.abs(strike_pcts - 0.95))
                call_5pct_idx = np.argmin(np.abs(strike_pcts - 1.05))
                skews = iv_pct[:, put_5pct_idx] - iv_pct[:, call_5pct_idx]

                fig_skew = go.Figure(go.Bar(
                    x=exp_labels, y=skews,
                    marker_color=[LOSS if s > 0 else ACCENT for s in skews],
                    hovertemplate="%{x}<br>Skew: %{y:.1f}%<extra></extra>",
                ))
                fig_skew.update_layout(**carbon_plotly_layout(
                    title=f"{vs_ticker} Put-Call Skew (5%-OTM Put IV − 5%-OTM Call IV)",
                    xaxis_title="Expiration",
                    yaxis_title="Skew (%)",
                    height=400,
                    showlegend=False,
                ))
                fig_skew.add_hline(y=0, line_dash="dash", line_color=SUBTLE)
                st.plotly_chart(fig_skew, use_container_width=True)
                st.caption("Positive = put IV > call IV = bearish hedging demand")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB F — OPTIONS FLOW (from 56_Options_Flow)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_f:
    section_header("Options Flow Scanner", AMBER)

    # Inline controls (replaces sidebar)
    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 1, 1, 1, 1])
    with fc1:
        f_ticker = st.text_input("Ticker Symbol", value="SPY", key="f_ticker_input").upper().strip()
    with fc2:
        f_n_exps = st.selectbox("Expirations", [3, 5, 8, "All"], index=0, key="f_n_exps")
    with fc3:
        f_vol_oi = st.slider("Vol/OI Threshold", 1.0, 10.0, 2.0, 0.5, key="f_vol_oi")
    with fc4:
        f_min_vol = st.number_input("Min Volume", min_value=10, max_value=1000, value=100, step=10, key="f_min_vol")
    with fc5:
        st.write("")
        st.write("")
        f_scan = st.button("Scan Flow", type="primary", key="f_scan_btn", use_container_width=True)

    if f_scan:
        st.session_state["flow_ticker"] = f_ticker
        st.session_state["flow_params"] = (f_n_exps, f_vol_oi, int(f_min_vol))
        st.session_state["flow_loaded"] = True

    if not st.session_state.get("flow_loaded", False):
        st.info("Set scanner parameters above and click **Scan Flow** to detect unusual activity.")
        st.markdown("""
        **How it works:**
        - **Volume/OI Ratio > threshold** → potential new position by informed traders
        - **Bought** → last price ≥ mid (paid the ask)
        - **Sold** → last price < mid (hit the bid)
        - **Dollar Premium** = Volume × Last × 100
        """)
    else:
        f_ticker_s = st.session_state["flow_ticker"]
        f_n_exps_s, f_vol_oi_s, f_min_vol_s = st.session_state["flow_params"]

        @st.cache_data(ttl=180, show_spinner="Scanning options flow…")
        def _get_flow(ticker: str, n_exps, vol_oi_thresh: float, min_volume: int):
            loader = OptionsDataLoader()
            expirations = loader.get_options_expirations(ticker)
            if not expirations:
                return pd.DataFrame(), {}
            exp_list = expirations if n_exps == "All" else expirations[:int(n_exps)]
            now = datetime.now()

            # ── Pre-fetch OI from yfinance for all needed expirations ─────────
            # yfinance is the only reliable source of open interest (OPRA data).
            # We build a master dict keyed (exp, strike, 'call'|'put') → OI.
            # Expirations yfinance doesn't carry (some weeklies) are tracked in
            # yf_exp_missing so we can skip them rather than falsely flagging OI=0.
            oi_master: dict[tuple[str, float, str], int] = {}
            yf_exp_missing: set[str] = set()
            try:
                yf_available = set(yf.Ticker(ticker).options)
                for exp in exp_list:
                    if exp not in yf_available:
                        yf_exp_missing.add(exp)
                        continue
                    try:
                        chain = yf.Ticker(ticker).option_chain(exp)
                        for row in chain.calls.itertuples():
                            oi_master[(exp, float(row.strike), 'call')] = int(getattr(row, 'openInterest', 0) or 0)
                        for row in chain.puts.itertuples():
                            oi_master[(exp, float(row.strike), 'put')] = int(getattr(row, 'openInterest', 0) or 0)
                    except Exception:
                        yf_exp_missing.add(exp)
            except Exception:
                pass  # If yfinance fails entirely, all exps treated as missing

            all_rows = []
            for exp in exp_list:
                try:
                    dte = (datetime.strptime(exp, '%Y-%m-%d') - now).days
                except ValueError:
                    continue
                calls, puts, underlying = loader.get_options_chain(ticker, exp)
                exp_has_oi_data = exp not in yf_exp_missing

                for df_opt, otype in [(calls, 'Call'), (puts, 'Put')]:
                    if df_opt.empty:
                        continue
                    for _, row in df_opt.iterrows():
                        vol    = _safe_float(row.get('volume'), 0.0)
                        strike = _safe_float(row.get('strike'), 0.0)
                        last   = _safe_float(row.get('lastPrice'), 0.0)
                        bid    = _safe_float(row.get('bid'), 0.0)
                        ask    = _safe_float(row.get('ask'), 0.0)
                        iv     = _safe_float(row.get('impliedVolatility'), 0.0)

                        if vol < min_volume:
                            continue

                        # OI: use pre-fetched yfinance master map (most reliable)
                        oi_key = (exp, float(strike), otype.lower())
                        if exp_has_oi_data:
                            oi = oi_master.get(oi_key, 0)
                        else:
                            # yfinance has no data for this expiration — skip so we
                            # don't falsely mark every contract as unusual
                            continue

                        vol_oi_ratio = vol / oi if oi > 0 else np.nan
                        # Only flag as unusual if vol/OI ratio exceeds threshold OR OI is
                        # genuinely zero (brand-new position on a known expiration)
                        if not np.isnan(vol_oi_ratio):
                            unusual = vol_oi_ratio >= vol_oi_thresh
                        else:
                            # OI = 0 on a date yfinance knows about → new position
                            unusual = True

                        mid  = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                        side = 'Bought' if last >= mid - 0.01 else 'Sold'
                        dollar_prem = vol * last * 100
                        all_rows.append({
                            'Type': otype, 'Strike': strike, 'Expiry': exp, 'DTE': dte,
                            'Last': last, 'Bid': bid, 'Ask': ask, 'IV (%)': round(iv * 100, 1),
                            'Volume': int(vol) if np.isfinite(vol) else 0,
                            'OI': int(oi),
                            'Vol/OI': round(vol_oi_ratio, 2) if not np.isnan(vol_oi_ratio) else None,
                            'Side': side, 'Dollar Premium': dollar_prem,
                            'Unusual': unusual, 'Underlying': underlying,
                        })

            if not all_rows:
                return pd.DataFrame(), {}
            df = pd.DataFrame(all_rows)
            call_df = df[df['Type'] == 'Call']
            put_df  = df[df['Type'] == 'Put']
            cp_total = call_df['Dollar Premium'].sum() + put_df['Dollar Premium'].sum()
            summary = {
                'total_call_prem': call_df['Dollar Premium'].sum(),
                'total_put_prem':  put_df['Dollar Premium'].sum(),
                'total_rows':      len(df),
                'unusual_rows':    int(df['Unusual'].sum()),
                'underlying':      float(df['Underlying'].iloc[0]) if not df.empty else 0,
                'flow_ratio':      (call_df['Dollar Premium'].sum() / cp_total) if cp_total > 0 else 0.5,
            }
            unusual_df = df[df['Unusual']].sort_values('Dollar Premium', ascending=False)
            return unusual_df.reset_index(drop=True), summary

        with st.spinner(f"Scanning {f_ticker_s} options flow…"):
            flow_df, flow_summary = _get_flow(f_ticker_s, f_n_exps_s, f_vol_oi_s, f_min_vol_s)

        if flow_df.empty:
            st.warning(
                f"No unusual flow detected for **{f_ticker_s}** with current settings. "
                "Try lowering the Vol/OI threshold or Min Volume."
            )
        else:
            st.subheader(f"{f_ticker_s} Flow Summary — ${flow_summary['underlying']:.2f}")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Call Premium", f"${flow_summary['total_call_prem']:,.0f}")
            m2.metric("Total Put Premium",  f"${flow_summary['total_put_prem']:,.0f}")
            ratio = flow_summary['flow_ratio']
            ratio_label = "Call-Heavy" if ratio > 0.6 else "Put-Heavy" if ratio < 0.4 else "Balanced"
            m3.metric("Call/Total Flow",  f"{ratio:.1%}", ratio_label)
            m4.metric("Unusual Trades",   f"{flow_summary['unusual_rows']}")
            m5.metric("Scanned Options",  f"{flow_summary['total_rows']}")

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=[flow_summary['total_call_prem']], y=['Flow'], orientation='h',
                name='Call Premium', marker_color=ACCENT,
                text=f"${flow_summary['total_call_prem']/1e6:.1f}M calls",
                textposition='inside', insidetextanchor='start',
            ))
            fig_bar.add_trace(go.Bar(
                x=[flow_summary['total_put_prem']], y=['Flow'], orientation='h',
                name='Put Premium', marker_color=LOSS,
                text=f"${flow_summary['total_put_prem']/1e6:.1f}M puts",
                textposition='inside', insidetextanchor='end',
            ))
            fig_bar.update_layout(**carbon_plotly_layout(
                height=120, barmode='stack',
                xaxis_title="Dollar Premium ($)",
                showlegend=True,
                margin=dict(l=50, r=30, t=30, b=30),
            ))
            st.plotly_chart(fig_bar, use_container_width=True)

            if ratio > 0.65:
                interp = "**Call-dominated flow** — Bullish bias."
                interp_color = ACCENT
            elif ratio < 0.35:
                interp = "**Put-dominated flow** — Bearish/protective bias."
                interp_color = LOSS
            else:
                interp = "**Balanced flow** — No strong directional bias."
                interp_color = AMBER

            st.markdown(
                f'<div style="background:rgba(255,255,255,0.03);border-left:3px solid {interp_color};'
                f'border-radius:4px;padding:8px 16px;margin-bottom:16px;">{interp}</div>',
                unsafe_allow_html=True,
            )

            st.subheader("Unusual Activity Table")
            st.markdown(
                f"Showing **{len(flow_df)}** trades "
                f"(Vol/OI ≥ {f_vol_oi_s}×, Volume ≥ {f_min_vol_s})"
            )

            _flow_df = flow_df[[
                'Type', 'Strike', 'Expiry', 'DTE', 'Last', 'Bid', 'Ask',
                'IV (%)', 'Volume', 'OI', 'Vol/OI', 'Side', 'Dollar Premium',
            ]].copy()
            flex_table(_flow_df, columns=[
                {"key": "Type",           "label": "Type",    "width": "7%",  "align": "center"},
                {"key": "Side",           "label": "Side",    "width": "8%",  "align": "center"},
                {"key": "Strike",         "label": "Strike",  "width": "8%",  "align": "right", "fmt": lambda x: f"${x:.0f}", "numeric": True},
                {"key": "Expiry",         "label": "Expiry",  "width": "10%", "align": "left"},
                {"key": "DTE",            "label": "DTE",     "width": "5%",  "align": "right", "numeric": True},
                {"key": "Last",           "label": "Last",    "width": "7%",  "align": "right", "fmt": lambda x: f"${x:.2f}", "numeric": True},
                {"key": "Bid",            "label": "Bid",     "width": "7%",  "align": "right", "fmt": lambda x: f"${x:.2f}", "numeric": True},
                {"key": "Ask",            "label": "Ask",     "width": "7%",  "align": "right", "fmt": lambda x: f"${x:.2f}", "numeric": True},
                {"key": "IV (%)",         "label": "IV%",     "width": "7%",  "align": "right", "fmt": lambda x: f"{x:.1f}%", "numeric": True},
                {"key": "Volume",         "label": "Vol",     "width": "7%",  "align": "right", "fmt": lambda x: f"{x:,}", "numeric": True},
                {"key": "OI",             "label": "OI",      "width": "7%",  "align": "right", "fmt": lambda x: f"{x:,}", "numeric": True},
                {"key": "Vol/OI",         "label": "Vol/OI",  "width": "8%",  "align": "right", "fmt": lambda x: f"{x:.1f}×" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "New", "numeric": True},
                {"key": "Dollar Premium", "label": "Premium", "width": "12%", "align": "right", "fmt": lambda x: f"${x/1_000:.1f}K" if x < 1_000_000 else f"${x/1_000_000:.2f}M", "numeric": True},
            ], key="opts_flow", row_height=32)

            st.subheader("Top 10 Trades by Dollar Premium")
            top10 = flow_df.head(10).copy()
            for rank, (_, row) in enumerate(top10.iterrows(), 1):
                type_color = ACCENT if row['Type'] == 'Call' else LOSS
                side_color = ACCENT if row['Side'] == 'Bought' else AMBER
                dollar_str = (
                    f"${row['Dollar Premium']/1_000:.1f}K"
                    if row['Dollar Premium'] < 1_000_000
                    else f"${row['Dollar Premium']/1_000_000:.2f}M"
                )
                vol_oi_str = (
                    f"{row['Vol/OI']:.1f}×" if row['Vol/OI'] is not None
                    and not (isinstance(row['Vol/OI'], float) and np.isnan(row['Vol/OI']))
                    else "New OI"
                )
                st.markdown(
                    f"""<div style="background:#1a1a1a;border:1px solid rgba(34,211,238,0.15);
                    border-radius:10px;padding:12px 18px;margin-bottom:8px;display:flex;
                    align-items:center;gap:16px;">
                    <span style="font-size:22px;font-weight:700;color:#888888;min-width:32px">#{rank}</span>
                    <span style="background:{type_color}22;color:{type_color};padding:2px 10px;
                    border-radius:12px;font-weight:700;font-size:13px;">{row['Type'].upper()}</span>
                    <span style="background:{side_color}22;color:{side_color};padding:2px 10px;
                    border-radius:12px;font-weight:600;font-size:13px;">{row['Side']}</span>
                    <span style="font-weight:700;font-size:15px;color:#ffffff;">
                        ${row['Strike']:.0f} strike — {row['Expiry']} ({row['DTE']}d)
                    </span>
                    <span style="margin-left:auto;text-align:right;">
                        <span style="font-size:18px;font-weight:700;color:#ffffff;">{dollar_str}</span>
                        <span style="font-size:12px;color:#888888;margin-left:8px;">
                            Vol {row['Volume']:,} / OI {row['OI']:,} ({vol_oi_str})
                            &nbsp;|&nbsp; IV {row['IV (%)']:.1f}%
                        </span>
                    </span></div>""",
                    unsafe_allow_html=True,
                )
