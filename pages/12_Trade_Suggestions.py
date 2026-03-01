"""
Regime-Aware Portfolio Manager
Section 12: Trade Suggestions — Data-Driven Strategy Recommendations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from data.options_data import OptionsDataLoader
from calculations.options_analytics import OptionsAnalytics
from calculations.strategy_builder import StrategyBuilder
from calculations.options_recommender import OptionsRecommender
from calculations.probability_utils import probability_of_profit, expected_value
from calculations.regime_detector import RegimeDetector
from data.market_data import MarketDataLoader
from utils.carbon_theme import apply_carbon_theme, carbon_plotly_layout, regime_color, page_header

st.set_page_config(page_title="Trade Suggestions", page_icon="💡", layout="wide")
apply_carbon_theme()

page_header("💡 Trade Suggestions", "Algorithm-scored strategy recommendations based on IV rank, regime & direction")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
STRATEGY_PROFILES = {
    'Long Call':        {'bias': 'bullish',  'premium_type': 'buy',  'risk_defined': True},
    'Long Put':         {'bias': 'bearish',  'premium_type': 'buy',  'risk_defined': True},
    'Bull Call Spread': {'bias': 'bullish',  'premium_type': 'buy',  'risk_defined': True},
    'Bear Put Spread':  {'bias': 'bearish',  'premium_type': 'buy',  'risk_defined': True},
    'Iron Condor':      {'bias': 'neutral',  'premium_type': 'sell', 'risk_defined': True},
    'Short Strangle':   {'bias': 'neutral',  'premium_type': 'sell', 'risk_defined': False},
    'Long Straddle':    {'bias': 'volatile', 'premium_type': 'buy',  'risk_defined': True},
    'Long Strangle':    {'bias': 'volatile', 'premium_type': 'buy',  'risk_defined': True},
    'Iron Butterfly':   {'bias': 'neutral',  'premium_type': 'sell', 'risk_defined': True},
    'Call Butterfly':   {'bias': 'bullish',  'premium_type': 'buy',  'risk_defined': True},
}

REGIME_PREFERRED = {
    'Low Vol':        ['Iron Condor', 'Iron Butterfly', 'Call Butterfly', 'Bull Call Spread'],
    'High Vol':       ['Long Straddle', 'Long Strangle', 'Bull Call Spread', 'Bear Put Spread'],
    'Trending':       ['Bull Call Spread', 'Bear Put Spread', 'Long Call', 'Long Put', 'Call Butterfly'],
    'Mean Reversion': ['Short Strangle', 'Iron Butterfly', 'Iron Condor'],
    'Unknown':        ['Bull Call Spread', 'Iron Condor'],
}

_RISK_FREE_RATE = 0.045

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
options_loader  = OptionsDataLoader()
market_loader   = MarketDataLoader()
regime_detector = RegimeDetector()
recommender     = OptionsRecommender()


@st.cache_data(ttl=1800, show_spinner="Detecting market regime…")
def _get_regime():
    spy_data  = market_loader.load_index_data('SPY', '1y')
    vix_data  = market_loader.load_vix_data('1y')
    spy_px, vix_px = market_loader.align_data(spy_data, vix_data)
    regime, _ = regime_detector.classify_regime(spy_px, vix_px)
    return regime.iloc[-1]


@st.cache_data(ttl=3600, show_spinner="Computing IV rank…")
def _compute_iv_rank(ticker: str, current_atm_iv: float) -> float:
    """Compare current ATM IV against 1-year rolling HV range."""
    try:
        hist = yf.Ticker(ticker).history(period='1y')
        if hist.empty or len(hist) < 30:
            return 0.5
        log_ret = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        hv_series = log_ret.rolling(20).std() * np.sqrt(252)
        hv_series = hv_series.dropna()
        if len(hv_series) < 2:
            return 0.5
        hv_min = float(hv_series.min())
        hv_max = float(hv_series.max())
        if hv_max <= hv_min:
            return 0.5
        rank = (current_atm_iv - hv_min) / (hv_max - hv_min)
        return float(np.clip(rank, 0.0, 1.0))
    except Exception:
        return 0.5


def score_strategy(name: str, view: str, iv_rank: float,
                   regime: str, risk_tolerance: int) -> tuple:
    """Score a strategy 0-100. Returns (score, list_of_reasons)."""
    if name not in STRATEGY_PROFILES:
        return 0.0, []

    profile = STRATEGY_PROFILES[name]
    score   = 0.0
    reasons = []

    # ── Direction alignment (40 pts) ─────────────────────────────────────────
    bias_map = {
        'bullish':  ['Bullish'],
        'bearish':  ['Bearish'],
        'neutral':  ['Neutral'],
        'volatile': ['Volatile / Expecting Big Move'],
    }
    if view in bias_map.get(profile['bias'], []):
        score += 40
        reasons.append(f"Direction aligned: {profile['bias']} strategy matches {view.lower()} view")

    # ── IV rank (25 pts) ─────────────────────────────────────────────────────
    is_sell = profile['premium_type'] == 'sell'
    if iv_rank >= 0.70 and is_sell:
        score += 25
        reasons.append(f"IV rank {iv_rank:.0%} is high → selling premium is advantageous")
    elif iv_rank <= 0.35 and not is_sell:
        score += 25
        reasons.append(f"IV rank {iv_rank:.0%} is low → buying options is relatively cheap")
    elif 0.35 < iv_rank < 0.70:
        score += 12
        reasons.append(f"IV rank {iv_rank:.0%} is moderate (partial credit)")

    # ── Regime alignment (20 pts) ────────────────────────────────────────────
    if name in REGIME_PREFERRED.get(regime, []):
        score += 20
        reasons.append(f"Strategy preferred in '{regime}' regime")

    # ── Risk-defined (15 pts) ────────────────────────────────────────────────
    conservatism = (4 - risk_tolerance)  # 0 (aggressive) to 3 (conservative)
    if profile['risk_defined']:
        score += 15 * (conservatism / 3) if risk_tolerance < 3 else 15
        if conservatism >= 2:
            reasons.append("Risk-defined strategy matches conservative risk tolerance")
    else:
        if risk_tolerance < 2:
            score -= 25
            reasons.append("⚠️ Undefined risk strategy penalised for conservative tolerance")
        else:
            reasons.append("Undefined risk accepted given higher risk tolerance")

    return float(np.clip(score, 0.0, 100.0)), reasons


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Suggestion Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()

view_opts = [
    'Let Algorithm Decide',
    'Bullish',
    'Bearish',
    'Neutral',
    'Volatile / Expecting Big Move',
]
directional_view = st.sidebar.selectbox(
    "Directional View", view_opts,
    help="'Let Algorithm Decide' infers the best view from regime, momentum, and IV rank"
)

account_size = st.sidebar.number_input(
    "Account Size ($)", min_value=5_000, max_value=5_000_000, value=50_000, step=5_000,
    format="%d"
)

risk_tolerance = st.sidebar.select_slider(
    "Risk Tolerance",
    options=[1, 2, 3],
    value=2,
    format_func=lambda x: {1: "Conservative", 2: "Moderate", 3: "Aggressive"}[x]
)

target_dte = st.sidebar.selectbox(
    "Target DTE",
    [14, 21, 30, 45, 60, 90, 120, 180, 270, 365, 545, 730],
    index=2,
    format_func=lambda d: (f"{d}d" if d < 365
                           else f"{d}d (1yr)" if d == 365
                           else f"{d}d ({d//365}yr {d%365}d)" if d % 365
                           else f"{d}d ({d//365}yr)")
)

if st.sidebar.button("⚡ Generate Suggestions", type="primary", use_container_width=True):
    st.session_state['ts_ticker']  = ticker
    st.session_state['ts_view']    = directional_view
    st.session_state['ts_acct']    = account_size
    st.session_state['ts_risk']    = risk_tolerance
    st.session_state['ts_dte']     = target_dte
    st.session_state['ts_loaded']  = True

if not st.session_state.get('ts_loaded', False):
    st.info("👈 Configure your inputs in the sidebar and click **Generate Suggestions**.")
    st.stop()

ticker   = st.session_state['ts_ticker']
view     = st.session_state['ts_view']
acct     = st.session_state['ts_acct']
risk_tol = st.session_state['ts_risk']
t_dte    = st.session_state['ts_dte']

# ─────────────────────────────────────────────────────────────────────────────
# Market context row
# ─────────────────────────────────────────────────────────────────────────────
current_regime = _get_regime()
rc = regime_color(current_regime)

# Get options data
with st.spinner(f"Loading {ticker} options…"):
    expirations = options_loader.get_options_expirations(ticker)
    if not expirations:
        st.error(f"No options data available for {ticker}")
        st.stop()

    # Find expiration closest to target DTE
    now = datetime.now()
    best_exp = expirations[0]
    best_diff = 9999
    for exp in expirations:
        try:
            dte = (datetime.strptime(exp, '%Y-%m-%d') - now).days
            if abs(dte - t_dte) < best_diff:
                best_diff = abs(dte - t_dte)
                best_exp  = exp
                actual_dte = dte
        except ValueError:
            continue

    calls, puts, spot = options_loader.get_options_chain(ticker, best_exp)

if calls.empty or puts.empty:
    st.error("Could not fetch options chain. Try another ticker or expiration.")
    st.stop()

T = max(actual_dte / 365, 0.003)

# ATM IV
atm_idx = len(calls) // 2
atm_iv  = float(calls['impliedVolatility'].iloc[atm_idx]) if not calls.empty else 0.25
atm_iv  = max(atm_iv, 0.05)

iv_rank = _compute_iv_rank(ticker, atm_iv)

# HV proxy (20d realized vol)
try:
    hist_px = yf.Ticker(ticker).history(period='3mo')
    hv_20 = float(
        np.log(hist_px['Close'] / hist_px['Close'].shift(1))
        .dropna()
        .rolling(20).std()
        .iloc[-1] * np.sqrt(252)
    )
except Exception:
    hv_20 = atm_iv * 0.85

# ── Infer directional view when "Let Algorithm Decide" is selected ────────────
inferred_view = None
infer_reasons = []

if view == 'Let Algorithm Decide':
    # 1. Regime gives the primary signal
    if current_regime == 'High Vol':
        inferred_view = 'Volatile / Expecting Big Move'
        infer_reasons.append(f"Regime is **High Vol** — elevated volatility favours non-directional volatility strategies")
    elif current_regime == 'Low Vol':
        inferred_view = 'Neutral'
        infer_reasons.append(f"Regime is **Low Vol** — calm, range-bound conditions favour premium-selling neutral strategies")
    elif current_regime == 'Mean Reversion':
        inferred_view = 'Neutral'
        infer_reasons.append(f"Regime is **Mean Reversion** — choppy conditions favour neutral range strategies")
    else:
        # Trending or Unknown: use price momentum to pick direction
        try:
            _px = yf.Ticker(ticker).history(period='3mo')['Close']
            _ret_20  = float(_px.iloc[-1] / _px.iloc[-20]  - 1) if len(_px) >= 20  else 0
            _ret_50  = float(_px.iloc[-1] / _px.iloc[-50]  - 1) if len(_px) >= 50  else _ret_20
            _ma20    = float(_px.rolling(20).mean().iloc[-1])
            _ma50    = float(_px.rolling(50).mean().iloc[-1]) if len(_px) >= 50 else _ma20
            _above_ma = _px.iloc[-1] > _ma50
        except Exception:
            _ret_20, _ret_50, _above_ma = 0.0, 0.0, True

        # Require agreement between short-term return AND MA cross to call direction
        if _ret_20 > 0.02 and _above_ma:
            inferred_view = 'Bullish'
            infer_reasons.append(
                f"20-day return is **{_ret_20:+.1%}** and price is **above the 50-day MA** — momentum is positive"
            )
        elif _ret_20 < -0.02 and not _above_ma:
            inferred_view = 'Bearish'
            infer_reasons.append(
                f"20-day return is **{_ret_20:+.1%}** and price is **below the 50-day MA** — momentum is negative"
            )
        else:
            inferred_view = 'Neutral'
            infer_reasons.append(
                f"20-day return is **{_ret_20:+.1%}** with mixed MA signals — no strong directional edge, defaulting to Neutral"
            )
        if current_regime == 'Trending':
            infer_reasons.append(f"Regime is **Trending** — directional strategies get a score bonus")

    # 2. IV rank refines: very high IV → prefer non-directional even in trending regime
    if iv_rank >= 0.75 and inferred_view in ('Bullish', 'Bearish'):
        infer_reasons.append(
            f"IV rank is **{iv_rank:.0%}** (high) — premium is expensive, so a defined-risk spread "
            f"is preferred over a naked directional option"
        )
    elif iv_rank >= 0.75 and inferred_view == 'Volatile / Expecting Big Move':
        infer_reasons.append(
            f"IV rank is **{iv_rank:.0%}** — premium is rich, which normally argues against buying straddles; "
            f"consider iron condors if the move fails to materialise"
        )
    elif iv_rank <= 0.30:
        infer_reasons.append(
            f"IV rank is **{iv_rank:.0%}** (low) — options are cheap, making bought strategies attractive"
        )

    view = inferred_view  # use inferred view for all downstream scoring

# ── Context banner ────────────────────────────────────────────────────────────
col_ctx1, col_ctx2, col_ctx3 = st.columns([2, 2, 3])

with col_ctx1:
    st.markdown(
        f'<div style="background:{rc}22;border:1px solid {rc}55;border-radius:10px;'
        f'padding:14px 18px;text-align:center;">'
        f'<div style="font-size:11px;color:#888888;text-transform:uppercase;letter-spacing:0.08em;">Market Regime</div>'
        f'<div style="font-size:22px;font-weight:700;color:{rc};margin-top:4px;">{current_regime}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with col_ctx2:
    iv_color = '#fb7185' if iv_rank >= 0.70 else '#22d3ee' if iv_rank <= 0.35 else '#f59e0b'
    iv_label = "High — Sell Premium" if iv_rank >= 0.70 else "Low — Buy Options" if iv_rank <= 0.35 else "Moderate"
    st.markdown(
        f'<div style="background:{iv_color}15;border:1px solid {iv_color}55;border-radius:10px;'
        f'padding:14px 18px;text-align:center;">'
        f'<div style="font-size:11px;color:#888888;text-transform:uppercase;letter-spacing:0.08em;">IV Rank</div>'
        f'<div style="font-size:22px;font-weight:700;color:{iv_color};margin-top:4px;">{iv_rank:.0%}</div>'
        f'<div style="font-size:11px;color:{iv_color};margin-top:2px;">{iv_label}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with col_ctx3:
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(f"{ticker} Price",   f"${spot:.2f}")
    mc2.metric("ATM IV",            f"{atm_iv*100:.1f}%")
    mc3.metric("20d HV",            f"{hv_20*100:.1f}%")

st.divider()

# IV Rank gauge
fig_ivr = go.Figure(go.Indicator(
    mode="gauge+number",
    value=iv_rank * 100,
    number={'suffix': '%', 'font': {'color': '#ffffff', 'size': 28}},
    gauge={
        'axis': {'range': [0, 100], 'tickcolor': '#888888'},
        'bar':  {'color': iv_color},
        'steps': [
            {'range': [0, 35],  'color': '#1a3a1a'},
            {'range': [35, 70], 'color': '#2d2a15'},
            {'range': [70, 100],'color': '#3a1515'},
        ],
        'threshold': {'line': {'color': '#ffffff', 'width': 2}, 'value': iv_rank * 100}
    },
    title={'text': 'IV Rank (1-Year)', 'font': {'color': '#ffffff', 'size': 13}}
))
fig_ivr.update_layout(**carbon_plotly_layout(height=200, margin=dict(l=20, r=20, t=50, b=10)))

col_g1, col_g2 = st.columns([1, 2])
with col_g1:
    st.plotly_chart(fig_ivr, use_container_width=True)
with col_g2:
    st.markdown("### Strike Recommendations")
    strike_recs = recommender.calculate_strike_recommendations(
        spot, current_regime, hv_20,
        trend_direction=1 if 'Bullish' in view else (-1 if 'Bearish' in view else 0),
        days_to_expiry=t_dte
    )
    if strike_recs:
        rows = []
        for strat_name, rec in list(strike_recs.items())[:4]:
            rows.append({'Strategy': strat_name, 'Details': str(rec.get('rationale', ''))})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Strike recommendations not available for current regime settings.")

st.divider()

# ── Algorithm inference banner ────────────────────────────────────────────────
if inferred_view is not None:
    view_color = {
        'Bullish':                    '#22d3ee',
        'Bearish':                    '#fb7185',
        'Neutral':                    '#22d3ee',
        'Volatile / Expecting Big Move': '#f59e0b',
    }.get(inferred_view, '#888888')

    reasons_md = "\n".join(f"- {r}" for r in infer_reasons)
    st.markdown(
        f'<div style="background:{view_color}12;border:1px solid {view_color}55;'
        f'border-radius:10px;padding:14px 20px;margin-bottom:16px;">'
        f'<div style="font-size:12px;color:#888888;text-transform:uppercase;'
        f'letter-spacing:0.08em;margin-bottom:6px;">Algorithm Inferred View</div>'
        f'<div style="font-size:20px;font-weight:700;color:{view_color};margin-bottom:8px;">'
        f'{inferred_view}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    for r in infer_reasons:
        st.markdown(f"- {r}")
    st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# Score all strategies
# ─────────────────────────────────────────────────────────────────────────────
scored = []
for name in STRATEGY_PROFILES:
    score, reasons = score_strategy(name, view, iv_rank, current_regime, risk_tol)
    scored.append((name, score, reasons))

scored.sort(key=lambda x: x[1], reverse=True)
top3 = scored[:3]

st.subheader("🏆 Top 3 Suggested Strategies")
st.markdown(f"Scored for **{ticker}** | View: **{view}** | Regime: **{current_regime}** | "
            f"IV Rank: **{iv_rank:.0%}** | DTE: **{actual_dte}d**")

# ─────────────────────────────────────────────────────────────────────────────
# Strategy cards
# ─────────────────────────────────────────────────────────────────────────────
for rank, (strat_name, score, reasons) in enumerate(top3, 1):
    rank_colors = {1: '#f59e0b', 2: '#9ca3af', 3: '#cd7f32'}
    rank_color  = rank_colors.get(rank, '#888888')
    rank_medals = {1: '🥇', 2: '🥈', 3: '🥉'}

    with st.container():
        st.markdown(
            f'<div style="background:#1a1a1a;border:1px solid {rank_color}55;'
            f'border-radius:12px;padding:4px 18px 2px;margin-bottom:4px;">'
            f'<span style="font-size:20px">{rank_medals[rank]}</span>'
            f' <span style="font-size:18px;font-weight:700;color:{rank_color};">#{rank}</span>'
            f' <span style="font-size:18px;font-weight:700;color:#ffffff;margin-left:8px;">'
            f'{strat_name}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Score bar
        st.progress(int(score), text=f"Score: {score:.0f} / 100")

        with st.expander("View Strategy Details", expanded=(rank == 1)):
            # Why selected
            st.markdown("**Why selected:**")
            for r in reasons:
                st.markdown(f"- {r}")

            # Load template and compute analytics
            builder = StrategyBuilder()
            success = builder.load_template(strat_name, spot, calls, puts)

            if not success or not builder.legs:
                st.warning(f"Could not construct {strat_name} from current options chain.")
                continue

            summary = builder.get_strategy_summary(spot, T)
            if 'error' in summary:
                st.warning(summary['error'])
                continue

            # Probability metrics
            try:
                pop = probability_of_profit(spot, builder, atm_iv, _RISK_FREE_RATE, T)
                ev  = expected_value(spot, builder, atm_iv, _RISK_FREE_RATE, T)
            except Exception:
                pop, ev = 0.5, 0.0

            # Legs table
            st.markdown("**Legs:**")
            legs_df = builder.get_legs_dataframe()
            st.dataframe(
                legs_df.style.format({
                    'Strike':  '${:.2f}',
                    'Premium': '${:.2f}',
                    'IV':      '{:.2%}',
                    'Cost':    '${:+,.2f}',
                }),
                use_container_width=True, hide_index=True
            )

            # Key metrics
            cost   = summary['initial_cost']
            mp     = summary['max_profit']
            ml     = summary['max_loss']

            km1, km2, km3, km4, km5 = st.columns(5)
            km1.metric("Net Cost",
                       f"{'Credit' if cost < 0 else 'Debit'} ${abs(cost):,.0f}")
            km2.metric("Max Profit",
                       f"${mp:,.0f}" if mp < 1e9 else "Unlimited")
            km3.metric("Max Loss",
                       f"${ml:,.0f}" if ml > -1e9 else "Unlimited")
            km4.metric("Prob of Profit", f"{pop:.1%}")
            km5.metric("Expected Value", f"${ev:,.0f}")

            # Payoff diagram
            price_range, pnl = builder.calculate_payoff(spot)

            fig_pay = go.Figure()
            fig_pay.add_trace(go.Scatter(
                x=price_range, y=pnl,
                fill='tozeroy',
                fillcolor='rgba(34,211,238,0.08)',
                line=dict(color='#22d3ee', width=2.5),
                name='P&L at Expiry',
                hovertemplate="$%{x:.2f} → %{y:+,.0f}<extra></extra>"
            ))

            fig_pay.add_hline(y=0, line_color='#888888', line_dash='dash', line_width=1)
            fig_pay.add_vline(x=spot, line_color='#4a9eff', line_dash='dot', line_width=1.5,
                              annotation_text=f"${spot:.2f}", annotation_font_color='#4a9eff')

            for be in summary['breakevens']:
                fig_pay.add_vline(x=be, line_color='#f59e0b', line_dash='dot', line_width=1,
                                  annotation_text=f"BE ${be:.0f}",
                                  annotation_font_color='#f59e0b')

            fig_pay.add_trace(go.Scatter(
                x=[summary['max_profit_price']],
                y=[summary['max_profit']],
                mode='markers', name='Max Profit',
                marker=dict(color='#22d3ee', size=10, symbol='star'),
                hovertemplate=f"Max Profit: ${mp:,.0f}<extra></extra>"
            ))

            if summary['max_loss'] > -1e9:
                fig_pay.add_trace(go.Scatter(
                    x=[summary['max_loss_price']],
                    y=[summary['max_loss']],
                    mode='markers', name='Max Loss',
                    marker=dict(color='#fb7185', size=10, symbol='x'),
                    hovertemplate=f"Max Loss: ${ml:,.0f}<extra></extra>"
                ))

            fig_pay.update_layout(
                **carbon_plotly_layout(
                    height=320,
                    title=f"{strat_name} Payoff at Expiration ({actual_dte}d)",
                    xaxis_title="Underlying Price",
                    yaxis_title="P&L ($)",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=-0.2),
                )
            )
            fig_pay.update_xaxes(tickprefix="$", gridcolor='rgba(107,122,143,0.15)')
            fig_pay.update_yaxes(tickprefix="$", tickformat="+,d",
                                  gridcolor='rgba(107,122,143,0.15)')
            st.plotly_chart(fig_pay, use_container_width=True)

            # Position sizing
            sizing = recommender.get_position_sizing_guide(current_regime, acct)
            st.markdown(
                f"**Position Sizing ({current_regime} regime):** "
                f"Max risk per trade = **${sizing['max_dollar_risk']:,.0f}** "
                f"({sizing['max_risk_per_trade']*100:.1f}% of ${acct:,.0f} account) · "
                f"Max concurrent positions: **{sizing['max_positions']}** · "
                f"*{sizing['notes']}*"
            )

    st.markdown("")  # spacer between cards

# ─────────────────────────────────────────────────────────────────────────────
# Full ranking table
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📊 Full Strategy Rankings"):
    rank_df = pd.DataFrame([
        {'Rank': i+1, 'Strategy': n, 'Score': s,
         'Bias': STRATEGY_PROFILES[n]['bias'],
         'Premium Type': STRATEGY_PROFILES[n]['premium_type'],
         'Risk Defined': '✅' if STRATEGY_PROFILES[n]['risk_defined'] else '❌'}
        for i, (n, s, _) in enumerate(scored)
    ])
    st.dataframe(
        rank_df.style.format({'Score': '{:.1f}'})
                .background_gradient(subset=['Score'], cmap='RdYlGn'),
        use_container_width=True, hide_index=True
    )
