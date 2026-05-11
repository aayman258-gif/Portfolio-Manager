"""
Regime-Aware Portfolio Manager
Section 2: Regime Detection + Market Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from calculations.regime_detector import RegimeDetector
from data.market_data import MarketDataLoader
from data.fred_client import get_all_macro, align_macro_to_index
from utils.carbon_theme import (
    apply_carbon_theme, carbon_plotly_layout, flex_table,
    page_header, top_nav, ACCENT, LOSS, AMBER, GAIN, BG, BORDER, DIM,
)


def build_regime_change_summary(old_regime: str, new_regime: str, signals: 'pd.DataFrame') -> dict:
    """
    Build human-readable reasons for a regime change.
    Pure Python — no API calls.

    Returns:
        dict with keys 'why' (list of strings) and 'adjustment' (str)
    """
    why = []

    vol_pct = signals['vol_percentile'].iloc[-1]
    entropy_val = signals['entropy'].iloc[-1]
    entropy_median = signals['entropy'].median()
    trend_val = signals['trend'].iloc[-1]

    if not pd.isna(vol_pct):
        if vol_pct < 0.3:
            why.append(
                f"Volatility has dropped to the bottom 30% of its historical range "
                f"(current percentile: {vol_pct:.0%})"
            )
        elif vol_pct > 0.7:
            why.append(
                f"Volatility has expanded to the top 30% of its historical range "
                f"(current percentile: {vol_pct:.0%})"
            )
        else:
            why.append(
                f"Volatility is at the {vol_pct:.0%} percentile — within moderate historical range"
            )

    if not pd.isna(entropy_val) and not pd.isna(entropy_median):
        if entropy_val > entropy_median:
            why.append("Market entropy is above median — elevated randomness and uncertainty detected")
        else:
            why.append("Market entropy is below median — more orderly, directional price action")

    if not pd.isna(trend_val):
        if trend_val > 0.02:
            why.append("Strong upward trend detected: 50-day MA slope exceeds +2% threshold")
        elif trend_val < -0.02:
            why.append("Strong downward trend detected: 50-day MA slope below -2% threshold")
        else:
            why.append("No strong directional trend — market is moving sideways")

    # Pull portfolio adjustment from new regime description
    _det = RegimeDetector()
    desc = _det.get_regime_description(new_regime)
    adjustment = desc.get('portfolio_implications', 'Adjust positioning based on new regime.')

    return {'why': why, 'adjustment': adjustment}


# Page config
st.set_page_config(
    page_title="Market Regime Dashboard",
    page_icon="◈",
    layout="wide"
)
apply_carbon_theme()
top_nav("Markets")

page_header("Market Regime Dashboard", "Detect current market regime and display macro indicators")

# Inline controls
_mc1, _mc2, _mc3 = st.columns([2, 2, 6])
with _mc1:
    index_ticker = st.selectbox(
        "Market Index", options=['SPY', 'QQQ', 'IWM', 'DIA'], index=0,
    )
with _mc2:
    period = st.selectbox(
        "Time Period", options=['1y', '2y', '3y', '5y'], index=1,
    )

# Chart style selector (inline — no sidebar)
with _mc3:
    _cs_choice = st.radio(
        "Chart Style", ["Line", "Candlestick"],
        horizontal=True, key="regime_chart_style",
    )
chart_style = "candlestick" if "Candlestick" in _cs_choice else "line"

# Initialize
loader = MarketDataLoader()
detector = RegimeDetector(lookback_vol=20, lookback_trend=50)

# Load data
@st.cache_data
def get_market_data(ticker, period):
    """Load and cache market data"""
    with st.spinner(f"Loading {ticker} data..."):
        spy_data = loader.load_index_data(ticker, period)
        vix_data = loader.load_vix_data(period)
        spy_prices, vix_prices = loader.align_data(spy_data, vix_data)
    return spy_data, spy_prices, vix_prices


@st.cache_data(ttl=14400, show_spinner=False)
def get_fred_data():
    try:
        return get_all_macro()
    except Exception:
        return {}


try:
    spy_data, spy_prices, vix_prices = get_market_data(index_ticker, period)

    # Load FRED macro data (cached 4h)
    with st.spinner("Loading macro indicators..."):
        _macro_raw = get_fred_data()

    _macro_df = align_macro_to_index(_macro_raw, spy_prices.index) if _macro_raw else pd.DataFrame()

    # Detect regime (macro-enhanced if FRED available)
    regime, signals = detector.classify_regime(spy_prices, vix_prices, macro_df=_macro_df if not _macro_df.empty else None)

    # Current state
    current_regime = regime.iloc[-1]
    current_price = spy_prices.iloc[-1]
    current_vix = vix_prices.iloc[-1]
    current_vol = signals['realized_vol'].iloc[-1]
    current_trend = signals['trend'].iloc[-1]

    # Per-regime confidence scores for the latest bar
    confidence_scores = detector.get_current_confidence(signals)
    best_confidence   = confidence_scores.get('best_confidence', 0.0)

    # Find previous non-Unknown / non-Uncertain regime
    prev_regime = None
    prev_regime_date = None
    for i in range(len(regime) - 2, -1, -1):
        if regime.iloc[i] not in ('Unknown', 'Uncertain'):
            prev_regime = regime.iloc[i]
            prev_regime_date = regime.index[i]
            break

    # Get regime description
    regime_info = detector.get_regime_description(current_regime)

    # Regime change callout
    if (
        prev_regime is not None
        and current_regime not in ('Unknown', 'Uncertain')
        and prev_regime != current_regime
    ):
        change_summary = build_regime_change_summary(prev_regime, current_regime, signals)
        why_bullets = ''.join(f'<li>{r}</li>' for r in change_summary['why'])
        st.markdown(
            f"""
            <div style="background-color: rgba(245,158,11,0.10); border-left: 3px solid #f59e0b;
                        padding: 16px 20px; border-radius: 8px; margin-bottom: 16px;">
                <h4 style="margin: 0 0 8px 0; color: #f59e0b; font-family: 'Palatino Linotype',serif;">Regime Change Detected</h4>
                <p style="margin: 0 0 8px 0; font-size: 16px;">
                    <strong>{prev_regime}</strong> &rarr;
                    <strong>{current_regime}</strong>
                </p>
                <p style="margin: 0 0 4px 0;"><strong>Why it changed:</strong></p>
                <ul style="margin: 0 0 8px 0;">{why_bullets}</ul>
                <p style="margin: 0;"><strong>Recommended portfolio adjustment:</strong>
                    {change_summary['adjustment']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Market stats
    market_stats = loader.get_current_market_stats(spy_data, current_vix)

    # Display Current Regime
    st.subheader(f"Current Market Regime: {index_ticker}")

    regime_colors = {
        'Risk-On':        '#22c55e',
        'Caution':        '#f59e0b',
        'High Volatility':'#ef4444',
        'Stagflation':    '#f97316',
        'Recession':      '#dc2626',
        'Mean Reversion': '#3b82f6',
        'Uncertain':      '#6b7280',
        'Unknown':        '#9ca3af',
        # legacy names (fallback)
        'Low Vol':        '#22c55e',
        'High Vol':       '#ef4444',
        'Trending':       '#3b82f6',
    }

    # Large regime indicator with confidence badge
    conf_pct = int(best_confidence * 100)
    conf_color = '#22d3ee' if conf_pct >= 60 else '#f59e0b' if conf_pct >= 35 else '#fb7185'
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: #1a1a1a; border-radius: 8px; border: 1px solid #2a2a2a; margin-bottom: 20px;">
        <div style="font-size: 1.6rem; font-style: italic; font-family: 'Palatino Linotype',serif; color: #22d3ee; margin: 8px 0;">{current_regime}</div>
        <div style="font-size: 0.9rem; color: #888888; font-family: 'Palatino Linotype',serif; margin-bottom: 8px;">{regime_info['description']}</div>
        <div style="display:inline-block; background:{conf_color}22; border:1px solid {conf_color}66; border-radius:6px; padding:3px 12px;">
            <span style="font-family:'Palatino Linotype',serif; font-size:0.78rem; color:{conf_color};">
                Confidence: {conf_pct}%
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence breakdown bars
    _macro_label = "6-signal macro-enhanced" if confidence_scores.get("macro_available") else "3-signal price-only (FRED unavailable)"
    with st.expander(f"Regime Confidence Breakdown ({_macro_label})", expanded=(current_regime == 'Uncertain')):
        _regime_order = ['Risk-On', 'Caution', 'High Volatility', 'Stagflation', 'Recession', 'Mean Reversion']
        for r_name in _regime_order:
            r_score  = confidence_scores.get(r_name, 0)
            bar_pct  = int(r_score * 100)
            bar_color = '#22d3ee' if r_name == current_regime else '#555555'
            st.markdown(
                f'<div style="margin-bottom:6px;">'
                f'<span style="font-family:\'Palatino Linotype\',serif;font-size:0.78rem;'
                f'color:#888888;display:inline-block;width:150px;">{r_name}</span>'
                f'<span style="display:inline-block;height:10px;width:{bar_pct * 2}px;'
                f'background:{bar_color};border-radius:3px;vertical-align:middle;"></span>'
                f'<span style="font-family:\'Palatino Linotype\',serif;font-size:0.75rem;'
                f'color:{bar_color};margin-left:8px;">{bar_pct}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.caption(f"Threshold to assign a regime: {int(0.40*100)}%")

    # Regime Details
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Regime Characteristics")
        st.info(regime_info['characteristics'])

        st.markdown("### Portfolio Implications")
        st.warning(regime_info['portfolio_implications'])

    with col2:
        st.markdown("### Recommended Strategy")
        st.success(regime_info['strategy'])

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Risk Level", regime_info['risk_level'])
        with col_b:
            st.metric("Recommended Exposure", regime_info['recommended_exposure'])

    # Market Metrics
    st.subheader("Market Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            f"{index_ticker} Price",
            f"${market_stats['current_price']:.2f}",
            f"{market_stats['returns_1d']:+.2f}% (1D)"
        )

    with col2:
        vix_color_text = "High" if current_vix > 30 else "Moderate" if current_vix > 20 else "Low"
        st.metric(
            f"VIX ({vix_color_text})",
            f"{current_vix:.2f}",
            "Fear Index"
        )

    with col3:
        st.metric(
            "Realized Vol (20D)",
            f"{market_stats['vol_20d']:.1f}%"
        )

    with col4:
        st.metric(
            "From 52W High",
            f"{market_stats['distance_from_high']:+.1f}%"
        )

    # Performance Table
    st.subheader("Performance Snapshot")

    perf_df = pd.DataFrame({
        'Period': ['1 Day', '5 Days', '1 Month', '3 Months', 'YTD'],
        'Return': [
            market_stats['returns_1d'], market_stats['returns_5d'],
            market_stats['returns_1m'], market_stats['returns_3m'],
            market_stats['returns_ytd'],
        ],
    })
    flex_table(perf_df, columns=[
        {"key": "Period", "label": "Period", "width": "55%", "align": "left"},
        {"key": "Return", "label": "Return (%)", "width": "45%", "align": "right",
         "fmt": lambda x: f"{x:+.2f}%", "numeric": True, "color_scale": "rg"},
    ], key="markets_perf")

    # Technical Indicators
    col1, col2 = st.columns(2)
    with col1:
        ma_status = "Above" if market_stats['above_ma50'] else "Below"
        st.metric("50-Day MA", f"${market_stats['ma_50']:.2f}", ma_status)
    with col2:
        ma_status = "Above" if market_stats['above_ma200'] else "Below"
        st.metric("200-Day MA", f"${market_stats['ma_200']:.2f}", ma_status)

    # ─────────────────────────────────────────────────────────────────────────
    # MACRO DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    _has_macro = bool(_macro_raw) and not _macro_df.empty

    st.divider()
    st.subheader("Macro Indicators")

    if not _has_macro:
        st.info("FRED macro data unavailable — regime running on price signals only.")
    else:
        def _latest_val(series_key: str):
            s = _macro_raw.get(series_key, pd.Series(dtype=float))
            return float(s.iloc[-1]) if not s.empty else None

        _spread   = _latest_val("T10Y2Y")
        _dgs10    = _latest_val("DGS10")
        _dgs2     = _latest_val("DGS2")
        _hy       = _latest_val("BAMLH0A0HYM2")
        _fed      = _latest_val("FEDFUNDS")
        _cpi      = _latest_val("CPI_YOY")
        _real     = _latest_val("REAL_RATE")
        _unemp    = _latest_val("UNRATE")

        # ── Macro metric cards ────────────────────────────────────────────────
        _mc1, _mc2, _mc3, _mc4, _mc5, _mc6, _mc7, _mc8 = st.columns(8)
        def _pct_delta(val, neutral, invert=False):
            if val is None: return None
            d = val - neutral
            return f"{'+' if d>=0 else ''}{d:.2f}pp" if not invert else f"{'+' if -d>=0 else ''}{-d:.2f}pp"

        with _mc1:
            _spread_color = "normal" if _spread and _spread > 0 else "inverse"
            st.metric("10Y-2Y Spread", f"{_spread:.2f}%" if _spread is not None else "N/A",
                      help="Positive = normal curve · Negative = inverted (recession signal)")
        with _mc2:
            st.metric("10Y Yield", f"{_dgs10:.2f}%" if _dgs10 is not None else "N/A")
        with _mc3:
            st.metric("2Y Yield", f"{_dgs2:.2f}%" if _dgs2 is not None else "N/A")
        with _mc4:
            hy_note = "Stressed" if _hy and _hy > 5.0 else "Elevated" if _hy and _hy > 3.5 else "Tight"
            st.metric("HY Spread (OAS)", f"{_hy:.2f}%" if _hy is not None else "N/A",
                      hy_note, help="ICE BofA US HY Index option-adjusted spread")
        with _mc5:
            st.metric("Fed Funds", f"{_fed:.2f}%" if _fed is not None else "N/A",
                      help="Effective federal funds rate")
        with _mc6:
            cpi_delta = f"{_cpi-2:.2f}pp vs 2% target" if _cpi is not None else None
            st.metric("CPI YoY", f"{_cpi:.2f}%" if _cpi is not None else "N/A", cpi_delta)
        with _mc7:
            real_note = "Restrictive" if _real and _real > 1.0 else "Neutral" if _real and _real > -0.5 else "Accommodative"
            st.metric("Real Rate", f"{_real:.2f}%" if _real is not None else "N/A", real_note,
                      help="Fed Funds minus CPI YoY — positive = restrictive policy")
        with _mc8:
            st.metric("Unemployment", f"{_unemp:.1f}%" if _unemp is not None else "N/A")

        # ── Charts: Yield Curve + Credit Spreads ──────────────────────────────
        _ch1, _ch2 = st.columns(2)

        with _ch1:
            st.markdown("##### Yield Curve (10Y vs 2Y)")
            _t10 = _macro_raw.get("DGS10", pd.Series(dtype=float))
            _t2  = _macro_raw.get("DGS2",  pd.Series(dtype=float))
            _spr = _macro_raw.get("T10Y2Y",pd.Series(dtype=float))
            if not _spr.empty:
                fig_yc = go.Figure()
                if not _t10.empty:
                    fig_yc.add_trace(go.Scatter(x=_t10.index, y=_t10,
                        name="10Y Yield", line=dict(color=ACCENT, width=1.6)))
                if not _t2.empty:
                    fig_yc.add_trace(go.Scatter(x=_t2.index, y=_t2,
                        name="2Y Yield", line=dict(color=AMBER, width=1.6)))
                # Spread as area on secondary axis
                fig_yc.add_trace(go.Scatter(
                    x=_spr.index, y=_spr,
                    name="10Y-2Y Spread", line=dict(color='#a78bfa', width=1.4),
                    fill='tozeroy',
                    fillcolor='rgba(167,139,250,0.12)',
                    yaxis='y2',
                ))
                fig_yc.add_hline(y=0, line_dash="dash", line_color=LOSS, opacity=0.5, yref='y2')
                fig_yc.update_layout(
                    **carbon_plotly_layout(height=300),
                    legend=dict(orientation='h', y=-0.25),
                )
                fig_yc.update_layout(
                    yaxis=dict(title="Yield (%)", gridcolor='rgba(107,122,143,0.15)'),
                    yaxis2=dict(title="Spread (%)", overlaying='y', side='right',
                                gridcolor='rgba(107,122,143,0.08)'),
                )
                st.plotly_chart(fig_yc, use_container_width=True)
            else:
                st.info("Yield curve data unavailable.")

        with _ch2:
            st.markdown("##### HY Credit Spreads (OAS)")
            _hy_s = _macro_raw.get("BAMLH0A0HYM2", pd.Series(dtype=float))
            if not _hy_s.empty:
                _hy_color = [LOSS if v > 5.0 else AMBER if v > 3.5 else GAIN for v in _hy_s]
                fig_hy = go.Figure()
                fig_hy.add_trace(go.Scatter(
                    x=_hy_s.index, y=_hy_s,
                    name="HY OAS", line=dict(color=LOSS, width=1.6),
                    fill='tozeroy', fillcolor='rgba(251,113,133,0.08)',
                ))
                fig_hy.add_hline(y=3.5, line_dash="dash", line_color=AMBER, opacity=0.6,
                                  annotation_text="Elevated (3.5%)", annotation_position="bottom right")
                fig_hy.add_hline(y=5.0, line_dash="dash", line_color=LOSS, opacity=0.6,
                                  annotation_text="Stressed (5.0%)", annotation_position="bottom right")
                fig_hy.update_layout(**carbon_plotly_layout(height=300))
                fig_hy.update_yaxes(title_text="OAS (%)", gridcolor='rgba(107,122,143,0.15)')
                st.plotly_chart(fig_hy, use_container_width=True)
            else:
                st.info("Credit spread data unavailable.")

        # ── Charts: Inflation + Fed Policy ────────────────────────────────────
        _ch3, _ch4 = st.columns(2)

        with _ch3:
            st.markdown("##### CPI Inflation (YoY %)")
            _cpi_s = _macro_raw.get("CPI_YOY", pd.Series(dtype=float))
            if not _cpi_s.empty:
                fig_cpi = go.Figure()
                fig_cpi.add_trace(go.Scatter(
                    x=_cpi_s.index, y=_cpi_s,
                    name="CPI YoY", line=dict(color=AMBER, width=1.8),
                    fill='tozeroy', fillcolor='rgba(245,158,11,0.10)',
                ))
                fig_cpi.add_hline(y=2.0, line_dash="dash", line_color=GAIN, opacity=0.7,
                                   annotation_text="Fed Target (2%)", annotation_position="bottom right")
                fig_cpi.update_layout(**carbon_plotly_layout(height=300))
                fig_cpi.update_yaxes(title_text="CPI YoY (%)", gridcolor='rgba(107,122,143,0.15)')
                st.plotly_chart(fig_cpi, use_container_width=True)
            else:
                st.info("CPI data unavailable.")

        with _ch4:
            st.markdown("##### Fed Funds vs Real Rate")
            _fed_s  = _macro_raw.get("FEDFUNDS",  pd.Series(dtype=float))
            _real_s = _macro_raw.get("REAL_RATE",  pd.Series(dtype=float))
            if not _fed_s.empty:
                fig_fed = go.Figure()
                fig_fed.add_trace(go.Scatter(
                    x=_fed_s.index, y=_fed_s,
                    name="Fed Funds", line=dict(color=ACCENT, width=1.8),
                ))
                if not _real_s.empty:
                    fig_fed.add_trace(go.Scatter(
                        x=_real_s.index, y=_real_s,
                        name="Real Rate", line=dict(color='#a78bfa', width=1.6, dash='dot'),
                        fill='tozeroy', fillcolor='rgba(167,139,250,0.08)',
                    ))
                fig_fed.add_hline(y=0, line_dash="dash", line_color=BORDER, opacity=0.5)
                fig_fed.update_layout(
                    **carbon_plotly_layout(height=300),
                    legend=dict(orientation='h', y=-0.25),
                )
                fig_fed.update_yaxes(title_text="Rate (%)", gridcolor='rgba(107,122,143,0.15)')
                st.plotly_chart(fig_fed, use_container_width=True)
            else:
                st.info("Fed policy data unavailable.")

        # ── Signal breakdown table ────────────────────────────────────────────
        st.markdown("##### Regime Signal Readings")
        _sig_last = signals.iloc[-1]
        _sig_rows = []

        def _sval(col): return signals[col].iloc[-1] if col in signals.columns else None

        _sig_rows = [
            {"Signal": "Realized Volatility (20d)",   "Value": f"{_sval('realized_vol')*100:.1f}%" if _sval('realized_vol') else "N/A",
             "Percentile": f"{_sval('vol_percentile')*100:.0f}th" if _sval('vol_percentile') else "N/A",
             "Reading": "Stressed" if (_sval('vol_percentile') or 0) > 0.7 else "Calm" if (_sval('vol_percentile') or 0) < 0.3 else "Moderate"},
            {"Signal": "Price Trend (50d MA slope)",  "Value": f"{_sval('trend'):+.3f}" if _sval('trend') else "N/A",
             "Percentile": "—",
             "Reading": "Uptrend" if (_sval('trend') or 0) > 0.01 else "Downtrend" if (_sval('trend') or 0) < -0.01 else "Sideways"},
            {"Signal": "Yield Curve (10Y-2Y)",         "Value": f"{_spread:.2f}%" if _spread else "N/A",
             "Percentile": f"{_sval('curve_pct')*100:.0f}th" if _sval('curve_pct') else "N/A",
             "Reading": "Inverted" if _spread and _spread < 0 else "Flat" if _spread and _spread < 0.5 else "Steep"},
            {"Signal": "HY Credit Spreads (OAS)",      "Value": f"{_hy:.2f}%" if _hy else "N/A",
             "Percentile": f"{_sval('credit_pct')*100:.0f}th" if _sval('credit_pct') else "N/A",
             "Reading": "Stressed" if _hy and _hy > 5.0 else "Elevated" if _hy and _hy > 3.5 else "Tight"},
            {"Signal": "CPI Inflation (YoY)",          "Value": f"{_cpi:.2f}%" if _cpi else "N/A",
             "Percentile": f"{_sval('infl_pct')*100:.0f}th" if _sval('infl_pct') else "N/A",
             "Reading": "High" if _cpi and _cpi > 4.0 else "Elevated" if _cpi and _cpi > 2.5 else "Low"},
            {"Signal": "Real Rate (Fed−CPI)",          "Value": f"{_real:.2f}%" if _real else "N/A",
             "Percentile": f"{_sval('rrate_pct')*100:.0f}th" if _sval('rrate_pct') else "N/A",
             "Reading": "Restrictive" if _real and _real > 1.0 else "Accommodative" if _real and _real < 0 else "Neutral"},
        ]
        flex_table(pd.DataFrame(_sig_rows), columns=[
            {"key": "Signal",     "label": "Signal",     "width": "36%", "align": "left"},
            {"key": "Value",      "label": "Current",    "width": "18%", "align": "right"},
            {"key": "Percentile", "label": "Percentile", "width": "18%", "align": "right"},
            {"key": "Reading",    "label": "Reading",    "width": "28%", "align": "left"},
        ], key="macro_signals")

    st.divider()

    # Regime Chart
    st.subheader("Market Regime Over Time")

    # Color map for regimes
    color_map = {
        'Risk-On':         'rgba(34,211,238,0.15)',
        'Caution':         'rgba(245,158,11,0.15)',
        'High Volatility': 'rgba(251,113,133,0.18)',
        'Stagflation':     'rgba(249,115,22,0.16)',
        'Recession':       'rgba(220,38,38,0.18)',
        'Mean Reversion':  'rgba(59,130,246,0.16)',
        'Uncertain':       'rgba(136,136,136,0.12)',
        'Unknown':         'rgba(136,136,136,0.06)',
        # legacy fallbacks
        'Low Vol':         'rgba(34,211,238,0.18)',
        'High Vol':        'rgba(251,113,133,0.18)',
        'Trending':        'rgba(34,211,238,0.10)',
    }

    # Build subplots — extra row for volume when candlestick is selected
    if chart_style == 'candlestick':
        n_rows = 4
        row_heights = [0.45, 0.15, 0.20, 0.20]
        subplot_titles = (
            f'{index_ticker} Price with Regime Overlay',
            'Volume',
            'VIX (Volatility Index)',
            'Realized Volatility'
        )
    else:
        n_rows = 3
        row_heights = [0.5, 0.25, 0.25]
        subplot_titles = (
            f'{index_ticker} Price with Regime Overlay',
            'VIX (Volatility Index)',
            'Realized Volatility'
        )

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )

    # Plot 1: Price trace (candlestick or line)
    if chart_style == 'candlestick':
        # Align OHLCV from spy_data onto signals index
        import numpy as np
        ohlcv = spy_data[['Open', 'High', 'Low', 'Close']].copy()
        ohlcv.index = pd.to_datetime(ohlcv.index).tz_localize(None)
        sig_idx = pd.to_datetime(signals.index).tz_localize(None)
        ohlcv = ohlcv.reindex(sig_idx, method='nearest')
        vol_series = spy_data['Volume'].copy()
        vol_series.index = pd.to_datetime(vol_series.index).tz_localize(None)
        vol_series = vol_series.reindex(sig_idx, method='nearest')

        fig.add_trace(
            go.Candlestick(
                x=signals.index,
                open=ohlcv['Open'],
                high=ohlcv['High'],
                low=ohlcv['Low'],
                close=ohlcv['Close'],
                name=index_ticker,
                increasing_line_color='#22d3ee',
                decreasing_line_color='#fb7185',
                increasing_fillcolor='rgba(34,211,238,0.6)',
                decreasing_fillcolor='rgba(251,113,133,0.6)',
            ),
            row=1, col=1
        )

        # Volume bars
        colors = ['#22d3ee' if c >= o else '#fb7185'
                  for c, o in zip(ohlcv['Close'].fillna(0), ohlcv['Open'].fillna(0))]
        fig.add_trace(
            go.Bar(
                x=signals.index,
                y=vol_series,
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        vix_row, vol_row = 3, 4
    else:
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals['price'],
                name=f'{index_ticker} Price',
                line=dict(color='#22d3ee', width=2),
                fill='tonexty',
            ),
            row=1, col=1
        )
        vix_row, vol_row = 2, 3

    # Regime background shading (row 1)
    for regime_name, color in color_map.items():
        if regime_name == 'Unknown':
            continue
        regime_periods = signals[signals['regime'] == regime_name].copy()
        if len(regime_periods) > 0:
            regime_periods['group'] = (
                regime_periods.index.to_series().diff() > pd.Timedelta(days=2)
            ).cumsum()
            for _, group in regime_periods.groupby('group'):
                if len(group) > 1:
                    fig.add_vrect(
                        x0=group.index[0], x1=group.index[-1],
                        fillcolor=color, opacity=1,
                        layer="below", line_width=0,
                        row=1, col=1
                    )

    # VIX subplot
    fig.add_trace(
        go.Scatter(
            x=signals.index, y=signals['vix'],
            name='VIX', line=dict(color='#b07eff', width=1.5)
        ),
        row=vix_row, col=1
    )
    fig.add_hline(y=20, line_dash="dash", line_color="#888888", opacity=0.6, row=vix_row, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#fb7185", opacity=0.6, row=vix_row, col=1)

    # Realized Vol subplot
    fig.add_trace(
        go.Scatter(
            x=signals.index, y=signals['realized_vol'] * 100,
            name='Realized Vol', line=dict(color='#4a9eff', width=1.5),
            fill='tozeroy', fillcolor='rgba(74,158,255,0.08)'
        ),
        row=vol_row, col=1
    )

    fig.update_layout(
        **carbon_plotly_layout(height=950, showlegend=False, hovermode='x unified'),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(title_text="Date", row=n_rows, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    if chart_style == 'candlestick':
        fig.update_yaxes(title_text="Volume", row=2, col=1, showticklabels=False)
    fig.update_yaxes(title_text="VIX", row=vix_row, col=1)
    fig.update_yaxes(title_text="Vol (%)", row=vol_row, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Regime Distribution
    st.subheader("Regime Distribution")

    stats_df = detector.get_regime_stats(regime)
    stats_df = stats_df[stats_df.index != 'Unknown']

    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar chart
        fig_dist = go.Figure(data=[go.Bar(
            x=stats_df.index,
            y=stats_df['percentage'],
            text=stats_df['percentage'],
            texttemplate='%{text:.1f}%',
            textposition='outside',
            marker_color=[
                '#22d3ee' if r == 'Risk-On' else
                '#f59e0b' if r == 'Caution' else
                '#fb7185' if r == 'High Volatility' else
                '#f97316' if r == 'Stagflation' else
                '#dc2626' if r == 'Recession' else
                '#3b82f6' if r == 'Mean Reversion' else
                '#6b7280'
                for r in stats_df.index
            ]
        )])

        fig_dist.update_layout(
            title=f"Time Spent in Each Regime ({period})",
            xaxis_title="Regime",
            yaxis_title="Percentage (%)",
            height=400,
            template='plotly_dark'
        )

        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.markdown("### Regime Stats")
        for regime_name, row in stats_df.iterrows():
            st.markdown(f"**{regime_name}**: {row['percentage']:.1f}% ({row['count']} days)")

    # Sector Performance
    st.subheader("Sector Performance (1 Month)")

    with st.spinner("Loading sector data..."):
        sector_perf = loader.get_sector_performance(period='1mo')

    if not sector_perf.empty:
        # Color code by performance
        colors = sector_perf['Return'].apply(lambda x: 'green' if x > 0 else 'red')

        fig_sectors = go.Figure(data=[go.Bar(
            x=sector_perf['Sector'],
            y=sector_perf['Return'],
            text=sector_perf['Return'],
            texttemplate='%{text:+.2f}%',
            textposition='outside',
            marker_color=colors
        )])

        fig_sectors.update_layout(
            title="Sector Returns (Past Month)",
            xaxis_title="Sector",
            yaxis_title="Return (%)",
            height=400,
            template='plotly_dark'
        )

        fig_sectors.add_hline(y=0, line_dash="dash", line_color="gray")

        st.plotly_chart(fig_sectors, use_container_width=True)

    # Regime Transition History
    with st.expander("Regime Transition History"):
        transitions = []
        prev_r = None
        for date, r in regime.items():
            if r != 'Unknown' and r != prev_r:
                if prev_r is not None:
                    transitions.append((date, r))
                prev_r = r

        if transitions:
            st.markdown("**Last 5 regime transitions (most recent first):**")
            for date, r in reversed(transitions[-5:]):
                st.markdown(f"**{date.strftime('%Y-%m-%d')}** — entered **{r}** regime")
        else:
            st.info("No regime transitions detected in the selected period.")

    # Regime Legend
    with st.expander("Regime Definitions & Portfolio Guidance"):
        st.markdown("""
        ## 6-Signal Macro-Enhanced Regime Model

        The regime detector scores six normalized signals (each 0→1 via rolling percentile)
        against ideal profiles for each regime. The best-matching regime is assigned when its
        confidence score exceeds the 40% threshold.

        **Signals:** Realized Volatility · Price Trend · Yield Curve (10Y-2Y) · HY Credit Spread ·
        CPI Inflation (YoY) · Real Rate (Fed Funds − CPI)

        ---

        ### Risk-On
        **Characteristics:** Low volatility, strong uptrend, steep yield curve, tight credit spreads,
        contained inflation, positive real rates.
        - **Strategy:** High equity exposure (85-100%). Cyclicals, growth, momentum.
        - **Options bias:** Sell premium, covered calls. IV typically low.

        ### Caution
        **Characteristics:** Moderate volatility, mixed trend, flattening curve, widening spreads.
        Transition zone — regime is shifting.
        - **Strategy:** Medium equity exposure (60-75%). Quality tilt, reduce beta.
        - **Options bias:** Protective puts on key positions. Collar strategies.

        ### High Volatility
        **Characteristics:** Elevated volatility spike, negative trend, widening credit spreads.
        Fear-driven sell-off or black-swan event.
        - **Strategy:** Defensive (40-60% equity). Cash, short-duration bonds, gold.
        - **Options bias:** Buy protection (puts/spreads). Sell into IV spikes.

        ### Stagflation
        **Characteristics:** Persistent above-target inflation + below-trend growth. Flat/inverted
        curve, accommodative real rates, moderate credit stress.
        - **Strategy:** Real assets (50-65% equity). Commodities, TIPS, energy, materials.
        - **Options bias:** Commodity call spreads. Avoid long-duration growth exposure.

        ### Recession
        **Characteristics:** Inverted yield curve, wide credit spreads, rising unemployment,
        declining trend. Classic leading-indicator configuration.
        - **Strategy:** Preserve capital (30-50% equity). Treasuries, defensives, cash.
        - **Options bias:** Long puts on cyclicals. Protective overlays across book.

        ### Mean Reversion
        **Characteristics:** Moderate signals across all dimensions — no regime is dominant.
        Markets choppy, range-bound.
        - **Strategy:** Balanced allocation (60-75% equity). Sector-neutral, quality tilt.
        - **Options bias:** Short straddles/strangles within range. Volatility-neutral.

        ---

        ## How to Use This Information

        1. **Check the current regime and confidence** at the top of this page
        2. **Review the Signal Readings table** — understand which signals are driving the call
        3. **Adjust portfolio exposure** per regime recommendations
        4. **Monitor the Regime Over Time chart** for transitions — rebalance when regime shifts
        5. **Watch macro charts** — yield curve, credit spreads, and real rates are leading indicators

        **Key Principle:** Regime drives top-down allocation. Fundamentals drive bottom-up selection.
        """)

except Exception as e:
    st.error(f"Error loading market data: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
    st.info("Try a different index or time period.")
