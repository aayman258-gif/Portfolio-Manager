"""
Regime-Aware Portfolio Manager
Section 2: Regime Detection + Market Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from calculations.regime_detector import RegimeDetector
from data.market_data import MarketDataLoader
from utils.carbon_theme import apply_carbon_theme, carbon_plotly_layout
from utils.theme import chart_style_toggle


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
    page_icon="🌍",
    layout="wide"
)
apply_carbon_theme()

st.title("🌍 Market Regime Dashboard")
st.markdown("**Section 2:** Detect current market regime and display macro indicators")

# Sidebar controls
st.sidebar.header("Market Settings")
index_ticker = st.sidebar.selectbox(
    "Market Index",
    options=['SPY', 'QQQ', 'IWM', 'DIA'],
    index=0,
    help="Choose market index to analyze"
)

period = st.sidebar.selectbox(
    "Time Period",
    options=['1y', '2y', '3y', '5y'],
    index=1
)

# Chart style selector
chart_style = chart_style_toggle("regime_chart_style")

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


try:
    spy_data, spy_prices, vix_prices = get_market_data(index_ticker, period)

    # Detect regime
    regime, signals = detector.classify_regime(spy_prices, vix_prices)

    # Current state
    current_regime = regime.iloc[-1]
    current_price = spy_prices.iloc[-1]
    current_vix = vix_prices.iloc[-1]
    current_vol = signals['realized_vol'].iloc[-1]
    current_trend = signals['trend'].iloc[-1]

    # Find previous non-Unknown regime (walk backwards from second-to-last point)
    prev_regime = None
    prev_regime_date = None
    for i in range(len(regime) - 2, -1, -1):
        if regime.iloc[i] != 'Unknown':
            prev_regime = regime.iloc[i]
            prev_regime_date = regime.index[i]
            break

    # Get regime description
    regime_info = detector.get_regime_description(current_regime)

    # Regime change callout
    if (
        prev_regime is not None
        and current_regime != 'Unknown'
        and prev_regime != current_regime
    ):
        change_summary = build_regime_change_summary(prev_regime, current_regime, signals)
        _rc = {
            'Low Vol': '🟢', 'High Vol': '🔴',
            'Trending': '🔵', 'Mean Reversion': '🟡', 'Unknown': '⚪'
        }
        old_emoji = _rc.get(prev_regime, '⚪')
        new_emoji = _rc.get(current_regime, '⚪')

        why_bullets = ''.join(f'<li>{r}</li>' for r in change_summary['why'])
        st.markdown(
            f"""
            <div style="background-color: rgba(245,158,11,0.10); border-left: 3px solid #f59e0b;
                        padding: 16px 20px; border-radius: 8px; margin-bottom: 16px;">
                <h4 style="margin: 0 0 8px 0; color: #f59e0b; font-family: 'Palatino Linotype',serif;">⚠️ Regime Change Detected</h4>
                <p style="margin: 0 0 8px 0; font-size: 16px;">
                    {old_emoji} <strong>{prev_regime}</strong> &rarr;
                    {new_emoji} <strong>{current_regime}</strong>
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
    st.subheader(f"📊 Current Market Regime: {index_ticker}")

    regime_colors = {
        'Low Vol': '🟢',
        'High Vol': '🔴',
        'Trending': '🔵',
        'Mean Reversion': '🟡',
        'Unknown': '⚪'
    }

    emoji = regime_colors.get(current_regime, '⚪')

    # Large regime indicator
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: #1a1a1a; border-radius: 8px; border: 1px solid #2a2a2a; margin-bottom: 20px;">
        <div style="font-size: 60px; margin: 0;">{emoji}</div>
        <div style="font-size: 1.6rem; font-style: italic; font-family: 'Palatino Linotype',serif; color: #22d3ee; margin: 8px 0;">{current_regime}</div>
        <div style="font-size: 0.9rem; color: #888888; font-family: 'Palatino Linotype',serif;">{regime_info['description']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Regime Details
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Regime Characteristics")
        st.info(regime_info['characteristics'])

        st.markdown("### 💼 Portfolio Implications")
        st.warning(regime_info['portfolio_implications'])

    with col2:
        st.markdown("### 🎯 Recommended Strategy")
        st.success(regime_info['strategy'])

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Risk Level", regime_info['risk_level'])
        with col_b:
            st.metric("Recommended Exposure", regime_info['recommended_exposure'])

    # Market Metrics
    st.subheader("📈 Market Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            f"{index_ticker} Price",
            f"${market_stats['current_price']:.2f}",
            f"{market_stats['returns_1d']:+.2f}% (1D)"
        )

    with col2:
        vix_color = "🔴" if current_vix > 30 else "🟡" if current_vix > 20 else "🟢"
        st.metric(
            f"{vix_color} VIX",
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
    st.subheader("📊 Performance Snapshot")

    perf_data = {
        'Period': ['1 Day', '5 Days', '1 Month', '3 Months', 'YTD'],
        'Return': [
            f"{market_stats['returns_1d']:+.2f}%",
            f"{market_stats['returns_5d']:+.2f}%",
            f"{market_stats['returns_1m']:+.2f}%",
            f"{market_stats['returns_3m']:+.2f}%",
            f"{market_stats['returns_ytd']:+.2f}%"
        ]
    }

    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # Technical Indicators
    col1, col2 = st.columns(2)

    with col1:
        ma_status = "✅ Above" if market_stats['above_ma50'] else "⚠️ Below"
        st.metric(
            "50-Day MA",
            f"${market_stats['ma_50']:.2f}",
            ma_status
        )

    with col2:
        ma_status = "✅ Above" if market_stats['above_ma200'] else "⚠️ Below"
        st.metric(
            "200-Day MA",
            f"${market_stats['ma_200']:.2f}",
            ma_status
        )

    # Regime Chart
    st.subheader("📈 Market Regime Over Time")

    # Color map for regimes
    color_map = {
        'Low Vol':       'rgba(34,197,94,0.18)',
        'High Vol':      'rgba(239,68,68,0.18)',
        'Trending':      'rgba(74,158,255,0.18)',
        'Mean Reversion':'rgba(245,158,11,0.18)',
        'Unknown':       'rgba(107,122,143,0.06)'
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
    st.subheader("📊 Regime Distribution")

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
            marker_color=['green' if r == 'Low Vol' else 'red' if r == 'High Vol' else 'blue' if r == 'Trending' else 'orange' for r in stats_df.index]
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
            emoji = regime_colors.get(regime_name, '⚪')
            st.markdown(f"{emoji} **{regime_name}**: {row['percentage']:.1f}% ({row['count']} days)")

    # Sector Performance
    st.subheader("📊 Sector Performance (1 Month)")

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
    with st.expander("🔄 Regime Transition History"):
        transitions = []
        prev_r = None
        for date, r in regime.items():
            if r != 'Unknown' and r != prev_r:
                if prev_r is not None:
                    transitions.append((date, r))
                prev_r = r

        if transitions:
            st.markdown("**Last 5 regime transitions (most recent first):**")
            _rc2 = {
                'Low Vol': '🟢', 'High Vol': '🔴',
                'Trending': '🔵', 'Mean Reversion': '🟡', 'Unknown': '⚪'
            }
            for date, r in reversed(transitions[-5:]):
                emoji = _rc2.get(r, '⚪')
                st.markdown(f"{emoji} **{date.strftime('%Y-%m-%d')}** — entered **{r}** regime")
        else:
            st.info("No regime transitions detected in the selected period.")

    # Regime Legend
    with st.expander("📚 Regime Definitions & Portfolio Guidance"):
        st.markdown("""
        ## Regime Definitions

        ### 🟢 Low Vol
        **Characteristics:** Low volatility, range-bound markets, calm conditions
        - **Portfolio Strategy:** High equity exposure (80-100%)
        - **Positioning:** Quality value stocks, dividend growth, defensive sectors
        - **Actions:** Buy quality on dips, income generation strategies

        ### 🔴 High Vol
        **Characteristics:** High volatility expansion, elevated uncertainty, fear
        - **Portfolio Strategy:** Reduced equity exposure (40-60%)
        - **Positioning:** Increase cash/bonds, focus on quality/defensives
        - **Actions:** Preserve capital, defensive positioning, wait for opportunities

        ### 🔵 Trending
        **Characteristics:** Strong directional movement, clear momentum
        - **Portfolio Strategy:** Medium-high equity exposure (70-90%)
        - **Positioning:** Momentum stocks, sector rotation, growth tilt if uptrend
        - **Actions:** Follow the trend, avoid fighting direction

        ### 🟡 Mean Reversion
        **Characteristics:** Range-bound with elevated volatility, choppy markets
        - **Portfolio Strategy:** Medium equity exposure (60-80%)
        - **Positioning:** Contrarian plays, buy oversold quality
        - **Actions:** Range trading, sell strength/buy weakness, patience

        ---

        ## How to Use This Information

        1. **Check current regime** at the top of the dashboard
        2. **Adjust your portfolio exposure** based on regime recommendations
        3. **Review sector performance** to identify strength/weakness
        4. **Monitor regime transitions** on the chart - when regimes shift, rebalance
        5. **Combine with fundamental analysis** (Section 3) for position-level decisions

        **Key Principle:** Market regime drives top-down allocation. Fundamentals drive bottom-up selection.
        """)

except Exception as e:
    st.error(f"Error loading market data: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
    st.info("Try a different index or time period.")
