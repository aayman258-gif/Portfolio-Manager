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


# Page config
st.set_page_config(
    page_title="Market Regime Dashboard",
    page_icon="🌍",
    layout="wide"
)

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

    # Get regime description
    regime_info = detector.get_regime_description(current_regime)

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
    <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="font-size: 60px; margin: 0;">{emoji}</h1>
        <h2 style="margin: 10px 0;">{current_regime}</h2>
        <p style="font-size: 18px; color: #666;">{regime_info['description']}</p>
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
        'Low Vol': 'rgba(0,255,0,0.2)',
        'High Vol': 'rgba(255,0,0,0.2)',
        'Trending': 'rgba(0,0,255,0.2)',
        'Mean Reversion': 'rgba(255,165,0,0.2)',
        'Unknown': 'rgba(128,128,128,0.1)'
    }

    # Create chart with regime backgrounds
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'{index_ticker} Price with Regime Overlay',
            'VIX (Volatility Index)',
            'Realized Volatility'
        ),
        row_heights=[0.5, 0.25, 0.25]
    )

    # Plot 1: Price with regime backgrounds
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals['price'],
            name=f'{index_ticker} Price',
            line=dict(color='black', width=2)
        ),
        row=1, col=1
    )

    # Add regime backgrounds
    for regime_name, color in color_map.items():
        if regime_name == 'Unknown':
            continue
        regime_periods = signals[signals['regime'] == regime_name]
        if len(regime_periods) > 0:
            regime_periods['group'] = (regime_periods.index.to_series().diff() > pd.Timedelta(days=2)).cumsum()
            for _, group in regime_periods.groupby('group'):
                if len(group) > 1:
                    fig.add_vrect(
                        x0=group.index[0],
                        x1=group.index[-1],
                        fillcolor=color,
                        opacity=1,
                        layer="below",
                        line_width=0,
                        row=1, col=1
                    )

    # Plot 2: VIX
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals['vix'],
            name='VIX',
            line=dict(color='purple', width=1.5)
        ),
        row=2, col=1
    )

    fig.add_hline(y=20, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)

    # Plot 3: Realized Vol
    fig.add_trace(
        go.Scatter(
            x=signals.index,
            y=signals['realized_vol'] * 100,
            name='Realized Vol',
            line=dict(color='blue', width=1.5)
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=900,
        showlegend=False,
        hovermode='x unified',
        template='plotly_white'
    )

    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="VIX", row=2, col=1)
    fig.update_yaxes(title_text="Vol (%)", row=3, col=1)

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
            template='plotly_white'
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
            template='plotly_white'
        )

        fig_sectors.add_hline(y=0, line_dash="dash", line_color="gray")

        st.plotly_chart(fig_sectors, use_container_width=True)

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
