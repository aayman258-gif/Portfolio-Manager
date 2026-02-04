"""
Regime-Aware Portfolio Manager
Section 1: Portfolio Position Tracker
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data.portfolio_loader import PortfolioLoader
from calculations.performance import PerformanceAnalytics


# Page config
st.set_page_config(
    page_title="Portfolio Position Tracker",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Portfolio Position Tracker")
st.markdown("**Section 1:** Load and track current positions with basic P&L, holdings breakdown, and performance metrics")

# Initialize
loader = PortfolioLoader()
analytics = PerformanceAnalytics()

# Sidebar - Portfolio Input
st.sidebar.header("Portfolio Input")

input_method = st.sidebar.radio(
    "How to load positions?",
    options=["Use Sample Portfolio", "Upload CSV"],
    help="Sample portfolio for demo, or upload your own positions"
)

positions_df = None

if input_method == "Use Sample Portfolio":
    if st.sidebar.button("Load Sample Portfolio"):
        positions_df = loader.create_sample_portfolio()
        st.session_state['positions'] = positions_df
        st.sidebar.success("Sample portfolio loaded!")

elif input_method == "Upload CSV":
    st.sidebar.markdown("""
    **CSV Format:**
    ```
    ticker,shares,cost_basis,purchase_date
    AAPL,100,150.00,2023-01-15
    MSFT,50,300.00,2023-02-20
    ```
    """)

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            positions_df = loader.load_from_csv(uploaded_file)
            st.session_state['positions'] = positions_df
            st.sidebar.success(f"Loaded {len(positions_df)} positions!")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

# Check if we have positions (either from current load or session state)
if 'positions' in st.session_state:
    positions_df = st.session_state['positions']

if positions_df is not None and not positions_df.empty:

    # Fetch current prices
    with st.spinner("Fetching current market data..."):
        tickers = positions_df['ticker'].tolist()
        current_prices = loader.fetch_current_prices(tickers)

    # Calculate position metrics
    position_metrics = loader.calculate_position_metrics(positions_df, current_prices)

    # Get portfolio summary
    summary = loader.get_portfolio_summary(position_metrics)

    # Display Portfolio Summary
    st.subheader("📊 Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Value",
            f"${summary['total_value']:,.2f}",
            f"{summary['total_pnl_pct']:+.2f}%"
        )

    with col2:
        st.metric(
            "Total P&L",
            f"${summary['total_pnl']:,.2f}",
            f"{summary['total_pnl_pct']:+.2f}%"
        )

    with col3:
        st.metric(
            "Positions",
            summary['num_positions'],
            f"{summary['winners']}W / {summary['losers']}L"
        )

    with col4:
        st.metric(
            "Largest Position",
            summary['largest_position_ticker'],
            f"{summary['largest_position_weight']:.1f}%"
        )

    # Position Details Table
    st.subheader("📋 Position Details")

    # Format the display dataframe
    display_df = position_metrics[[
        'ticker', 'shares', 'cost_basis', 'current_price',
        'cost_value', 'current_value', 'total_pnl', 'total_pnl_pct', 'weight_pct'
    ]].copy()

    # Color code P&L
    def color_pnl(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
        return f'color: {color}'

    styled_df = display_df.style.format({
        'cost_basis': '${:.2f}',
        'current_price': '${:.2f}',
        'cost_value': '${:,.2f}',
        'current_value': '${:,.2f}',
        'total_pnl': '${:,.2f}',
        'total_pnl_pct': '{:+.2f}%',
        'weight_pct': '{:.2f}%'
    }).applymap(color_pnl, subset=['total_pnl', 'total_pnl_pct'])

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Holdings Breakdown
    st.subheader("🥧 Holdings Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        # Pie chart by value
        fig_pie = go.Figure(data=[go.Pie(
            labels=position_metrics['ticker'],
            values=position_metrics['current_value'],
            hole=0.3,
            textinfo='label+percent',
            marker=dict(
                colors=position_metrics['total_pnl'].apply(
                    lambda x: '#00CC00' if x > 0 else '#FF6B6B' if x < 0 else '#95A5A6'
                )
            )
        )])

        fig_pie.update_layout(
            title="Portfolio Allocation by Value",
            height=400
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Bar chart P&L
        colors = position_metrics['total_pnl'].apply(
            lambda x: 'green' if x > 0 else 'red' if x < 0 else 'gray'
        )

        fig_bar = go.Figure(data=[go.Bar(
            x=position_metrics['ticker'],
            y=position_metrics['total_pnl'],
            marker_color=colors,
            text=position_metrics['total_pnl'],
            texttemplate='$%{text:,.0f}',
            textposition='outside'
        )])

        fig_bar.update_layout(
            title="P&L by Position",
            xaxis_title="Ticker",
            yaxis_title="P&L ($)",
            height=400
        )

        fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")

        st.plotly_chart(fig_bar, use_container_width=True)

    # Performance Analytics
    st.subheader("📈 Performance Analytics")

    # Fetch historical data
    with st.spinner("Calculating performance metrics..."):
        # Get historical data for past year
        start_date = datetime.now() - timedelta(days=365)
        historical_data = loader.fetch_historical_data(tickers, start_date=start_date)

        # Calculate weights based on current allocation
        weights = {}
        total_val = position_metrics['current_value'].sum()
        for _, row in position_metrics.iterrows():
            weights[row['ticker']] = row['current_value'] / total_val

        # Calculate portfolio returns
        portfolio_returns = analytics.calculate_returns(historical_data, weights)

        if not portfolio_returns.empty and len(portfolio_returns) > 20:
            # Calculate metrics
            metrics = analytics.calculate_all_metrics(portfolio_returns)

            # Display metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Annualized Return", f"{metrics['cagr']:.2f}%")
                st.metric("Total Return", f"{metrics['total_return']:.2f}%")

            with col2:
                st.metric("Volatility (Annual)", f"{metrics['volatility']:.2f}%")
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")

            with col3:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

            # Cumulative returns chart
            st.subheader("📊 Cumulative Returns")

            cumulative_returns = (1 + portfolio_returns).cumprod()

            fig_cum = go.Figure()

            fig_cum.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=(cumulative_returns - 1) * 100,
                name='Portfolio',
                line=dict(color='#00CC00', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,204,0,0.1)'
            ))

            fig_cum.update_layout(
                title="Portfolio Cumulative Return",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                height=400,
                hovermode='x unified',
                template='plotly_white'
            )

            fig_cum.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            st.plotly_chart(fig_cum, use_container_width=True)

            # Drawdown chart
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max * 100

            fig_dd = go.Figure()

            fig_dd.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                name='Drawdown',
                line=dict(color='#FF6B6B', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,107,107,0.2)'
            ))

            fig_dd.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=300,
                template='plotly_white'
            )

            st.plotly_chart(fig_dd, use_container_width=True)

        else:
            st.warning("Insufficient historical data for performance analytics. Need at least 20 days of data.")

    # Export
    st.subheader("💾 Export Data")

    csv = position_metrics.to_csv(index=False)

    st.download_button(
        label="Download Position Details (CSV)",
        data=csv,
        file_name=f"portfolio_positions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Load a portfolio using the sidebar to get started!")

    st.markdown("""
    ### Getting Started

    **Option 1: Use Sample Portfolio**
    - Click "Load Sample Portfolio" in the sidebar
    - See a demo portfolio with 5 positions

    **Option 2: Upload Your Own Positions**
    - Prepare a CSV file with your positions
    - Required columns: `ticker`, `shares`, `cost_basis`
    - Optional column: `purchase_date`
    - Upload the file in the sidebar

    ### What You'll See

    1. **Portfolio Overview** - Total value, P&L, win/loss ratio
    2. **Position Details** - Each position with current prices and P&L
    3. **Holdings Breakdown** - Visual charts showing allocation
    4. **Performance Analytics** - Returns, volatility, Sharpe ratio, drawdowns

    This is **Section 1** of the Regime-Aware Portfolio Manager.
    """)
