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
from utils.theme import apply_dark_theme, dark_plotly_layout
from utils.portfolio_store import (
    save_portfolio, load_portfolio, portfolio_file_exists, get_last_saved_time
)

# Distinct color palette for holdings
_HOLDING_COLORS = [
    '#00d4aa', '#7B68EE', '#FF8C00', '#45B7D1', '#DDA0DD',
    '#4ECDC4', '#F7DC6F', '#EB984E', '#5DADE2', '#58D68D',
    '#BB8FCE', '#96CEB4', '#FF6B6B', '#98D8C8', '#FFEAA7',
]

# Page config
st.set_page_config(
    page_title="Portfolio Position Tracker",
    page_icon="💼",
    layout="wide"
)
apply_dark_theme()

st.title("💼 Portfolio Position Tracker")
st.markdown("**Section 1:** Load and track current positions with basic P&L, holdings breakdown, and performance metrics")

# Initialize
loader = PortfolioLoader()
analytics = PerformanceAnalytics()

# Sidebar - Portfolio Input
st.sidebar.header("Portfolio Input")

input_method = st.sidebar.radio(
    "How to load positions?",
    options=["Manual Entry", "Use Sample Portfolio", "Upload CSV"],
    help="Enter positions manually, use sample data, or upload CSV"
)

positions_df = None

if input_method == "Manual Entry":
    st.sidebar.subheader("Add Position")

    # Initialize manual positions in session state if not exists
    if 'manual_positions' not in st.session_state:
        st.session_state['manual_positions'] = []

    with st.sidebar.form("add_position_form", clear_on_submit=True):
        ticker = st.text_input("Ticker Symbol", placeholder="e.g., AAPL").upper()
        shares = st.number_input("Number of Shares", min_value=0.0, step=1.0, format="%.2f")
        cost_basis = st.number_input("Cost Basis per Share ($)", min_value=0.0, step=0.01, format="%.2f")
        purchase_date = st.date_input("Purchase Date (Optional)", value=datetime.now())

        submitted = st.form_submit_button("➕ Add Position")

        if submitted:
            if ticker and shares > 0 and cost_basis > 0:
                # Add to manual positions list
                st.session_state['manual_positions'].append({
                    'ticker': ticker,
                    'shares': shares,
                    'cost_basis': cost_basis,
                    'purchase_date': purchase_date.strftime('%Y-%m-%d')
                })
                st.sidebar.success(f"Added {shares} shares of {ticker}!")
            else:
                st.sidebar.error("Please fill in all required fields")

    # Show current manual positions
    if st.session_state['manual_positions']:
        st.sidebar.subheader(f"Current Positions ({len(st.session_state['manual_positions'])})")

        for idx, pos in enumerate(st.session_state['manual_positions']):
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.sidebar.text(f"{pos['ticker']}: {pos['shares']} @ ${pos['cost_basis']:.2f}")
            with col2:
                if st.sidebar.button("🗑️", key=f"delete_{idx}"):
                    st.session_state['manual_positions'].pop(idx)
                    st.rerun()

        # Convert manual positions to DataFrame
        positions_df = pd.DataFrame(st.session_state['manual_positions'])
        st.session_state['positions'] = positions_df

        if st.sidebar.button("🗑️ Clear All Positions"):
            st.session_state['manual_positions'] = []
            if 'positions' in st.session_state:
                del st.session_state['positions']
            st.rerun()
    else:
        st.sidebar.info("No positions added yet. Use the form above to add your first position.")

elif input_method == "Use Sample Portfolio":
    if st.sidebar.button("Load Sample Portfolio"):
        positions_df = loader.create_sample_portfolio()
        st.session_state['positions'] = positions_df
        # Clear manual positions when loading sample
        if 'manual_positions' in st.session_state:
            st.session_state['manual_positions'] = []
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
            # Clear manual positions when loading CSV
            if 'manual_positions' in st.session_state:
                st.session_state['manual_positions'] = []
            st.sidebar.success(f"Loaded {len(positions_df)} positions!")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

# ── Portfolio Persistence ────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("💾 Portfolio Persistence")

if portfolio_file_exists():
    last_saved = get_last_saved_time()
    if last_saved:
        try:
            saved_dt = datetime.fromisoformat(last_saved)
            st.sidebar.caption(f"Last saved: {saved_dt.strftime('%Y-%m-%d %H:%M')}")
        except Exception:
            pass
    if st.sidebar.button("📂 Load Saved Portfolio"):
        loaded = load_portfolio()
        if loaded is not None:
            st.session_state['positions'] = loaded
            st.session_state['manual_positions'] = loaded.to_dict(orient='records')
            st.sidebar.success(f"Loaded {len(loaded)} positions!")
            st.rerun()
        else:
            st.sidebar.error("Could not load saved portfolio.")
else:
    st.sidebar.caption("No saved portfolio found.")

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

    # Save button
    _save_col1, _save_col2 = st.columns([5, 1])
    with _save_col2:
        if st.button("💾 Save Portfolio"):
            if save_portfolio(positions_df):
                st.success("Saved!")
            else:
                st.error("Save failed.")

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
        # Pie chart by value — one distinct color per holding
        n = len(position_metrics)
        holding_colors = [_HOLDING_COLORS[i % len(_HOLDING_COLORS)] for i in range(n)]

        fig_pie = go.Figure(data=[go.Pie(
            labels=position_metrics['ticker'],
            values=position_metrics['current_value'],
            hole=0.3,
            textinfo='label+percent',
            marker=dict(colors=holding_colors)
        )])

        fig_pie.update_layout(
            title="Portfolio Allocation by Value",
            height=400,
            **dark_plotly_layout()
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

    # Benchmark selector (outside spinner so it persists)
    benchmark_ticker = st.selectbox(
        "Compare against benchmark:",
        options=['SPY', 'QQQ', 'DIA', 'IWM', 'None'],
        index=0,
        key='benchmark_select'
    )

    # Fetch historical data
    with st.spinner("Calculating performance metrics..."):
        # Get historical data for past year
        start_date = datetime.now() - timedelta(days=365)
        historical_data = loader.fetch_historical_data(tickers, start_date=start_date)

        # Fetch benchmark data
        benchmark_data = None
        if benchmark_ticker != 'None':
            bench_hist = loader.fetch_historical_data([benchmark_ticker], start_date=start_date)
            if bench_hist.get(benchmark_ticker) is not None and not bench_hist[benchmark_ticker].empty:
                benchmark_data = bench_hist[benchmark_ticker]['Close']

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

    # Tabs for performance, P&L history, risk
    tab_perf, tab_history, tab_risk = st.tabs(["📈 Performance", "📊 P&L History", "⚠️ Risk & Stress"])

    with tab_perf:
        if not portfolio_returns.empty and len(portfolio_returns) > 20:
            # Core metrics
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

            # Alpha / Beta vs benchmark
            if benchmark_data is not None and not benchmark_data.empty:
                bench_returns_aligned = benchmark_data.pct_change().dropna()
                common_idx = portfolio_returns.index.intersection(bench_returns_aligned.index)
                if len(common_idx) > 20:
                    p_ret = portfolio_returns.loc[common_idx]
                    b_ret = bench_returns_aligned.loc[common_idx]
                    beta_val   = p_ret.cov(b_ret) / b_ret.var() if b_ret.var() > 0 else 0
                    alpha_ann  = (p_ret.mean() - beta_val * b_ret.mean()) * 252 * 100
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(f"Beta vs {benchmark_ticker}", f"{beta_val:.2f}")
                    with col_b:
                        st.metric(f"Alpha vs {benchmark_ticker} (Ann.)", f"{alpha_ann:+.2f}%")

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

            # Benchmark overlay
            if benchmark_data is not None and not benchmark_data.empty:
                bench_rets = benchmark_data.pct_change().dropna()
                bench_cum  = (1 + bench_rets).cumprod()
                fig_cum.add_trace(go.Scatter(
                    x=bench_cum.index,
                    y=(bench_cum - 1) * 100,
                    name=benchmark_ticker,
                    line=dict(color='#FFD700', width=1.5, dash='dash'),
                ))

            fig_cum.update_layout(
                title="Portfolio vs Benchmark Cumulative Return",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                height=400,
                hovermode='x unified',
                **dark_plotly_layout()
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
                **dark_plotly_layout()
            )
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.warning("Insufficient historical data. Need at least 20 days.")

    with tab_history:
        st.markdown("#### Position P&L History")
        lookback_days = st.slider("Lookback (days)", min_value=30, max_value=365, value=180, step=30,
                                   key='pnl_history_lookback')
        hist_start = datetime.now() - timedelta(days=lookback_days)
        with st.spinner("Loading position histories..."):
            hist_data_pos = loader.fetch_historical_data(tickers, start_date=hist_start)
        fig_hist = go.Figure()
        for i, ticker in enumerate(tickers):
            if hist_data_pos.get(ticker) is not None and not hist_data_pos[ticker].empty:
                prices_h   = hist_data_pos[ticker]['Close']
                pos_row    = positions_df[positions_df['ticker'] == ticker].iloc[0]
                shares_h   = float(pos_row['shares'])
                cost_h     = float(pos_row['cost_basis'])
                pnl_series = prices_h * shares_h - cost_h * shares_h
                color      = _HOLDING_COLORS[i % len(_HOLDING_COLORS)]
                fig_hist.add_trace(go.Scatter(
                    x=prices_h.index, y=pnl_series,
                    name=ticker,
                    line=dict(color=color, width=1.5),
                ))
        fig_hist.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_hist.update_layout(
            title="Position Unrealized P&L Over Time ($)",
            xaxis_title="Date",
            yaxis_title="Unrealized P&L ($)",
            height=450,
            hovermode='x unified',
            **dark_plotly_layout()
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab_risk:
        st.markdown("#### Value at Risk (VaR)")
        if not portfolio_returns.empty and len(portfolio_returns) > 20:
            port_val      = summary['total_value']
            daily_vol     = portfolio_returns.std()
            var_95_param  = 1.645 * daily_vol * port_val
            var_99_param  = 2.326 * daily_vol * port_val
            var_95_hist   = abs(portfolio_returns.quantile(0.05)) * port_val
            var_99_hist   = abs(portfolio_returns.quantile(0.01)) * port_val

            col_v1, col_v2, col_v3, col_v4 = st.columns(4)
            with col_v1:
                st.metric("VaR 95% (Param)", f"${var_95_param:,.0f}", help="1-day 95% parametric VaR")
            with col_v2:
                st.metric("VaR 99% (Param)", f"${var_99_param:,.0f}", help="1-day 99% parametric VaR")
            with col_v3:
                st.metric("VaR 95% (Hist)", f"${var_95_hist:,.0f}", help="1-day 95% historical VaR")
            with col_v4:
                st.metric("VaR 99% (Hist)", f"${var_99_hist:,.0f}", help="1-day 99% historical VaR")

            st.markdown("#### Stress Test Scenarios")
            stress_rows = []
            for spct in [-10, -20, -30, -40]:
                impact    = port_val * (spct / 100)
                new_val   = port_val + impact
                stress_rows.append({
                    'SPY Move':              f"{spct:+.0f}%",
                    'Est. Portfolio Impact': f"${impact:,.0f}",
                    'Est. Portfolio Value':  f"${new_val:,.0f}",
                    'Impact %':              f"{spct:+.0f}%",
                })
            st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("Insufficient data for risk calculations.")

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
    st.info("👈 Add positions using the sidebar to get started!")

    st.markdown("""
    ### Getting Started

    **Option 1: Manual Entry (Recommended)**
    - Select "Manual Entry" in the sidebar
    - Enter ticker symbol, number of shares, and cost basis
    - Click "Add Position" to add to your portfolio
    - Add multiple positions one by one
    - Delete individual positions or clear all

    **Option 2: Use Sample Portfolio**
    - Click "Load Sample Portfolio" in the sidebar
    - See a demo portfolio with 5 positions

    **Option 3: Upload Your Own Positions**
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
