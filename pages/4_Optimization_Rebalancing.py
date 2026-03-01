"""
Regime-Aware Portfolio Manager
Section 4: Optimization & Rebalancing Tool
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.carbon_theme import apply_carbon_theme, carbon_plotly_layout, ACCENT, CARD, page_header

from calculations.optimizer import RegimeAwareOptimizer
from calculations.regime_detector import RegimeDetector
from data.portfolio_loader import PortfolioLoader
from data.market_data import MarketDataLoader


# Page config
st.set_page_config(
    page_title="Optimization & Rebalancing",
    page_icon="⚖️",
    layout="wide"
)
apply_carbon_theme()

page_header("⚖️ Portfolio Optimization & Rebalancing", "Generate regime-aware optimal allocation and rebalancing recommendations")

# Check if portfolio exists
if 'positions' not in st.session_state:
    st.warning("⚠️ No portfolio loaded. Please go to Home page and load a portfolio first.")
    st.stop()

positions_df = st.session_state['positions']

# Initialize
optimizer = RegimeAwareOptimizer()
loader = PortfolioLoader()
market_loader = MarketDataLoader()
detector = RegimeDetector()

# Sidebar settings
st.sidebar.header("Optimization Settings")

optimization_method = st.sidebar.selectbox(
    "Optimization Objective",
    options=['max_sharpe', 'min_volatility', 'max_quadratic_utility'],
    format_func=lambda x: {
        'max_sharpe': 'Maximize Sharpe Ratio',
        'min_volatility': 'Minimize Volatility',
        'max_quadratic_utility': 'Maximize Quadratic Utility'
    }[x]
)

st.sidebar.divider()
st.sidebar.header("💰 Rebalancing Controls")

min_trade_threshold = st.sidebar.slider(
    "Min Trade Size (% of portfolio)",
    min_value=0.5, max_value=10.0, value=2.0, step=0.5,
    help="Ignore rebalancing trades smaller than this % of portfolio value"
)

transaction_cost_bps = st.sidebar.number_input(
    "Transaction Cost (bps per trade)",
    min_value=0, max_value=100, value=10,
    help="Basis points charged per trade (10 bps = 0.10%)"
)

st.sidebar.caption(f"Trades < {min_trade_threshold:.1f}% of portfolio filtered out.")

# Get current regime
@st.cache_data
def get_current_regime():
    """Get current market regime"""
    spy_data = market_loader.load_index_data('SPY', '2y')
    vix_data = market_loader.load_vix_data('2y')
    spy_prices, vix_prices = market_loader.align_data(spy_data, vix_data)
    regime, signals = detector.classify_regime(spy_prices, vix_prices)
    return regime.iloc[-1]

current_regime = get_current_regime()

# Display current regime
st.subheader(f"📊 Current Market Regime: {current_regime}")

regime_constraints = optimizer.regime_constraints.get(current_regime, {})

col1, col2 = st.columns(2)
with col1:
    st.metric("Max Equity Exposure", f"{regime_constraints.get('max_equity_exposure', 0)*100:.0f}%")
with col2:
    st.metric("Min Cash Allocation", f"{regime_constraints.get('risk_free_min', 0)*100:.0f}%")

st.info(f"**{current_regime} Strategy:** {regime_constraints.get('description', '')}")

# Run optimization
st.subheader("⚙️ Running Optimization...")

if st.button("🚀 Optimize Portfolio", type="primary"):
    with st.spinner("Optimizing portfolio allocation..."):
        try:
            # Get tickers
            tickers = positions_df['ticker'].tolist()

            # Fetch current prices
            current_prices = loader.fetch_current_prices(tickers)

            # Calculate current portfolio metrics
            position_metrics = loader.calculate_position_metrics(positions_df, current_prices)
            total_value = position_metrics['current_value'].sum()

            # Run optimization
            optimization_result = optimizer.optimize_portfolio(
                tickers=tickers,
                current_regime=current_regime,
                optimization_method=optimization_method
            )

            # Store in session state
            st.session_state['optimization_result'] = optimization_result
            st.session_state['current_prices'] = current_prices
            st.session_state['total_value'] = total_value

            st.success("✅ Optimization complete!")

        except Exception as e:
            st.error(f"Error during optimization: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Display results if optimization has been run
if 'optimization_result' in st.session_state:
    result = st.session_state['optimization_result']
    current_prices = st.session_state['current_prices']
    total_value = st.session_state['total_value']

    # Display optimal allocation
    st.subheader("🎯 Optimal Allocation")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Expected Return", f"{result['expected_return']:.2f}%")

    with col2:
        st.metric("Expected Volatility", f"{result['volatility']:.2f}%")

    with col3:
        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")

    with col4:
        st.metric("Cash Allocation", f"{result['cash_allocation']:.1f}%")

    # Weights comparison
    st.subheader("📊 Allocation Comparison")

    # Current weights
    current_weights = {}
    for _, pos in positions_df.iterrows():
        ticker = pos['ticker']
        value = pos['shares'] * current_prices.get(ticker, 0)
        current_weights[ticker] = value / total_value

    # Create comparison dataframe
    all_tickers = set(list(current_weights.keys()) + list(result['weights'].keys()))
    comparison_data = []

    for ticker in all_tickers:
        current_weight = current_weights.get(ticker, 0) * 100
        optimal_weight = result['weights'].get(ticker, 0) * 100
        diff = optimal_weight - current_weight

        comparison_data.append({
            'Ticker': ticker,
            'Current %': current_weight,
            'Optimal %': optimal_weight,
            'Change %': diff,
            'Action': 'INCREASE' if diff > 1 else 'DECREASE' if diff < -1 else 'HOLD'
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Optimal %', ascending=False)

    # Color code the action
    def color_action(val):
        if val == 'INCREASE':
            return 'background-color: lightgreen'
        elif val == 'DECREASE':
            return 'background-color: #FFB3B3'
        else:
            return 'background-color: lightyellow'

    styled_comparison = comparison_df.style.format({
        'Current %': '{:.2f}%',
        'Optimal %': '{:.2f}%',
        'Change %': '{:+.2f}%'
    }).applymap(color_action, subset=['Action'])

    st.dataframe(styled_comparison, use_container_width=True, hide_index=True)

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        # Current allocation pie chart
        fig_current = go.Figure(data=[go.Pie(
            labels=list(current_weights.keys()),
            values=[v*100 for v in current_weights.values()],
            hole=0.3,
            title="Current Allocation"
        )])

        fig_current.update_layout(**carbon_plotly_layout(height=400))
        st.plotly_chart(fig_current, use_container_width=True)

    with col2:
        # Optimal allocation pie chart
        fig_optimal = go.Figure(data=[go.Pie(
            labels=list(result['weights'].keys()),
            values=[v*100 for v in result['weights'].values()],
            hole=0.3,
            title="Optimal Allocation",
            marker=dict(colors=[ACCENT if k == 'CASH' else None for k in result['weights'].keys()])
        )])

        fig_optimal.update_layout(**carbon_plotly_layout(height=400))
        st.plotly_chart(fig_optimal, use_container_width=True)

    # Bar chart comparison
    st.subheader("📊 Weight Changes")

    comparison_no_cash = comparison_df[comparison_df['Ticker'] != 'CASH'].copy()

    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        name='Current',
        x=comparison_no_cash['Ticker'],
        y=comparison_no_cash['Current %'],
        marker_color='rgba(34,211,238,0.45)'
    ))

    fig_bar.add_trace(go.Bar(
        name='Optimal',
        x=comparison_no_cash['Ticker'],
        y=comparison_no_cash['Optimal %'],
        marker_color='rgba(34,211,238,0.90)'
    ))

    fig_bar.update_layout(**carbon_plotly_layout(
        barmode='group',
        title="Current vs. Optimal Weights",
        xaxis_title="Ticker",
        yaxis_title="Weight (%)",
        height=400,
    ))

    st.plotly_chart(fig_bar, use_container_width=True)

    # Rebalancing trades
    st.subheader("🔄 Rebalancing Trades")

    trades_df = optimizer.calculate_rebalancing_trades(
        current_positions=positions_df,
        optimal_weights=result['weights'],
        current_prices=current_prices,
        total_portfolio_value=total_value
    )

    if not trades_df.empty:
        # Apply minimum trade size filter
        min_trade_value = total_value * (min_trade_threshold / 100)
        trades_filtered = trades_df[trades_df['Value'] >= min_trade_value].copy()
        trades_below = trades_df[trades_df['Value'] < min_trade_value]

        # Cost/benefit analysis
        total_trade_val = trades_filtered['Value'].sum() if not trades_filtered.empty else 0
        cost_per_trade  = transaction_cost_bps / 10000
        total_tx_cost   = total_trade_val * cost_per_trade
        net_benefit_ann = (result['expected_return'] / 100 * total_value) - total_tx_cost

        if not trades_filtered.empty:
            st.markdown(
                f"**Execute {len(trades_filtered)} trades to reach optimal allocation**"
                + (f" ({len(trades_below)} below threshold filtered out)" if not trades_below.empty else "") + ":"
            )
        else:
            st.success("✅ No trades exceed the minimum threshold. Portfolio effectively at target.")

        if not trades_below.empty:
            st.caption(
                f"Filtered: {', '.join(trades_below['Ticker'].tolist())} "
                f"(< ${min_trade_value:,.0f} each)"
            )

        # Cost/benefit summary row
        col_cb1, col_cb2, col_cb3 = st.columns(3)
        with col_cb1:
            st.metric("Total Trade Value", f"${total_trade_val:,.0f}")
        with col_cb2:
            st.metric("Est. Transaction Costs", f"${total_tx_cost:,.0f}",
                      help=f"{transaction_cost_bps} bps × trade value")
        with col_cb3:
            st.metric("Net Annual Benefit", f"${net_benefit_ann:,.0f}",
                      delta=f"{net_benefit_ann/total_value*100:.2f}% of portfolio",
                      help="Expected return improvement minus transaction costs")

        if not trades_filtered.empty:
            def color_trade_action(val):
                if val in ['BUY', 'INCREASE']:
                    return 'background-color: lightgreen'
                else:
                    return 'background-color: #FFB3B3'

            styled_trades = trades_filtered.style.format({
                'Shares': '{:.2f}',
                'Price': '${:.2f}',
                'Value': '${:,.2f}',
                'Current Weight': '{:.2f}%',
                'Target Weight': '{:.2f}%',
                'Weight Change': '{:+.2f}%'
            }).applymap(color_trade_action, subset=['Action'])

            st.dataframe(styled_trades, use_container_width=True, hide_index=True)

            csv_trades = trades_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Rebalancing Trades (CSV)",
                data=csv_trades,
                file_name=f"rebalancing_trades_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    else:
        st.success("✅ Portfolio is already optimally allocated! No trades needed.")

    # Expected impact
    st.subheader("📈 Expected Impact")

    st.markdown(f"""
    By rebalancing to the optimal allocation for the current **{current_regime}** regime:

    - **Expected Annual Return:** {result['expected_return']:.2f}%
    - **Expected Volatility:** {result['volatility']:.2f}%
    - **Sharpe Ratio:** {result['sharpe_ratio']:.2f}
    - **Cash Cushion:** {result['cash_allocation']:.1f}% ({result['cash_allocation']/100 * total_value:,.0f})

    This allocation is optimized for {optimization_method.replace('_', ' ')} in the current market regime.
    """)

    # Correlation Matrix & Diversification Score
    st.subheader("🔗 Correlation Matrix & Diversification")

    with st.spinner("Calculating 60-day correlations..."):
        try:
            _corr_tickers = positions_df['ticker'].tolist()
            prices_corr = optimizer.fetch_price_data(_corr_tickers, period='3mo')
            if not prices_corr.empty and prices_corr.shape[1] > 1:
                rets_corr  = prices_corr.pct_change().dropna()
                corr_mtx   = rets_corr.corr()
                n          = len(corr_mtx)

                # Average absolute off-diagonal correlation → diversification score
                off_diag = [
                    abs(corr_mtx.iloc[i, j])
                    for i in range(n) for j in range(i + 1, n)
                ]
                avg_corr          = sum(off_diag) / len(off_diag) if off_diag else 0
                diversif_score    = (1 - avg_corr) * 100

                # High-correlation pairs (≥0.85)
                high_corr_pairs = [
                    (corr_mtx.index[i], corr_mtx.columns[j], corr_mtx.iloc[i, j])
                    for i in range(n) for j in range(i + 1, n)
                    if abs(corr_mtx.iloc[i, j]) >= 0.85
                ]

                col_div, col_heat = st.columns([1, 3])

                with col_div:
                    score_color = "#22d3ee" if diversif_score >= 60 else "#f59e0b" if diversif_score >= 40 else "#fb7185"
                    label_text  = "Good" if diversif_score >= 60 else "Moderate" if diversif_score >= 40 else "Poor"
                    st.markdown(
                        f"<div style='text-align:center; padding:20px;'>"
                        f"<p style='margin:0; font-size:13px; color:#aaa;'>Diversification Score</p>"
                        f"<p style='margin:0; font-size:52px; font-weight:bold; color:{score_color};'>{diversif_score:.0f}</p>"
                        f"<p style='margin:0; font-size:13px; color:{score_color};'>{label_text}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.caption(f"Avg pairwise correlation: {avg_corr:.2f}")
                    if high_corr_pairs:
                        st.markdown("**Highly correlated (≥0.85):**")
                        for t1, t2, val in high_corr_pairs:
                            st.warning(f"{t1} ↔ {t2}: {val:.2f}")
                    else:
                        st.success("No pairs ≥ 0.85 correlation")

                with col_heat:
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_mtx.values,
                        x=corr_mtx.columns.tolist(),
                        y=corr_mtx.index.tolist(),
                        colorscale='RdBu_r',
                        zmid=0, zmin=-1, zmax=1,
                        text=corr_mtx.round(2).values,
                        texttemplate='%{text}',
                        textfont=dict(size=11),
                        showscale=True,
                        colorbar=dict(title='Corr')
                    ))
                    fig_corr.update_layout(**carbon_plotly_layout(
                        title="60-Day Return Correlations",
                        height=400,
                    ))
                    st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Need at least 2 tickers for correlation analysis.")
        except Exception as e:
            st.warning(f"Could not calculate correlations: {e}")

    # Risk warning
    st.warning("""
    ⚠️ **Important Considerations:**
    - Historical optimization does not guarantee future results
    - Consider transaction costs and tax implications
    - Review individual position fundamentals before executing
    - Market regime can change - monitor regularly
    - This is guidance, not financial advice
    """)

else:
    st.info("👆 Click **'Optimize Portfolio'** to generate optimal allocation and rebalancing recommendations.")

# Educational content
with st.expander("📚 How Optimization Works"):
    st.markdown("""
    ## Portfolio Optimization Methodology

    ### Regime-Based Constraints

    Different market regimes require different allocation strategies:

    | Regime | Max Equity | Min Cash | Strategy |
    |--------|-----------|----------|----------|
    | **Low Vol** | 100% | 0% | Aggressive - calm markets allow full exposure |
    | **High Vol** | 60% | 40% | Defensive - preserve capital in volatile markets |
    | **Trending** | 90% | 10% | High exposure - capture trend momentum |
    | **Mean Reversion** | 80% | 20% | Moderate - balance opportunity and risk |

    ---

    ### Optimization Objectives

    **1. Maximize Sharpe Ratio (Recommended)**
    - Best risk-adjusted returns
    - Balances return vs. volatility
    - Formula: (Return - Risk Free Rate) / Volatility

    **2. Minimize Volatility**
    - Lowest risk portfolio
    - Prioritizes stability over returns
    - Good for conservative investors

    **3. Maximize Quadratic Utility**
    - Balances return and risk aversion
    - Penalizes variance based on risk tolerance

    ---

    ### Position Constraints

    - **Max Position Size:** 30% (prevents over-concentration)
    - **No Shorting:** Weights between 0-30%
    - **Regime Cash Constraint:** Enforces minimum cash based on regime

    ---

    ### Calculation Process

    1. **Fetch Historical Data:** 2 years of daily prices
    2. **Calculate Expected Returns:** Mean historical returns
    3. **Calculate Covariance Matrix:** Ledoit-Wolf shrinkage (reduces estimation error)
    4. **Run Optimization:** PyPortfolioOpt efficient frontier
    5. **Apply Regime Constraints:** Scale down if needed, add cash
    6. **Generate Trades:** Compare optimal vs. current weights

    ---

    ### When to Rebalance

    **Rebalance when:**
    - Market regime changes
    - Position drifts >5% from target
    - Quarterly or monthly (calendar-based)
    - New capital added/withdrawn

    **Don't rebalance if:**
    - Changes are <1% (transaction costs > benefit)
    - Tax implications are significant
    - Individual position fundamentals have changed (review Section 3)

    ---

    ### Limitations

    - Based on historical data (past ≠ future)
    - Does not account for transaction costs, taxes, or slippage
    - Assumes normal distribution of returns (may not hold in crisis)
    - Ignores position-specific factors (use Section 3 for that)

    **Best Practice:** Combine optimization (top-down allocation) with scoring (bottom-up selection).
    """)
