"""
Regime-Aware Portfolio Manager
Section 7: Live Options Chain
Pull real options data from public API
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.theme import apply_dark_theme, dark_plotly_layout

from data.options_data import OptionsDataLoader
from calculations.options_analytics import OptionsAnalytics
from calculations.strategy_builder import StrategyBuilder
from calculations.probability_utils import probability_of_profit, expected_value, probability_of_touching


# Page config
st.set_page_config(
    page_title="Live Options Chain",
    page_icon="🔴",
    layout="wide"
)
apply_dark_theme()

st.title("🔴 Live Options Chain")
st.markdown("**Section 7:** Pull real-time options data from public API")

# Initialize
options_loader = OptionsDataLoader()
options_calc = OptionsAnalytics()

# Sidebar - Ticker selection
st.sidebar.header("Options Chain Lookup")

ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="SPY",
    help="Enter stock ticker (e.g., SPY, AAPL, TSLA)"
).upper()

if st.sidebar.button("🔍 Load Options Chain", type="primary"):
    st.session_state['options_ticker'] = ticker
    st.session_state['options_loaded'] = True

# Check if we have a ticker loaded
if 'options_ticker' in st.session_state and st.session_state.get('options_loaded', False):
    ticker = st.session_state['options_ticker']

    with st.spinner(f"Loading options data for {ticker}..."):
        # Get summary
        summary = options_loader.get_options_summary(ticker)

        if 'error' in summary:
            st.error(f"Error: {summary['error']}")
            st.stop()

        # Display summary
        st.subheader(f"📊 {ticker} Options Overview")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Current Price", f"${summary['current_price']:.2f}")

        with col2:
            st.metric("Expirations", summary['num_expirations'])

        with col3:
            st.metric("Call Volume", f"{summary['total_call_volume']:,.0f}")

        with col4:
            st.metric("Put Volume", f"{summary['total_put_volume']:,.0f}")

        with col5:
            st.metric("Put/Call Ratio", f"{summary['put_call_ratio']:.2f}")

        # Get available expirations
        expirations = options_loader.get_options_expirations(ticker)

        if not expirations:
            st.error(f"No options available for {ticker}")
            st.stop()

        # Expiration selection
        st.sidebar.markdown("---")
        st.sidebar.subheader("Select Expiration")

        # Calculate days to expiration for each
        exp_with_dte = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days
            exp_with_dte.append(f"{exp} ({dte} DTE)")

        selected_exp_display = st.sidebar.selectbox(
            "Expiration Date",
            options=exp_with_dte,
            index=0
        )

        # Extract actual date from display string
        selected_expiration = selected_exp_display.split(' ')[0]

        # Load options chain
        with st.spinner("Loading options chain..."):
            calls, puts, underlying_price = options_loader.get_options_chain(ticker, selected_expiration)

        if calls.empty and puts.empty:
            st.error("No options data available for this expiration")
            st.stop()

        # Calculate days to expiration
        exp_date = datetime.strptime(selected_expiration, '%Y-%m-%d')
        days_to_exp = (exp_date - datetime.now()).days
        time_to_exp = max(days_to_exp / 365, 0.001)

        st.info(f"**Expiration:** {selected_expiration} | **Days to Expiry:** {days_to_exp} | **Underlying:** ${underlying_price:.2f}")

        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Options Chain",
            "🔥 High Volume Options",
            "📈 IV Smile",
            "⚡ Quick Analyze",
            "🔧 Strategy Builder",
            "📉 Backtest Strategy"
        ])

        # === TAB 1: FULL OPTIONS CHAIN ===
        with tab1:
            st.subheader("Full Options Chain with Greeks")

            # Calculate Greeks for all options
            with st.spinner("Calculating Greeks for all strikes..."):

                # Calls
                calls_with_greeks = calls.copy()
                for idx, row in calls_with_greeks.iterrows():
                    if row['impliedVolatility'] > 0:
                        greeks = options_calc.calculate_greeks(
                            S=underlying_price,
                            K=row['strike'],
                            T=time_to_exp,
                            sigma=row['impliedVolatility'],
                            option_type='call'
                        )
                        calls_with_greeks.at[idx, 'delta'] = greeks['delta']
                        calls_with_greeks.at[idx, 'gamma'] = greeks['gamma']
                        calls_with_greeks.at[idx, 'theta'] = greeks['theta']
                        calls_with_greeks.at[idx, 'vega'] = greeks['vega']

                # Puts
                puts_with_greeks = puts.copy()
                for idx, row in puts_with_greeks.iterrows():
                    if row['impliedVolatility'] > 0:
                        greeks = options_calc.calculate_greeks(
                            S=underlying_price,
                            K=row['strike'],
                            T=time_to_exp,
                            sigma=row['impliedVolatility'],
                            option_type='put'
                        )
                        puts_with_greeks.at[idx, 'delta'] = greeks['delta']
                        puts_with_greeks.at[idx, 'gamma'] = greeks['gamma']
                        puts_with_greeks.at[idx, 'theta'] = greeks['theta']
                        puts_with_greeks.at[idx, 'vega'] = greeks['vega']

            # Display options
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📞 CALLS")

                # Select columns to display
                call_display_cols = [
                    'strike', 'lastPrice', 'bid', 'ask', 'volume',
                    'openInterest', 'impliedVolatility', 'delta',
                    'gamma', 'theta', 'vega', 'ITM'
                ]

                # Filter to available columns
                call_display_cols = [col for col in call_display_cols if col in calls_with_greeks.columns]

                calls_display = calls_with_greeks[call_display_cols].copy()

                # Highlight ITM
                def highlight_itm(row):
                    if row.get('ITM', False):
                        return ['background-color: #90EE90'] * len(row)
                    return [''] * len(row)

                styled_calls = calls_display.style.format({
                    'strike': '${:.2f}',
                    'lastPrice': '${:.2f}',
                    'bid': '${:.2f}',
                    'ask': '${:.2f}',
                    'volume': '{:,.0f}',
                    'openInterest': '{:,.0f}',
                    'impliedVolatility': '{:.2%}',
                    'delta': '{:.4f}',
                    'gamma': '{:.4f}',
                    'theta': '${:.2f}',
                    'vega': '${:.2f}'
                }).apply(highlight_itm, axis=1)

                st.dataframe(styled_calls, use_container_width=True, height=600)

            with col2:
                st.markdown("### 📉 PUTS")

                put_display_cols = [
                    'strike', 'lastPrice', 'bid', 'ask', 'volume',
                    'openInterest', 'impliedVolatility', 'delta',
                    'gamma', 'theta', 'vega', 'ITM'
                ]

                put_display_cols = [col for col in put_display_cols if col in puts_with_greeks.columns]

                puts_display = puts_with_greeks[put_display_cols].copy()

                styled_puts = puts_display.style.format({
                    'strike': '${:.2f}',
                    'lastPrice': '${:.2f}',
                    'bid': '${:.2f}',
                    'ask': '${:.2f}',
                    'volume': '{:,.0f}',
                    'openInterest': '{:,.0f}',
                    'impliedVolatility': '{:.2%}',
                    'delta': '{:.4f}',
                    'gamma': '{:.4f}',
                    'theta': '${:.2f}',
                    'vega': '${:.2f}'
                }).apply(highlight_itm, axis=1)

                st.dataframe(styled_puts, use_container_width=True, height=600)

            # Download options
            st.markdown("### 💾 Export Data")

            col_a, col_b = st.columns(2)

            with col_a:
                calls_csv = calls_with_greeks.to_csv(index=False)
                st.download_button(
                    "📥 Download Calls (CSV)",
                    data=calls_csv,
                    file_name=f"{ticker}_calls_{selected_expiration}.csv",
                    mime="text/csv"
                )

            with col_b:
                puts_csv = puts_with_greeks.to_csv(index=False)
                st.download_button(
                    "📥 Download Puts (CSV)",
                    data=puts_csv,
                    file_name=f"{ticker}_puts_{selected_expiration}.csv",
                    mime="text/csv"
                )

        # === TAB 2: HIGH VOLUME OPTIONS ===
        with tab2:
            st.subheader("🔥 Most Actively Traded Options")

            min_volume = st.slider(
                "Minimum Volume",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )

            high_vol_options = options_loader.get_high_volume_options(
                ticker=ticker,
                expiration=selected_expiration,
                min_volume=min_volume,
                top_n=20
            )

            if not high_vol_options.empty:
                # Format and display
                def color_by_type(row):
                    if row['type'] == 'call':
                        return ['background-color: #E6F3FF'] * len(row)
                    else:
                        return ['background-color: #FFE6E6'] * len(row)

                styled_high_vol = high_vol_options.style.format({
                    'strike': '${:.2f}',
                    'lastPrice': '${:.2f}',
                    'bid': '${:.2f}',
                    'ask': '${:.2f}',
                    'volume': '{:,.0f}',
                    'openInterest': '{:,.0f}',
                    'impliedVolatility': '{:.2%}',
                    'moneyness_pct': '{:+.2f}%',
                    'underlying_price': '${:.2f}'
                }).apply(color_by_type, axis=1)

                st.dataframe(styled_high_vol, use_container_width=True, hide_index=True)

                # Volume chart
                fig_vol = go.Figure()

                calls_vol = high_vol_options[high_vol_options['type'] == 'call']
                puts_vol = high_vol_options[high_vol_options['type'] == 'put']

                fig_vol.add_trace(go.Bar(
                    name='Calls',
                    x=calls_vol['strike'],
                    y=calls_vol['volume'],
                    marker_color='blue'
                ))

                fig_vol.add_trace(go.Bar(
                    name='Puts',
                    x=puts_vol['strike'],
                    y=puts_vol['volume'],
                    marker_color='red'
                ))

                fig_vol.update_layout(
                    title="Volume by Strike",
                    xaxis_title="Strike Price",
                    yaxis_title="Volume",
                    barmode='group',
                    height=400,
                    template='plotly_dark'
                )

                st.plotly_chart(fig_vol, use_container_width=True)

            else:
                st.warning(f"No options found with volume >= {min_volume}")

        # === TAB 3: IV SMILE ===
        with tab3:
            st.subheader("📈 Implied Volatility Smile/Skew")

            calls_iv, puts_iv = options_loader.calculate_implied_volatility_smile(
                ticker=ticker,
                expiration=selected_expiration
            )

            if not calls_iv.empty and not puts_iv.empty:
                fig_iv = go.Figure()

                fig_iv.add_trace(go.Scatter(
                    x=calls_iv['strike'],
                    y=calls_iv['impliedVolatility'] * 100,
                    name='Calls IV',
                    mode='lines+markers',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))

                fig_iv.add_trace(go.Scatter(
                    x=puts_iv['strike'],
                    y=puts_iv['impliedVolatility'] * 100,
                    name='Puts IV',
                    mode='lines+markers',
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ))

                # Add underlying price line
                fig_iv.add_vline(
                    x=underlying_price,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Current: ${underlying_price:.2f}"
                )

                fig_iv.update_layout(
                    title=f"Implied Volatility by Strike ({selected_expiration})",
                    xaxis_title="Strike Price ($)",
                    yaxis_title="Implied Volatility (%)",
                    height=500,
                    template='plotly_dark',
                    hovermode='x unified'
                )

                st.plotly_chart(fig_iv, use_container_width=True)

                # Interpretation
                st.markdown("### 📊 IV Skew Interpretation")

                avg_call_iv = calls_iv['impliedVolatility'].mean() * 100
                avg_put_iv = puts_iv['impliedVolatility'].mean() * 100

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Avg Call IV", f"{avg_call_iv:.1f}%")

                with col2:
                    st.metric("Avg Put IV", f"{avg_put_iv:.1f}%")

                with col3:
                    skew = avg_put_iv - avg_call_iv
                    st.metric("Put Skew", f"{skew:+.1f}%")

                if skew > 2:
                    st.info("**Put skew detected:** Puts trading at higher IV than calls, suggesting demand for downside protection")
                elif skew < -2:
                    st.info("**Call skew detected:** Calls trading at higher IV than puts, suggesting demand for upside exposure")
                else:
                    st.info("**Balanced IV:** Calls and puts trading at similar volatility levels")

            else:
                st.warning("Insufficient data for IV smile analysis")

        # === TAB 4: QUICK ANALYZE ===
        with tab4:
            st.subheader("⚡ Quick Analyze from Chain")

            st.markdown("Select an option from the chain to analyze:")

            col1, col2 = st.columns(2)

            with col1:
                option_type = st.radio("Option Type", options=['call', 'put'])

            with col2:
                if option_type == 'call':
                    available_strikes = calls['strike'].tolist()
                else:
                    available_strikes = puts['strike'].tolist()

                selected_strike = st.selectbox(
                    "Strike Price",
                    options=available_strikes,
                    index=len(available_strikes)//2 if available_strikes else 0
                )

            # Get option data
            option_data = options_loader.get_option_quote(
                ticker=ticker,
                strike=selected_strike,
                expiration=selected_expiration,
                option_type=option_type
            )

            if option_data:
                st.markdown("---")
                st.markdown(f"### Analyzing: {ticker} ${selected_strike} {option_type.upper()} ({selected_expiration})")

                # Display option data
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Last Price", f"${option_data.get('lastPrice', 0):.2f}")

                with col2:
                    st.metric("Bid/Ask", f"${option_data.get('bid', 0):.2f} / ${option_data.get('ask', 0):.2f}")

                with col3:
                    st.metric("Volume", f"{option_data.get('volume', 0):,.0f}")

                with col4:
                    st.metric("Open Interest", f"{option_data.get('openInterest', 0):,.0f}")

                # Analyze with full calculator
                analysis = options_calc.analyze_option_position(
                    S=underlying_price,
                    K=selected_strike,
                    T=time_to_exp,
                    sigma=option_data.get('impliedVolatility', 0.3),
                    option_type=option_type,
                    contracts=1,
                    premium_paid=None,
                    current_price=option_data.get('lastPrice', 0)
                )

                # Greeks
                st.markdown("### Greeks (Per Contract)")

                col1, col2, col3, col4, col5 = st.columns(5)

                greeks = analysis['greeks']

                with col1:
                    delta_color = "green" if greeks['delta'] > 0 else "red"
                    st.markdown(f"<h3 style='color: {delta_color};'>{greeks['delta']:.4f}</h3>", unsafe_allow_html=True)
                    st.caption("Delta")

                with col2:
                    st.metric("Gamma", f"{greeks['gamma']:.4f}")

                with col3:
                    theta_color = "red" if greeks['theta'] < 0 else "green"
                    st.markdown(f"<h3 style='color: {theta_color};'>${greeks['theta']:.2f}</h3>", unsafe_allow_html=True)
                    st.caption("Theta")

                with col4:
                    st.metric("Vega", f"${greeks['vega']:.2f}")

                with col5:
                    st.metric("Rho", f"${greeks['rho']:.2f}")

                # Fair value
                if analysis['fair_value_analysis']:
                    st.markdown("### Fair Value Analysis")

                    fv = analysis['fair_value_analysis']

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Market Price", f"${fv['market_price']:.2f}")

                    with col2:
                        st.metric("Theoretical Value", f"${fv['theoretical_value']:.2f}")

                    with col3:
                        st.metric(
                            "Difference",
                            f"${fv['difference']:.2f}",
                            f"{fv['difference_pct']:+.2f}%"
                        )

                    if abs(fv['difference_pct']) < 5:
                        st.success(f"✅ **{fv['rating']}** - Trading near theoretical value")
                    elif fv['difference_pct'] > 5:
                        st.warning(f"⚠️ **{fv['rating']}** - Trading above theoretical value")
                    else:
                        st.info(f"💡 **{fv['rating']}** - Trading below theoretical value")

            else:
                st.error("Could not load option data")

        # === TAB 5: STRATEGY BUILDER ===
        with tab5:
            st.subheader("🔧 Multi-Leg Strategy Builder")

            # Initialize strategy builder in session state
            if 'strategy_builder' not in st.session_state:
                st.session_state['strategy_builder'] = StrategyBuilder()

            builder = st.session_state['strategy_builder']

            # Template selection or manual
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### Build Your Strategy")

                build_mode = st.radio(
                    "How would you like to build?",
                    options=['Use Template', 'Build Manually'],
                    horizontal=True
                )

            with col2:
                if st.button("🗑️ Clear All Legs", use_container_width=True):
                    builder.clear_legs()
                    st.rerun()

            # Template mode
            if build_mode == 'Use Template':
                st.markdown("### Pre-Built Strategy Templates")

                template_name = st.selectbox(
                    "Select Strategy Template",
                    options=list(builder.strategy_templates.keys())
                )

                template_info = builder.strategy_templates[template_name]
                st.info(f"**{template_name}:** {template_info['description']}")

                if st.button("📥 Load Template", type="primary"):
                    success = builder.load_template(
                        template_name=template_name,
                        underlying_price=underlying_price,
                        calls_df=calls,
                        puts_df=puts
                    )

                    if success:
                        st.success(f"✅ Loaded {template_name} with {len(builder.legs)} legs")
                        st.rerun()
                    else:
                        st.error("Failed to load template")

            # Manual mode
            else:
                st.markdown("### Add Legs Manually")

                with st.form("add_leg_form"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        leg_type = st.selectbox("Type", options=['call', 'put'])
                        position = st.selectbox("Position", options=['long', 'short'])

                    with col2:
                        # Get available strikes
                        if leg_type == 'call':
                            available_strikes = calls['strike'].tolist()
                        else:
                            available_strikes = puts['strike'].tolist()

                        leg_strike = st.selectbox(
                            "Strike",
                            options=available_strikes,
                            index=len(available_strikes)//2 if available_strikes else 0
                        )

                    with col3:
                        leg_contracts = st.number_input(
                            "Contracts",
                            min_value=1,
                            max_value=100,
                            value=1
                        )

                    # Get option data for selected strike
                    option_data = options_loader.get_option_quote(
                        ticker=ticker,
                        strike=leg_strike,
                        expiration=selected_expiration,
                        option_type=leg_type
                    )

                    if option_data:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Last Price", f"${option_data.get('lastPrice', 0):.2f}")
                        with col_b:
                            st.metric("IV", f"{option_data.get('impliedVolatility', 0)*100:.1f}%")

                    submitted = st.form_submit_button("➕ Add Leg", type="primary", use_container_width=True)

                    if submitted and option_data:
                        builder.add_leg(
                            option_type=leg_type,
                            strike=leg_strike,
                            position=position,
                            contracts=leg_contracts,
                            premium=option_data.get('lastPrice', 0),
                            implied_vol=option_data.get('impliedVolatility', 0.3)
                        )
                        st.success(f"Added {position} {leg_type} @ ${leg_strike}")
                        st.rerun()

            # Display current strategy
            if builder.legs:
                st.markdown("---")
                st.markdown("### 📋 Current Strategy Legs")

                legs_df = builder.get_legs_dataframe()

                # Add remove buttons
                col_remove, col_table = st.columns([1, 5])

                with col_table:
                    st.dataframe(
                        legs_df.style.format({
                            'Strike': '${:.2f}',
                            'Premium': '${:.2f}',
                            'IV': '{:.2%}',
                            'Cost': '${:,.2f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                with col_remove:
                    st.markdown("**Remove**")
                    for i in range(len(builder.legs)):
                        if st.button(f"❌ {i+1}", key=f"remove_{i}"):
                            builder.remove_leg(i)
                            st.rerun()

                # Analyze strategy
                st.markdown("---")
                st.markdown("### 📊 Strategy Analysis")

                if st.button("🔍 Analyze Strategy", type="primary", use_container_width=True):
                    summary = builder.get_strategy_summary(underlying_price, time_to_exp)

                    if 'error' not in summary:
                        # Display summary
                        st.markdown("#### Strategy Overview")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            cost = summary['initial_cost']
                            if cost < 0:
                                st.metric("Net Credit", f"${abs(cost):,.2f}", "Collected")
                            else:
                                st.metric("Net Debit", f"${cost:,.2f}", "Paid")

                        with col2:
                            st.metric(
                                "Current P&L",
                                f"${summary['pnl']:,.2f}",
                                f"{summary['pnl_pct']:+.2f}%"
                            )

                        with col3:
                            st.metric("Max Profit", f"${summary['max_profit']:,.2f}")

                        with col4:
                            st.metric("Max Loss", f"${summary['max_loss']:,.2f}")

                        # Breakevens
                        if summary['breakevens']:
                            st.markdown("#### Breakeven Points")
                            breakeven_str = ', '.join([f"${be:.2f}" for be in summary['breakevens']])
                            st.info(f"**Breakevens:** {breakeven_str}")
                        else:
                            st.warning("No breakeven points (check strategy construction)")

                        # Greeks
                        st.markdown("#### Combined Greeks")

                        greeks = summary['greeks']

                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col1:
                            delta_color = "green" if greeks['delta'] > 0 else "red" if greeks['delta'] < 0 else "gray"
                            st.markdown(f"<h3 style='color: {delta_color};'>{greeks['delta']:.2f}</h3>", unsafe_allow_html=True)
                            st.caption("Delta")

                        with col2:
                            st.metric("Gamma", f"{greeks['gamma']:.2f}")

                        with col3:
                            theta_color = "green" if greeks['theta'] > 0 else "red"
                            st.markdown(f"<h3 style='color: {theta_color};'>${greeks['theta']:.2f}</h3>", unsafe_allow_html=True)
                            st.caption("Theta/Day")

                        with col4:
                            st.metric("Vega", f"${greeks['vega']:.2f}")

                        with col5:
                            st.metric("Rho", f"${greeks['rho']:.2f}")

                        # Payoff diagram
                        st.markdown("#### 📈 Payoff Diagram (At Expiration)")

                        price_range, pnl = builder.calculate_payoff(underlying_price)

                        fig_payoff = go.Figure()

                        # P&L line
                        fig_payoff.add_trace(go.Scatter(
                            x=price_range,
                            y=pnl,
                            name='P&L',
                            line=dict(color='blue', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(0,0,255,0.1)'
                        ))

                        # Zero line
                        fig_payoff.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

                        # Current price
                        fig_payoff.add_vline(
                            x=underlying_price,
                            line_dash="dash",
                            line_color="green",
                            annotation_text=f"Current: ${underlying_price:.2f}"
                        )

                        # Breakevens
                        for be in summary['breakevens']:
                            fig_payoff.add_vline(
                                x=be,
                                line_dash="dot",
                                line_color="orange",
                                annotation_text=f"BE: ${be:.2f}"
                            )

                        # Max profit/loss points
                        fig_payoff.add_trace(go.Scatter(
                            x=[summary['max_profit_price']],
                            y=[summary['max_profit']],
                            mode='markers',
                            name='Max Profit',
                            marker=dict(color='green', size=12, symbol='star')
                        ))

                        fig_payoff.add_trace(go.Scatter(
                            x=[summary['max_loss_price']],
                            y=[summary['max_loss']],
                            mode='markers',
                            name='Max Loss',
                            marker=dict(color='red', size=12, symbol='x')
                        ))

                        fig_payoff.update_layout(
                            title=f"Strategy Payoff Diagram ({summary['num_legs']} Legs)",
                            xaxis_title="Underlying Price at Expiration ($)",
                            yaxis_title="Profit/Loss ($)",
                            height=500,
                            template='plotly_dark',
                            hovermode='x unified'
                        )

                        st.plotly_chart(fig_payoff, use_container_width=True)

                        # Risk analysis
                        st.markdown("#### ⚠️ Risk Analysis")

                        risk_reward_ratio = abs(summary['max_profit'] / summary['max_loss']) if summary['max_loss'] != 0 else float('inf')

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Risk/Reward Ratio", f"1:{risk_reward_ratio:.2f}")

                            if greeks['delta'] > 50:
                                st.info("📈 **Bullish Bias** - Profits if price rises")
                            elif greeks['delta'] < -50:
                                st.info("📉 **Bearish Bias** - Profits if price falls")
                            else:
                                st.info("⚖️ **Neutral Bias** - Delta-neutral strategy")

                        with col2:
                            if greeks['theta'] > 0:
                                st.success(f"✅ **Positive Theta**: Earning ${greeks['theta']:.2f}/day from time decay")
                            elif greeks['theta'] < -20:
                                st.warning(f"⚠️ **High Theta Decay**: Losing ${abs(greeks['theta']):.2f}/day")
                            else:
                                st.info(f"**Theta**: ${greeks['theta']:.2f}/day")

                            if greeks['vega'] > 50:
                                st.info("📊 **Long Vega**: Benefits from IV increase")
                            elif greeks['vega'] < -50:
                                st.info("📊 **Short Vega**: Benefits from IV decrease")

                        # ── Probability Analysis ──────────────────────────────────────────
                        if time_to_exp > 0:
                            with st.expander("📊 Probability Analysis", expanded=True):
                                _atm_idx = max(0, len(calls) // 2)
                                _atm_iv  = float(calls['impliedVolatility'].iloc[_atm_idx]) if not calls.empty else 0.25
                                _atm_iv  = max(_atm_iv, 0.05)
                                _r       = 0.045

                                _pop = probability_of_profit(underlying_price, builder, _atm_iv, _r, time_to_exp)
                                _ev  = expected_value(underlying_price, builder, _atm_iv, _r, time_to_exp)

                                _pc1, _pc2, _pc3 = st.columns(3)
                                _pc1.metric("Probability of Profit", f"{_pop:.1%}")
                                _pc2.metric("Expected Value",        f"${_ev:,.0f}")
                                _pc3.metric("Breakevens",            len(summary['breakevens']))

                                # PoP gauge
                                _fig_gauge = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=_pop * 100,
                                    number={'suffix': '%', 'font': {'color': '#e8edf3'}},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar':  {'color': '#00d4aa'},
                                        'steps': [
                                            {'range': [0,  40], 'color': '#3a1515'},
                                            {'range': [40, 60], 'color': '#2d2a15'},
                                            {'range': [60, 100], 'color': '#1a3a1a'},
                                        ],
                                    },
                                    title={'text': 'Probability of Profit',
                                           'font': {'color': '#e8edf3'}}
                                ))
                                _fig_gauge.update_layout(**dark_plotly_layout(height=220))
                                st.plotly_chart(_fig_gauge, use_container_width=True)

                                # Touch probabilities for short legs
                                _short_legs = [l for l in builder.legs if l['contracts'] < 0]
                                if _short_legs:
                                    st.markdown("**Short Strike Touch Probabilities**")
                                    for _leg in _short_legs:
                                        _touch = probability_of_touching(
                                            underlying_price, _leg['strike'],
                                            _atm_iv, _r, time_to_exp
                                        )
                                        _tc = ('#22c55e' if _touch < 0.3
                                               else '#f59e0b' if _touch < 0.5
                                               else '#ef4444')
                                        st.markdown(
                                            f"- Strike **${_leg['strike']:.1f}** "
                                            f"({_leg['option_type']}): "
                                            f"<span style='color:{_tc}'>"
                                            f"{_touch:.1%} touch probability</span>",
                                            unsafe_allow_html=True
                                        )

                    else:
                        st.error(summary['error'])

            else:
                st.info("👆 Add legs to your strategy using templates or manual entry above")

                st.markdown("""
                ### Strategy Builder Guide

                **1. Use Templates:**
                - Quick way to build common strategies
                - Iron Condor, Spreads, Straddles, etc.
                - Auto-selects strikes based on current price

                **2. Build Manually:**
                - Select type (call/put)
                - Choose position (long/short)
                - Pick strike from live chain
                - Add multiple legs

                **3. Analyze:**
                - See payoff diagram
                - Calculate max profit/loss
                - Check breakeven points
                - View combined Greeks
                - Risk/reward analysis

                **Popular Multi-Leg Strategies:**
                - **Iron Condor**: Limited risk, limited profit, range-bound
                - **Vertical Spreads**: Directional with defined risk
                - **Straddle/Strangle**: Volatility plays
                - **Butterflies**: Precise price target strategies
                """)

        # === TAB 6: BACKTEST STRATEGY ===
        with tab6:
            import yfinance as yf
            from calculations.regime_detector import RegimeDetector as _RD

            st.subheader("📉 Options Strategy Backtester")
            st.markdown(
                "Simulates buying a strategy at rolling entry points using historical prices "
                "and realized volatility as a proxy for IV.  P&L is measured at expiry."
            )

            # ── Controls ──────────────────────────────────────────────────────
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                bt_strategy = st.selectbox(
                    "Strategy",
                    [
                        "Long ATM Call",
                        "Long OTM Put (Protective)",
                        "Iron Condor",
                        "Bull Call Spread",
                        "Bear Put Spread",
                    ],
                    key="bt_strategy"
                )
                bt_dte = st.select_slider(
                    "Days to Expiration",
                    options=[14, 21, 30, 45, 60],
                    value=30,
                    key="bt_dte"
                )

            with col_b:
                bt_offset_pct = st.slider(
                    "Strike Offset from ATM (%)",
                    min_value=1.0, max_value=12.0, value=5.0, step=0.5,
                    help="OTM distance used for spread wings and protective strikes",
                    key="bt_offset"
                )
                bt_period = st.selectbox(
                    "Lookback Period",
                    ["6mo", "1y", "2y", "3y"],
                    index=1,
                    key="bt_period"
                )

            with col_c:
                bt_risk_free = st.number_input(
                    "Risk-Free Rate (%)", value=4.5, step=0.25, key="bt_rf"
                ) / 100
                st.metric("Ticker", ticker)
                st.metric("Current Price", f"${underlying_price:.2f}")

            # ── Compute button ─────────────────────────────────────────────────
            run_bt = st.button("▶  Run Backtest", type="primary", use_container_width=True, key="run_bt")

            @st.cache_data(show_spinner="Running backtest…")
            def _run_backtest(ticker_, strategy_, dte_, offset_pct_, period_, r_):
                """Vectorised backtest — cached for performance."""
                import numpy as _np
                import pandas as _pd

                hist = yf.Ticker(ticker_).history(period=period_)
                if hist.empty:
                    return _pd.DataFrame()

                prices_ = hist['Close'].copy()
                prices_.index = _pd.to_datetime(prices_.index).tz_localize(None)

                det = _RD(lookback_vol=20, lookback_trend=50)
                regime_, _ = det.classify_regime(prices_)

                returns_ = prices_.pct_change()
                realized_vol_ = returns_.rolling(20).std() * _np.sqrt(252)

                _calc = OptionsAnalytics()
                T_ = dte_ / 365.0
                off = offset_pct_ / 100.0

                def _pnl(S, S_T, sig):
                    sig = max(min(float(sig), 2.0), 0.05)
                    call_pay = lambda K: max(float(S_T) - K, 0)
                    put_pay  = lambda K: max(K - float(S_T), 0)

                    if strategy_ == "Long ATM Call":
                        K = float(S)
                        cost = _calc.black_scholes(S, K, T_, sig, 'call', r_)
                        return (call_pay(K) - cost) * 100

                    elif strategy_ == "Long OTM Put (Protective)":
                        K = float(S) * (1 - off)
                        cost = _calc.black_scholes(S, K, T_, sig, 'put', r_)
                        return (put_pay(K) - cost) * 100

                    elif strategy_ == "Iron Condor":
                        ps, pl = S * (1 - off), S * (1 - 2 * off)
                        cs, cl = S * (1 + off), S * (1 + 2 * off)
                        credit = (
                            _calc.black_scholes(S, ps, T_, sig, 'put',  r_)
                            - _calc.black_scholes(S, pl, T_, sig, 'put',  r_)
                            + _calc.black_scholes(S, cs, T_, sig, 'call', r_)
                            - _calc.black_scholes(S, cl, T_, sig, 'call', r_)
                        )
                        loss = (
                            (put_pay(ps)  - put_pay(pl))
                            + (call_pay(cs) - call_pay(cl))
                        )
                        return (credit - loss) * 100

                    elif strategy_ == "Bull Call Spread":
                        lK, sK = float(S), S * (1 + off)
                        cost = _calc.black_scholes(S, lK, T_, sig, 'call', r_) \
                             - _calc.black_scholes(S, sK, T_, sig, 'call', r_)
                        return (call_pay(lK) - call_pay(sK) - cost) * 100

                    elif strategy_ == "Bear Put Spread":
                        lK, sK = float(S), S * (1 - off)
                        cost = _calc.black_scholes(S, lK, T_, sig, 'put', r_) \
                             - _calc.black_scholes(S, sK, T_, sig, 'put', r_)
                        return (put_pay(lK) - put_pay(sK) - cost) * 100

                    return 0.0

                rows = []
                step = 5  # enter every 5 trading days
                for i in range(30, len(prices_) - dte_ - 2, step):
                    S_  = float(prices_.iloc[i])
                    S_T = float(prices_.iloc[i + dte_])
                    sig = float(realized_vol_.iloc[i]) if not _pd.isna(realized_vol_.iloc[i]) else 0.20
                    reg = regime_.iloc[i]
                    pnl = _pnl(S_, S_T, sig)
                    rows.append({
                        'date':     prices_.index[i],
                        'S':        S_,
                        'S_T':      S_T,
                        'move_pct': (S_T - S_) / S_ * 100,
                        'IV_proxy': sig * 100,
                        'regime':   reg,
                        'pnl':      pnl,
                    })
                return _pd.DataFrame(rows)

            if run_bt or st.session_state.get('bt_df') is not None:
                if run_bt:
                    df_bt = _run_backtest(
                        ticker, bt_strategy, bt_dte,
                        bt_offset_pct, bt_period, bt_risk_free
                    )
                    st.session_state['bt_df'] = df_bt
                else:
                    df_bt = st.session_state.get('bt_df', None)

                if df_bt is not None and not df_bt.empty:
                    valid = df_bt[df_bt['regime'] != 'Unknown'].copy()

                    # ── Summary metrics ──────────────────────────────────────
                    total_trades = len(valid)
                    winners = (valid['pnl'] > 0).sum()
                    win_rate = winners / total_trades * 100 if total_trades else 0
                    avg_pnl = valid['pnl'].mean()
                    best = valid['pnl'].max()
                    worst = valid['pnl'].min()
                    total_pnl = valid['pnl'].sum()

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Trades", total_trades)
                    c2.metric("Win Rate", f"{win_rate:.1f}%")
                    c3.metric("Avg P&L", f"${avg_pnl:+.0f}")
                    c4.metric("Best Trade", f"${best:+.0f}")
                    c5.metric("Worst Trade", f"${worst:+.0f}")

                    # ── Cumulative P&L chart ─────────────────────────────────
                    st.markdown("### Cumulative P&L Over Time")
                    valid_sorted = valid.sort_values('date').copy()
                    valid_sorted['cum_pnl'] = valid_sorted['pnl'].cumsum()

                    _rc_bt = {
                        'Low Vol': '#22c55e', 'High Vol': '#ef4444',
                        'Trending': '#4a9eff', 'Mean Reversion': '#f59e0b', 'Unknown': '#6b7a8f'
                    }

                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=valid_sorted['date'], y=valid_sorted['cum_pnl'],
                        mode='lines', name='Cumulative P&L',
                        line=dict(color='#00d4aa', width=2.5),
                        fill='tozeroy', fillcolor='rgba(0,212,170,0.08)'
                    ))

                    # Colour dots by regime
                    for reg_name, reg_col in _rc_bt.items():
                        mask = valid_sorted['regime'] == reg_name
                        if mask.sum():
                            fig_cum.add_trace(go.Scatter(
                                x=valid_sorted.loc[mask, 'date'],
                                y=valid_sorted.loc[mask, 'cum_pnl'],
                                mode='markers', name=reg_name,
                                marker=dict(color=reg_col, size=6, opacity=0.7)
                            ))

                    fig_cum.add_hline(y=0, line_color='#6b7a8f', line_dash='dash')
                    fig_cum.update_layout(
                        **dark_plotly_layout(height=400, hovermode='x unified',
                                             title=f"{bt_strategy} — Cumulative P&L")
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)

                    # ── Per-regime breakdown ─────────────────────────────────
                    st.markdown("### Performance by Regime")

                    regime_stats = valid.groupby('regime')['pnl'].agg(
                        Trades='count',
                        Win_Rate=lambda x: (x > 0).mean() * 100,
                        Avg_PnL='mean',
                        Total_PnL='sum',
                        Best='max',
                        Worst='min'
                    ).reset_index()

                    regime_stats = regime_stats.rename(columns={
                        'regime': 'Regime', 'Trades': 'Trades',
                        'Win_Rate': 'Win Rate (%)', 'Avg_PnL': 'Avg P&L ($)',
                        'Total_PnL': 'Total P&L ($)', 'Best': 'Best ($)', 'Worst': 'Worst ($)'
                    })

                    st.dataframe(
                        regime_stats.style.format({
                            'Win Rate (%)': '{:.1f}%',
                            'Avg P&L ($)': '${:+.0f}',
                            'Total P&L ($)': '${:+,.0f}',
                            'Best ($)': '${:+.0f}',
                            'Worst ($)': '${:+.0f}'
                        }).background_gradient(subset=['Win Rate (%)'], cmap='RdYlGn'),
                        use_container_width=True, hide_index=True
                    )

                    # ── P&L distribution chart ───────────────────────────────
                    st.markdown("### P&L Distribution")

                    fig_dist = go.Figure()
                    for reg_name, reg_col in _rc_bt.items():
                        mask = valid['regime'] == reg_name
                        if mask.sum():
                            fig_dist.add_trace(go.Box(
                                y=valid.loc[mask, 'pnl'],
                                name=reg_name,
                                marker_color=reg_col,
                                boxmean=True
                            ))

                    fig_dist.add_hline(y=0, line_color='#6b7a8f', line_dash='dash')
                    fig_dist.update_layout(
                        **dark_plotly_layout(height=400,
                                             title="P&L Distribution by Regime ($)",
                                             yaxis_title="P&L per Trade ($)")
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                    # ── Raw data expander ────────────────────────────────────
                    with st.expander("📋 Raw Trade Log"):
                        st.dataframe(
                            valid[['date', 'regime', 'S', 'S_T', 'move_pct', 'IV_proxy', 'pnl']]
                            .sort_values('date', ascending=False)
                            .style.format({
                                'S': '${:.2f}', 'S_T': '${:.2f}',
                                'move_pct': '{:+.2f}%', 'IV_proxy': '{:.1f}%',
                                'pnl': '${:+.0f}'
                            }),
                            use_container_width=True, hide_index=True
                        )
                else:
                    st.warning("Not enough historical data to run backtest. Try a longer lookback period.")

else:
    st.info("👈 Enter a ticker symbol in the sidebar and click 'Load Options Chain' to get started")

    st.markdown("""
    ### What You Can Do Here

    **Pull live options data from yfinance API:**

    1. **Full Options Chain**
       - All available strikes for selected expiration
       - Live bid/ask/last prices
       - Volume and open interest
       - Calculated Greeks for every strike
       - Export to CSV

    2. **High Volume Options**
       - Most actively traded options
       - Filter by minimum volume
       - Visual volume charts

    3. **IV Smile/Skew**
       - Implied volatility by strike
       - Detect put/call skew
       - Identify mispricing opportunities

    4. **Quick Analyze**
       - Select any option from the chain
       - Instant Greeks calculation
       - Fair value analysis (Black-Scholes vs. market price)

    ### Data Source

    - **API:** yfinance (free, public data)
    - **Update Frequency:** Real-time during market hours
    - **Coverage:** All US-listed options
    - **No API key required**

    ### Try Popular Tickers

    - **SPY** - S&P 500 ETF (most liquid)
    - **QQQ** - Nasdaq 100 ETF
    - **AAPL** - Apple Inc.
    - **TSLA** - Tesla Inc.
    - **NVDA** - NVIDIA Corp.
    """)
