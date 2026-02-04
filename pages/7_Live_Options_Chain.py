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

from data.options_data import OptionsDataLoader
from calculations.options_analytics import OptionsAnalytics
from calculations.strategy_builder import StrategyBuilder


# Page config
st.set_page_config(
    page_title="Live Options Chain",
    page_icon="🔴",
    layout="wide"
)

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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Options Chain",
            "🔥 High Volume Options",
            "📈 IV Smile",
            "⚡ Quick Analyze",
            "🔧 Strategy Builder"
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
                    template='plotly_white'
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
                    template='plotly_white',
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
                            template='plotly_white',
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
