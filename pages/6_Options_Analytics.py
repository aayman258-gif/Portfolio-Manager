"""
Regime-Aware Portfolio Manager
Section 6: Options Analytics & Recommendations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.carbon_theme import apply_carbon_theme, carbon_plotly_layout, page_header

from calculations.options_analytics import OptionsAnalytics
from calculations.options_recommender import OptionsRecommender
from calculations.regime_detector import RegimeDetector
from data.market_data import MarketDataLoader


# Page config
st.set_page_config(
    page_title="Options Analytics",
    page_icon="📊",
    layout="wide"
)
apply_carbon_theme()

page_header("📊 Options Analytics & Trading", "Analyze options positions with Greeks and get regime-aware recommendations")

# Initialize
options_calc = OptionsAnalytics(risk_free_rate=0.045)
recommender = OptionsRecommender()
detector = RegimeDetector()
market_loader = MarketDataLoader()

# Get current regime
@st.cache_data
def get_current_regime():
    """Get current market regime"""
    spy_data = market_loader.load_index_data('SPY', '2y')
    vix_data = market_loader.load_vix_data('2y')
    spy_prices, vix_prices = market_loader.align_data(spy_data, vix_data)
    regime, signals = detector.classify_regime(spy_prices, vix_prices)
    current_vol = signals['realized_vol'].iloc[-1]
    return regime.iloc[-1], current_vol

current_regime, market_volatility = get_current_regime()

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["📈 Options Recommendations", "🔢 Analyze Position", "📊 Portfolio Greeks"])

# === TAB 1: OPTIONS RECOMMENDATIONS ===
with tab1:
    st.subheader(f"🎯 Options Strategies for {current_regime} Regime")

    st.info(f"**Current Regime:** {current_regime} | **Market Volatility:** {market_volatility*100:.1f}%")

    # Get strategies for regime
    strategies = recommender.get_strategies_for_regime(current_regime)

    # Display each strategy
    for idx, strategy in enumerate(strategies):
        with st.expander(f"**{strategy['name']}** - {strategy['type']}", expanded=(idx == 0)):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Description:** {strategy['description']}")
                st.markdown(f"**Structure:** {strategy['structure']}")
                st.markdown(f"**Best When:** {strategy['best_when']}")

            with col2:
                st.metric("Risk Level", strategy['risk'])
                st.metric("Complexity", strategy['complexity'])

            # Get strike recommendations
            st.markdown("### 📍 Strike Recommendations")

            underlying_price = st.number_input(
                f"Underlying Price ({strategy['name']})",
                min_value=1.0,
                value=450.0,
                step=1.0,
                key=f"price_{idx}"
            )

            col_a, col_b = st.columns(2)

            with col_a:
                days_to_expiry = st.slider(
                    "Days to Expiration",
                    min_value=7,
                    max_value=90,
                    value=30,
                    key=f"dte_{idx}"
                )

            with col_b:
                trend = st.selectbox(
                    "Market Trend",
                    options=[0, 1, -1],
                    format_func=lambda x: "Sideways" if x == 0 else "Uptrend" if x == 1 else "Downtrend",
                    key=f"trend_{idx}"
                )

            # Calculate strikes
            strike_recs = recommender.calculate_strike_recommendations(
                current_price=underlying_price,
                regime=current_regime,
                volatility=market_volatility,
                trend_direction=trend,
                days_to_expiry=days_to_expiry
            )

            if strategy['name'] in strike_recs:
                strikes = strike_recs[strategy['name']]

                st.success(f"**Rationale:** {strikes.get('rationale', 'N/A')}")

                # Display strikes
                strike_data = {k: v for k, v in strikes.items() if k != 'rationale'}
                if strike_data:
                    strike_df = pd.DataFrame([strike_data])
                    st.dataframe(strike_df, use_container_width=True, hide_index=True)

    # Position sizing
    st.subheader("💰 Position Sizing Guide")

    account_size = st.number_input(
        "Account Size ($)",
        min_value=1000,
        max_value=10000000,
        value=50000,
        step=1000
    )

    sizing = recommender.get_position_sizing_guide(current_regime, account_size)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Max Risk Per Trade",
            f"${sizing['max_dollar_risk']:,.0f}",
            f"{sizing['max_risk_per_trade']*100:.1f}%"
        )

    with col2:
        st.metric("Max Positions", sizing['max_positions'])

    with col3:
        total_risk = sizing['max_dollar_risk'] * sizing['max_positions']
        st.metric(
            "Max Total Risk",
            f"${total_risk:,.0f}",
            f"{total_risk/account_size*100:.1f}%"
        )

    st.info(f"**Note:** {sizing['notes']}")

# === TAB 2: ANALYZE POSITION ===
with tab2:
    st.subheader("🔍 Analyze Individual Option Position")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Position Details")

        underlying_price = st.number_input(
            "Current Underlying Price ($)",
            min_value=1.0,
            value=450.0,
            step=0.50
        )

        strike_price = st.number_input(
            "Strike Price ($)",
            min_value=1.0,
            value=455.0,
            step=1.0
        )

        option_type = st.selectbox(
            "Option Type",
            options=['call', 'put']
        )

        contracts = st.number_input(
            "Contracts (negative for short)",
            min_value=-1000,
            max_value=1000,
            value=1,
            step=1
        )

    with col2:
        st.markdown("### Time & Volatility")

        expiration_date = st.date_input(
            "Expiration Date",
            value=datetime.now() + timedelta(days=30)
        )

        days_to_exp = (expiration_date - datetime.now().date()).days
        time_to_exp = max(days_to_exp / 365, 0.001)  # In years

        st.metric("Days to Expiration", days_to_exp)

        implied_vol = st.slider(
            "Implied Volatility (%)",
            min_value=5.0,
            max_value=150.0,
            value=30.0,
            step=1.0
        ) / 100

        risk_free = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=4.5,
            step=0.1
        ) / 100

        dividend_yield = st.slider(
            "Dividend Yield (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        ) / 100

    # Optional: Premium and market price for P&L
    st.markdown("### Optional: P&L Analysis")
    col_a, col_b = st.columns(2)

    with col_a:
        premium_paid = st.number_input(
            "Premium Paid per Contract ($, optional)",
            min_value=0.0,
            value=0.0,
            step=0.10
        )

    with col_b:
        current_market_price = st.number_input(
            "Current Market Price ($, optional)",
            min_value=0.0,
            value=0.0,
            step=0.10
        )

    # Set to None if 0
    premium_paid = premium_paid if premium_paid > 0 else None
    current_market_price = current_market_price if current_market_price > 0 else None

    # Analyze button
    if st.button("🔍 Analyze Position", type="primary"):
        # Update calculator with risk-free rate
        options_calc.risk_free_rate = risk_free

        # Analyze position
        analysis = options_calc.analyze_option_position(
            S=underlying_price,
            K=strike_price,
            T=time_to_exp,
            sigma=implied_vol,
            option_type=option_type,
            contracts=contracts,
            premium_paid=premium_paid,
            current_price=current_market_price,
            q=dividend_yield
        )

        # Display results
        st.success("✅ Analysis Complete")

        # Key metrics
        st.subheader("📊 Key Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Theoretical Value",
                f"${analysis['theoretical_value']:.2f}"
            )

        with col2:
            st.metric(
                "Intrinsic Value",
                f"${analysis['intrinsic_value']:.2f}"
            )

        with col3:
            st.metric(
                "Time Value",
                f"${analysis['time_value']:.2f}"
            )

        with col4:
            st.metric(
                "Moneyness",
                analysis['moneyness']
            )

        # P&L if available
        if analysis['pnl'] is not None:
            st.subheader("💰 Profit & Loss")

            col1, col2 = st.columns(2)

            with col1:
                pnl_color = "green" if analysis['pnl'] > 0 else "red"
                st.markdown(f"<h2 style='color: {pnl_color};'>${analysis['pnl']:,.2f}</h2>", unsafe_allow_html=True)
                st.caption(f"P&L: {analysis['pnl_pct']:+.2f}%")

            with col2:
                total_value = abs(contracts) * current_market_price * 100 if current_market_price else 0
                st.metric("Position Value", f"${total_value:,.0f}")

        # Fair value analysis
        if analysis['fair_value_analysis']:
            st.subheader("⚖️ Fair Value Analysis")

            fv = analysis['fair_value_analysis']

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Market Price", f"${fv['market_price']:.2f}")

            with col2:
                st.metric("Theoretical Value", f"${fv['theoretical_value']:.2f}")

            with col3:
                diff_color = "green" if fv['difference'] < 0 else "red"
                st.metric(
                    "Difference",
                    f"${fv['difference']:.2f}",
                    f"{fv['difference_pct']:+.2f}%",
                    delta_color="inverse" if fv['difference'] > 0 else "normal"
                )

            if fv['rating'] in ['Overpriced', 'Bad Short']:
                st.warning(f"**Rating:** {fv['rating']}")
            elif fv['rating'] in ['Underpriced', 'Good Short']:
                st.success(f"**Rating:** {fv['rating']}")
            else:
                st.info(f"**Rating:** {fv['rating']}")

        # Greeks
        st.subheader("🔢 The Greeks")

        greeks = analysis['greeks']
        position_greeks = analysis['position_greeks']

        # Per-contract Greeks
        st.markdown("### Per Contract")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Delta", f"{greeks['delta']:.4f}")
            interp = options_calc.get_greek_interpretation('delta', greeks['delta'])
            st.caption(interp['description'])

        with col2:
            st.metric("Gamma", f"{greeks['gamma']:.4f}")
            interp = options_calc.get_greek_interpretation('gamma', greeks['gamma'])
            st.caption(interp['description'])

        with col3:
            st.metric("Theta", f"${greeks['theta']:.2f}")
            interp = options_calc.get_greek_interpretation('theta', greeks['theta'])
            st.caption(interp['description'])

        with col4:
            st.metric("Vega", f"${greeks['vega']:.2f}")
            interp = options_calc.get_greek_interpretation('vega', greeks['vega'])
            st.caption(interp['description'])

        with col5:
            st.metric("Rho", f"${greeks['rho']:.2f}")
            interp = options_calc.get_greek_interpretation('rho', greeks['rho'])
            st.caption(interp['description'])

        # Position Greeks
        st.markdown("### Total Position")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            delta_color = "green" if position_greeks['delta'] > 0 else "red"
            st.markdown(f"<h3 style='color: {delta_color};'>{position_greeks['delta']:.2f}</h3>", unsafe_allow_html=True)
            st.caption("Position Delta")

        with col2:
            st.metric("Position Gamma", f"{position_greeks['gamma']:.2f}")

        with col3:
            theta_color = "green" if position_greeks['theta'] > 0 else "red"
            st.markdown(f"<h3 style='color: {theta_color};'>${position_greeks['theta']:.2f}</h3>", unsafe_allow_html=True)
            st.caption("Daily Theta")

        with col4:
            st.metric("Position Vega", f"${position_greeks['vega']:.2f}")

        with col5:
            st.metric("Position Rho", f"${position_greeks['rho']:.2f}")

        # Interpretations
        with st.expander("📚 Understanding Your Greeks"):
            st.markdown("### Delta")
            st.write(options_calc.get_greek_interpretation('delta', position_greeks['delta'])['interpretation'])

            st.markdown("### Gamma")
            st.write(options_calc.get_greek_interpretation('gamma', position_greeks['gamma'])['interpretation'])

            st.markdown("### Theta")
            st.write(options_calc.get_greek_interpretation('theta', position_greeks['theta'])['interpretation'])

            st.markdown("### Vega")
            st.write(options_calc.get_greek_interpretation('vega', position_greeks['vega'])['interpretation'])

            st.markdown("### Rho")
            st.write(options_calc.get_greek_interpretation('rho', position_greeks['rho'])['interpretation'])

        # Visualizations
        st.subheader("📈 Price Sensitivity Analysis")

        # Create price range
        price_range = np.linspace(underlying_price * 0.8, underlying_price * 1.2, 50)
        option_values = []
        deltas = []

        for price in price_range:
            value = options_calc.black_scholes(price, strike_price, time_to_exp, implied_vol, option_type, risk_free, dividend_yield)
            option_values.append(value * contracts * 100)  # Total position value

            greek = options_calc.calculate_greeks(price, strike_price, time_to_exp, implied_vol, option_type, risk_free, dividend_yield)
            deltas.append(greek['delta'])

        # Plot
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Position Value vs Underlying Price', 'Delta vs Underlying Price'),
            vertical_spacing=0.1
        )

        # Position value
        fig.add_trace(
            go.Scatter(x=price_range, y=option_values, name='Position Value', line=dict(color='blue', width=2)),
            row=1, col=1
        )

        fig.add_vline(x=underlying_price, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_vline(x=strike_price, line_dash="dash", line_color="red", row=1, col=1)

        # Delta
        fig.add_trace(
            go.Scatter(x=price_range, y=deltas, name='Delta', line=dict(color='green', width=2)),
            row=2, col=1
        )

        fig.add_vline(x=underlying_price, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_vline(x=strike_price, line_dash="dash", line_color="red", row=2, col=1)

        fig.update_layout(**carbon_plotly_layout(height=600, showlegend=False))
        fig.update_xaxes(title_text="Underlying Price ($)", row=2, col=1)
        fig.update_yaxes(title_text="Position Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Delta", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

# === TAB 3: PORTFOLIO GREEKS ===
with tab3:
    st.subheader("📊 Portfolio-Level Greeks Analysis")

    st.info("🚧 **Feature:** Track multiple options positions and calculate portfolio-level Greeks")

    st.markdown("""
    ### Coming Soon: Portfolio Greeks Tracker

    **Features:**
    - Track multiple options positions simultaneously
    - Calculate total portfolio Delta, Gamma, Theta, Vega, Rho
    - Visualize net exposure and risk
    - Greeks-based rebalancing recommendations

    **For now, analyze individual positions in the "Analyze Position" tab**

    **Workaround:**
    1. Analyze each position individually
    2. Sum the Position Greeks manually
    3. Monitor total portfolio exposure
    """)

    # Placeholder for future implementation
    st.markdown("### Manual Portfolio Greeks Calculator")

    num_positions = st.number_input("Number of Options Positions", min_value=1, max_value=10, value=1)

    total_delta = 0
    total_theta = 0
    total_vega = 0

    st.markdown("**Quick estimate (simplified):**")

    for i in range(num_positions):
        col1, col2, col3 = st.columns(3)

        with col1:
            delta = st.number_input(f"Position {i+1} Delta", value=0.0, key=f"delta_{i}")

        with col2:
            theta = st.number_input(f"Position {i+1} Theta", value=0.0, key=f"theta_{i}")

        with col3:
            vega = st.number_input(f"Position {i+1} Vega", value=0.0, key=f"vega_{i}")

        total_delta += delta
        total_theta += theta
        total_vega += vega

    st.divider()

    st.markdown("### Total Portfolio Greeks")

    col1, col2, col3 = st.columns(3)

    with col1:
        delta_color = "green" if total_delta > 0 else "red" if total_delta < 0 else "gray"
        st.markdown(f"<h2 style='color: {delta_color};'>{total_delta:.2f}</h2>", unsafe_allow_html=True)
        st.caption("Total Delta")

    with col2:
        theta_color = "green" if total_theta > 0 else "red"
        st.markdown(f"<h2 style='color: {theta_color};'>${total_theta:.2f}</h2>", unsafe_allow_html=True)
        st.caption("Daily Theta")

    with col3:
        st.markdown(f"<h2>${total_vega:.2f}</h2>", unsafe_allow_html=True)
        st.caption("Total Vega")

# Educational content
with st.expander("📚 Options Trading Guide"):
    st.markdown(f"""
    ## Options Trading in {current_regime} Regime

    ### Current Market Environment

    **Regime:** {current_regime}
    **Volatility:** {market_volatility*100:.1f}%

    ---

    ### Regime-Based Strategy Selection

    **Why Regime Matters for Options:**

    Different market regimes require different options strategies because:
    - **Volatility levels** determine premium pricing
    - **Trend strength** affects directional plays
    - **Range vs. breakout** environments favor different structures

    ---

    ### The Greeks Explained

    **Delta (Δ):** Price sensitivity
    - Range: 0 to 1 (calls), 0 to -1 (puts)
    - Position Delta = net directional exposure
    - +50 Delta = Long 50 shares equivalent

    **Gamma (Γ):** Delta acceleration
    - Highest for ATM options near expiration
    - Long options = positive gamma (good)
    - Short options = negative gamma (risk)

    **Theta (Θ):** Time decay
    - Always negative for long options
    - Accelerates near expiration
    - Short options benefit from theta

    **Vega (V):** IV sensitivity
    - Long options = positive vega (want IV ↑)
    - Short options = negative vega (want IV ↓)
    - Critical in high vol regimes

    **Rho (ρ):** Interest rate sensitivity
    - Usually small impact
    - More relevant for LEAPS

    ---

    ### Black-Scholes Model

    **What it does:**
    - Calculates theoretical "fair value" of options
    - Uses: Stock price, Strike, Time, Volatility, Risk-free rate

    **Limitations:**
    - Assumes constant volatility (not true in reality)
    - Assumes log-normal distribution (fat tails in real markets)
    - Ignores dividends (we use simplified version)

    **How to use it:**
    - Compare market price to theoretical value
    - If market > theoretical → Overpriced
    - If market < theoretical → Underpriced
    - But remember: IV can change!

    ---

    ### Risk Management

    **Position Sizing:**
    - Never risk more than 2-3% on single trade
    - Reduce size in High Vol regimes
    - Account for undefined risk strategies

    **Greeks Management:**
    - Monitor total portfolio Delta (directional exposure)
    - Watch Theta decay on long positions
    - Hedge Vega in volatile environments

    **Exit Rules:**
    - Take profits at 50% of max gain
    - Cut losses at 2x credit received (spreads)
    - Don't hold to expiration (gamma risk)

    ---

    ### Strategy Selection by Regime

    Use the recommendations in the first tab for regime-appropriate strategies!
    """)
