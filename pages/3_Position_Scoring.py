"""
Regime-Aware Portfolio Manager
Section 3: Unified Scoring Engine
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.theme import apply_dark_theme

from calculations.scoring_engine import ScoringEngine
from calculations.regime_detector import RegimeDetector
from data.portfolio_loader import PortfolioLoader
from data.market_data import MarketDataLoader


# Page config
st.set_page_config(
    page_title="Position Scoring",
    page_icon="⭐",
    layout="wide"
)
apply_dark_theme()

st.title("⭐ Position Scoring Engine")
st.markdown("**Section 3:** Score each position on quant signals and fundamental metrics")

# Check if portfolio exists
if 'positions' not in st.session_state:
    st.warning("⚠️ No portfolio loaded. Please go to Home page and load a portfolio first.")
    st.stop()

positions_df = st.session_state['positions']

# Initialize
scorer = ScoringEngine()
loader = PortfolioLoader()
market_loader = MarketDataLoader()
detector = RegimeDetector()

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

st.subheader(f"📊 Current Market Regime: {current_regime}")

# Display regime weighting
regime_weights = scorer.regime_weights.get(current_regime, {'quant': 0.5, 'fundamental': 0.5})

col1, col2 = st.columns(2)
with col1:
    st.metric("Quant Weight", f"{regime_weights['quant']*100:.0f}%")
with col2:
    st.metric("Fundamental Weight", f"{regime_weights['fundamental']*100:.0f}%")

st.info(f"**{current_regime} Regime:** Weighting quant signals at {regime_weights['quant']*100:.0f}% and fundamentals at {regime_weights['fundamental']*100:.0f}%")

# Score all positions
st.subheader("📋 Position Scores")

with st.spinner("Scoring all positions... This may take a moment."):

    scores_data = []

    for _, position in positions_df.iterrows():
        ticker = position['ticker']

        try:
            # Fetch historical prices
            historical = loader.fetch_historical_data([ticker], start_date=datetime.now() - timedelta(days=365))

            if historical[ticker] is not None and not historical[ticker].empty:
                prices = historical[ticker]['Close']

                # Calculate unified score
                score_result = scorer.calculate_unified_score(ticker, prices, current_regime)

                # Get interpretation
                rating, color = scorer.get_score_interpretation(score_result['unified_score'])

                scores_data.append({
                    'Ticker': ticker,
                    'Unified Score': score_result['unified_score'],
                    'Rating': rating,
                    'Quant Score': score_result['quant_score'],
                    'Fundamental Score': score_result['fundamental_score'],
                    'Momentum': score_result['momentum_score'],
                    'Volatility': score_result['volatility_score'],
                    'Regime Fit': score_result['regime_fit_score'],
                    'Growth': score_result['growth_score'],
                    'Quality': score_result['quality_score'],
                    'Valuation': score_result['valuation_score'],
                    'Color': color,
                    'Full_Data': score_result
                })
            else:
                st.warning(f"Could not fetch data for {ticker}")

        except Exception as e:
            st.warning(f"Error scoring {ticker}: {str(e)}")

# Create scores dataframe
if scores_data:
    scores_df = pd.DataFrame(scores_data)
    scores_df = scores_df.sort_values('Unified Score', ascending=False)

    # Display summary table
    display_df = scores_df[[
        'Ticker', 'Unified Score', 'Rating', 'Quant Score',
        'Fundamental Score', 'Momentum', 'Volatility',
        'Regime Fit', 'Growth', 'Quality', 'Valuation'
    ]].copy()

    # Style the dataframe
    def color_score(val):
        if val >= 80:
            return 'background-color: lightgreen'
        elif val >= 65:
            return 'background-color: #90EE90'
        elif val >= 50:
            return 'background-color: lightyellow'
        elif val >= 35:
            return 'background-color: #FFB366'
        else:
            return 'background-color: #FFB3B3'

    styled_df = display_df.style.format({
        'Unified Score': '{:.1f}',
        'Quant Score': '{:.1f}',
        'Fundamental Score': '{:.1f}',
        'Momentum': '{:.1f}',
        'Volatility': '{:.1f}',
        'Regime Fit': '{:.1f}',
        'Growth': '{:.1f}',
        'Quality': '{:.1f}',
        'Valuation': '{:.1f}'
    }).applymap(color_score, subset=['Unified Score'])

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Score visualization
    st.subheader("📊 Score Comparison")

    col1, col2 = st.columns(2)

    with col1:
        # Bar chart of unified scores
        colors_list = scores_df['Color'].tolist()

        fig_scores = go.Figure(data=[go.Bar(
            x=scores_df['Ticker'],
            y=scores_df['Unified Score'],
            text=scores_df['Unified Score'],
            texttemplate='%{text:.1f}',
            textposition='outside',
            marker_color=colors_list
        )])

        fig_scores.update_layout(
            title="Unified Scores by Position",
            xaxis_title="Ticker",
            yaxis_title="Score (0-100)",
            height=400,
            template='plotly_dark'
        )

        fig_scores.add_hline(y=80, line_dash="dash", line_color="green", opacity=0.5)
        fig_scores.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)

        st.plotly_chart(fig_scores, use_container_width=True)

    with col2:
        # Quant vs Fundamental scatter
        fig_scatter = go.Figure(data=[go.Scatter(
            x=scores_df['Quant Score'],
            y=scores_df['Fundamental Score'],
            mode='markers+text',
            text=scores_df['Ticker'],
            textposition='top center',
            marker=dict(
                size=15,
                color=scores_df['Unified Score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Unified Score")
            )
        )])

        fig_scatter.update_layout(
            title="Quant vs Fundamental Scores",
            xaxis_title="Quant Score",
            yaxis_title="Fundamental Score",
            height=400,
            template='plotly_dark'
        )

        fig_scatter.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3)
        fig_scatter.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.3)

        st.plotly_chart(fig_scatter, use_container_width=True)

    # Detailed breakdown for each position
    st.subheader("🔍 Detailed Position Analysis")

    selected_ticker = st.selectbox(
        "Select position to analyze:",
        options=scores_df['Ticker'].tolist()
    )

    selected_data = scores_df[scores_df['Ticker'] == selected_ticker].iloc[0]
    full_data = selected_data['Full_Data']

    # Display scores
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Unified Score")
        rating, color = scorer.get_score_interpretation(full_data['unified_score'])
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{full_data['unified_score']:.1f}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>{rating}</h3>", unsafe_allow_html=True)

    with col2:
        st.markdown("### Quant Score")
        st.metric("Overall", f"{full_data['quant_score']:.1f}")
        st.metric("Momentum", f"{full_data['momentum_score']:.1f}")
        st.metric("Volatility", f"{full_data['volatility_score']:.1f}")
        st.metric("Regime Fit", f"{full_data['regime_fit_score']:.1f}")

    with col3:
        st.markdown("### Fundamental Score")
        st.metric("Overall", f"{full_data['fundamental_score']:.1f}")
        st.metric("Growth", f"{full_data['growth_score']:.1f}")
        st.metric("Quality", f"{full_data['quality_score']:.1f}")
        st.metric("Valuation", f"{full_data['valuation_score']:.1f}")

    # Radar chart
    st.markdown("### Score Breakdown")

    categories = ['Momentum', 'Volatility', 'Regime Fit', 'Growth', 'Quality', 'Valuation']
    values = [
        full_data['momentum_score'],
        full_data['volatility_score'],
        full_data['regime_fit_score'],
        full_data['growth_score'],
        full_data['quality_score'],
        full_data['valuation_score']
    ]

    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='blue'
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    # Fundamental details
    st.markdown("### 📊 Fundamental Metrics")

    fundamentals = full_data['fundamentals']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Valuation**")
        st.write(f"P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}")
        st.write(f"Forward P/E: {fundamentals.get('forward_pe', 'N/A')}")
        st.write(f"PEG Ratio: {fundamentals.get('peg_ratio', 'N/A')}")
        st.write(f"Price/Book: {fundamentals.get('price_to_book', 'N/A')}")

    with col2:
        st.markdown("**Profitability**")
        profit_margin = fundamentals.get('profit_margin', None)
        st.write(f"Profit Margin: {profit_margin*100:.2f}%" if profit_margin else "N/A")

        roe = fundamentals.get('roe', None)
        st.write(f"ROE: {roe*100:.2f}%" if roe else "N/A")

        roa = fundamentals.get('roa', None)
        st.write(f"ROA: {roa*100:.2f}%" if roa else "N/A")

    with col3:
        st.markdown("**Growth & Health**")
        rev_growth = fundamentals.get('revenue_growth', None)
        st.write(f"Revenue Growth: {rev_growth*100:.2f}%" if rev_growth else "N/A")

        debt_eq = fundamentals.get('debt_to_equity', None)
        st.write(f"Debt/Equity: {debt_eq:.2f}" if debt_eq else "N/A")

        current_ratio = fundamentals.get('current_ratio', None)
        st.write(f"Current Ratio: {current_ratio:.2f}" if current_ratio else "N/A")

    # Action recommendation
    st.markdown("### 💡 Recommendation")

    if full_data['unified_score'] >= 80:
        st.success(f"""
        **Strong Buy:** {selected_ticker} scores very highly ({full_data['unified_score']:.1f}/100) in the current {current_regime} regime.
        - Consider increasing position size
        - Strong fundamentals and good regime fit
        """)
    elif full_data['unified_score'] >= 65:
        st.success(f"""
        **Buy:** {selected_ticker} scores well ({full_data['unified_score']:.1f}/100) and fits the current regime.
        - Good candidate to hold or add
        - Monitor for any regime changes
        """)
    elif full_data['unified_score'] >= 50:
        st.info(f"""
        **Hold:** {selected_ticker} has a neutral score ({full_data['unified_score']:.1f}/100).
        - Maintain current position
        - Re-evaluate if regime changes
        """)
    elif full_data['unified_score'] >= 35:
        st.warning(f"""
        **Sell:** {selected_ticker} scores below average ({full_data['unified_score']:.1f}/100).
        - Consider reducing position
        - May not fit current {current_regime} regime well
        """)
    else:
        st.error(f"""
        **Strong Sell:** {selected_ticker} scores poorly ({full_data['unified_score']:.1f}/100).
        - Consider exiting position
        - Poor fit for current market regime
        """)

    # Export
    st.subheader("💾 Export Scores")

    export_df = display_df.copy()
    csv = export_df.to_csv(index=False)

    st.download_button(
        label="Download Position Scores (CSV)",
        data=csv,
        file_name=f"position_scores_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    st.error("No position scores calculated. Please check your portfolio data.")

# Educational content
with st.expander("📚 How the Scoring System Works"):
    st.markdown("""
    ## Unified Scoring Methodology

    ### Quantitative Score (0-100)

    **1. Momentum Score (40% weight)**
    - Multiple timeframe returns (1M, 3M, 6M)
    - Positive momentum = higher score

    **2. Volatility Score (30% weight)**
    - 60-day realized volatility
    - Lower volatility = higher score (stability)

    **3. Regime Fit Score (30% weight)**
    - How well stock fits current market regime
    - Different criteria for each regime:
      - Low Vol: Favor stable, low volatility
      - High Vol: Favor defensive, quality
      - Trending: Favor momentum alignment
      - Mean Reversion: Favor oversold quality

    ---

    ### Fundamental Score (0-100)

    **1. Growth Score (35% weight)**
    - Revenue and earnings growth rates
    - Higher growth = higher score

    **2. Quality Score (35% weight)**
    - Profit margins, ROE, ROA
    - Financial health (debt ratios, liquidity)
    - Higher quality = higher score

    **3. Valuation Score (30% weight)**
    - P/E, PEG, Price/Book ratios
    - Lower multiples = higher score (better value)

    ---

    ### Unified Score Calculation

    **Weighted by Current Regime:**

    - **Low Vol Regime:** 30% Quant + 70% Fundamental (value-focused)
    - **High Vol Regime:** 40% Quant + 60% Fundamental (quality-focused)
    - **Trending Regime:** 70% Quant + 30% Fundamental (momentum-focused)
    - **Mean Reversion:** 50% Quant + 50% Fundamental (balanced)

    **Final Score:** `(Quant × Quant_Weight) + (Fundamental × Fundamental_Weight)`

    ---

    ### Score Interpretation

    - **80-100:** Strong Buy - Excellent on both quant and fundamental metrics
    - **65-79:** Buy - Good overall fit for current regime
    - **50-64:** Hold - Neutral, maintain position
    - **35-49:** Sell - Below average, consider reducing
    - **0-34:** Strong Sell - Poor fit, exit position

    ---

    ### Why Regime-Based Weighting?

    Different market regimes favor different characteristics:

    - **Trending markets:** Momentum matters more than valuation
    - **Low vol markets:** Fundamentals and value matter more
    - **High vol markets:** Quality and safety matter most

    The scoring system adapts to prioritize what works in the current environment.
    """)
