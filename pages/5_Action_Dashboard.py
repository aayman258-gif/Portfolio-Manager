"""
Regime-Aware Portfolio Manager
Section 5: Action Dashboard
Single-page view: "What should I do today?"
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

from calculations.regime_detector import RegimeDetector
from calculations.scoring_engine import ScoringEngine
from calculations.optimizer import RegimeAwareOptimizer
from data.portfolio_loader import PortfolioLoader
from data.market_data import MarketDataLoader


# Page config
st.set_page_config(
    page_title="Action Dashboard",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Action Dashboard")
st.markdown("**What should I do today?**")

# Check if portfolio exists
if 'positions' not in st.session_state:
    st.warning("⚠️ No portfolio loaded. Please go to Home page and load a portfolio first.")
    st.stop()

positions_df = st.session_state['positions']

# Initialize
detector = RegimeDetector()
scorer = ScoringEngine()
optimizer = RegimeAwareOptimizer()
loader = PortfolioLoader()
market_loader = MarketDataLoader()

# Load market data and detect regime
@st.cache_data
def get_market_context():
    """Get comprehensive market context"""
    spy_data = market_loader.load_index_data('SPY', '2y')
    vix_data = market_loader.load_vix_data('2y')
    spy_prices, vix_prices = market_loader.align_data(spy_data, vix_data)
    regime, signals = detector.classify_regime(spy_prices, vix_prices)

    current_regime = regime.iloc[-1]
    current_vix = vix_prices.iloc[-1]
    current_vol = signals['realized_vol'].iloc[-1]

    # Get market stats
    market_stats = market_loader.get_current_market_stats(spy_data, current_vix)

    # Get regime description
    regime_info = detector.get_regime_description(current_regime)

    return current_regime, regime_info, market_stats

current_regime, regime_info, market_stats = get_market_context()

# Fetch current prices and calculate metrics
with st.spinner("Analyzing your portfolio..."):
    tickers = positions_df['ticker'].tolist()
    current_prices = loader.fetch_current_prices(tickers)
    position_metrics = loader.calculate_position_metrics(positions_df, current_prices)
    summary = loader.get_portfolio_summary(position_metrics)

# === SECTION 1: MARKET REGIME & QUICK STATS ===
st.subheader("🌍 Market Status")

regime_colors = {
    'Low Vol': '🟢',
    'High Vol': '🔴',
    'Trending': '🔵',
    'Mean Reversion': '🟡',
    'Unknown': '⚪'
}

emoji = regime_colors.get(current_regime, '⚪')

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"<div style='text-align: center; font-size: 40px;'>{emoji}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center;'><strong>{current_regime}</strong></div>", unsafe_allow_html=True)

with col2:
    vix_color = "🔴" if market_stats['vix_current'] > 30 else "🟡" if market_stats['vix_current'] > 20 else "🟢"
    st.metric("VIX", f"{vix_color} {market_stats['vix_current']:.1f}")

with col3:
    st.metric("SPY (1D)", f"{market_stats['returns_1d']:+.2f}%")

with col4:
    st.metric("From 52W High", f"{market_stats['distance_from_high']:+.1f}%")

st.info(f"**Strategy:** {regime_info['strategy']}")

# === SECTION 2: PORTFOLIO HEALTH ===
st.subheader("💼 Portfolio Health")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Value",
        f"${summary['total_value']:,.2f}",
        f"{summary['total_pnl_pct']:+.2f}%"
    )

with col2:
    st.metric(
        "P&L",
        f"${summary['total_pnl']:,.2f}",
        delta_color="normal" if summary['total_pnl'] > 0 else "inverse"
    )

with col3:
    st.metric(
        "Positions",
        summary['num_positions'],
        f"{summary['winners']}W / {summary['losers']}L"
    )

with col4:
    largest_weight = summary['largest_position_weight']
    concentration_warning = largest_weight > 30
    st.metric(
        "Largest Position",
        summary['largest_position_ticker'],
        f"{largest_weight:.1f}%",
        delta_color="inverse" if concentration_warning else "off"
    )

# === SECTION 3: ACTION ITEMS ===
st.subheader("⚡ Action Items")

actions = []
priority_scores = []

# Check 1: Regime-based exposure
regime_constraints = optimizer.regime_constraints.get(current_regime, {})
max_equity = regime_constraints.get('max_equity_exposure', 1.0)
current_equity = (summary['total_value'] - 0) / summary['total_value']  # Assuming no cash position tracked

if current_equity > max_equity:
    excess = (current_equity - max_equity) * summary['total_value']
    actions.append({
        'priority': 'HIGH',
        'type': '⚠️ Reduce Exposure',
        'action': f"Current regime ({current_regime}) recommends max {max_equity*100:.0f}% equity exposure. You're at ~100%. Consider raising ${excess:,.0f} in cash.",
        'score': 90
    })
    priority_scores.append(90)

# Check 2: Concentration risk
if largest_weight > 30:
    actions.append({
        'priority': 'MEDIUM',
        'type': '📊 Diversification',
        'action': f"{summary['largest_position_ticker']} represents {largest_weight:.1f}% of portfolio. Consider trimming to under 30%.",
        'score': 70
    })
    priority_scores.append(70)

# Check 3: Score positions and find issues
with st.spinner("Scoring positions..."):
    position_scores = []

    for _, position in positions_df.iterrows():
        ticker = position['ticker']
        try:
            historical = loader.fetch_historical_data([ticker], start_date=datetime.now() - timedelta(days=365))
            if historical[ticker] is not None and not historical[ticker].empty:
                prices = historical[ticker]['Close']
                score_result = scorer.calculate_unified_score(ticker, prices, current_regime)
                position_scores.append({
                    'ticker': ticker,
                    'score': score_result['unified_score']
                })
        except:
            pass

# Find low-scoring positions
if position_scores:
    scores_df = pd.DataFrame(position_scores)
    low_scores = scores_df[scores_df['score'] < 35]

    if not low_scores.empty:
        for _, row in low_scores.iterrows():
            actions.append({
                'priority': 'HIGH',
                'type': '💀 Underperformer',
                'action': f"Consider selling {row['ticker']} (score: {row['score']:.0f}/100). Poor fit for {current_regime} regime.",
                'score': 85
            })
            priority_scores.append(85)

    # Find high-scoring positions
    high_scores = scores_df[scores_df['score'] >= 80]
    if not high_scores.empty:
        for _, row in high_scores.head(2).iterrows():
            actions.append({
                'priority': 'LOW',
                'type': '✅ Strong Position',
                'action': f"{row['ticker']} scores {row['score']:.0f}/100. Strong fit for current regime. Hold or add.",
                'score': 20
            })
            priority_scores.append(20)

# Check 4: VIX warning
if market_stats['vix_current'] > 30:
    actions.append({
        'priority': 'HIGH',
        'type': '🔴 High Volatility',
        'action': f"VIX at {market_stats['vix_current']:.1f} (elevated fear). Consider defensive positioning and increased cash.",
        'score': 95
    })
    priority_scores.append(95)

# Check 5: Market near 52-week high
if market_stats['distance_from_high'] > -5:
    actions.append({
        'priority': 'LOW',
        'type': '📈 Near Highs',
        'action': f"SPY within 5% of 52-week high. Markets strong. Monitor for overbought conditions.",
        'score': 30
    })
    priority_scores.append(30)

# Check 6: No action needed
if not actions or (actions and max(priority_scores) < 50):
    actions.append({
        'priority': 'NONE',
        'type': '✅ All Clear',
        'action': f"Portfolio looks healthy for {current_regime} regime. No immediate action required.",
        'score': 0
    })

# Sort actions by priority score
actions_df = pd.DataFrame(actions)
if not actions_df.empty:
    actions_df = actions_df.sort_values('score', ascending=False)

# Display actions
for _, action in actions_df.iterrows():
    priority = action['priority']

    if priority == 'HIGH':
        st.error(f"**{action['type']}:** {action['action']}")
    elif priority == 'MEDIUM':
        st.warning(f"**{action['type']}:** {action['action']}")
    elif priority == 'LOW':
        st.info(f"**{action['type']}:** {action['action']}")
    else:
        st.success(f"**{action['type']}:** {action['action']}")

# === SECTION 4: TOP 3 POSITIONS ===
st.subheader("🏆 Top Positions by Value")

top_positions = position_metrics.nlargest(3, 'current_value')

cols = st.columns(3)

for idx, (_, pos) in enumerate(top_positions.iterrows()):
    with cols[idx]:
        pnl_color = "green" if pos['total_pnl'] > 0 else "red"

        st.markdown(f"""
        <div style='border: 1px solid #ddd; padding: 15px; border-radius: 10px;'>
            <h3 style='margin: 0;'>{pos['ticker']}</h3>
            <p style='font-size: 24px; margin: 5px 0; color: {pnl_color};'>${pos['current_value']:,.0f}</p>
            <p style='margin: 0;'>Weight: {pos['weight_pct']:.1f}%</p>
            <p style='margin: 0; color: {pnl_color};'>P&L: ${pos['total_pnl']:,.0f} ({pos['total_pnl_pct']:+.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)

# === SECTION 5: QUICK METRICS ===
st.subheader("📊 Quick Metrics")

col1, col2 = st.columns(2)

with col1:
    # P&L distribution
    fig_pnl = go.Figure(data=[go.Bar(
        x=position_metrics['ticker'],
        y=position_metrics['total_pnl'],
        marker_color=position_metrics['total_pnl'].apply(lambda x: 'green' if x > 0 else 'red'),
        text=position_metrics['total_pnl'],
        texttemplate='$%{text:,.0f}',
        textposition='outside'
    )])

    fig_pnl.update_layout(
        title="Position P&L",
        xaxis_title="",
        yaxis_title="P&L ($)",
        height=300,
        template='plotly_white',
        showlegend=False
    )

    fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")

    st.plotly_chart(fig_pnl, use_container_width=True)

with col2:
    # Allocation pie
    fig_pie = go.Figure(data=[go.Pie(
        labels=position_metrics['ticker'],
        values=position_metrics['current_value'],
        hole=0.4,
        marker=dict(
            colors=position_metrics['total_pnl'].apply(
                lambda x: '#00CC00' if x > 0 else '#FF6B6B'
            )
        )
    )])

    fig_pie.update_layout(
        title="Portfolio Allocation",
        height=300,
        showlegend=True
    )

    st.plotly_chart(fig_pie, use_container_width=True)

# === SECTION 6: NAVIGATION ===
st.subheader("🔗 Deep Dive")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 📋 Details")
    st.markdown("View full position breakdown with P&L and metrics")
    st.page_link("Home.py", label="→ Position Tracker", icon="💼")

with col2:
    st.markdown("### 🌍 Regime")
    st.markdown("Analyze market regime and sector performance")
    st.page_link("pages/2_Market_Regime_Dashboard.py", label="→ Market Dashboard", icon="🌍")

with col3:
    st.markdown("### ⭐ Scores")
    st.markdown("See quant + fundamental scores for each position")
    st.page_link("pages/3_Position_Scoring.py", label="→ Position Scoring", icon="⭐")

with col4:
    st.markdown("### ⚖️ Optimize")
    st.markdown("Generate optimal allocation and rebalancing trades")
    st.page_link("pages/4_Optimization_Rebalancing.py", label="→ Optimization", icon="⚖️")

# === FOOTER ===
st.divider()

st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Portfolio Value:</strong> ${summary['total_value']:,.2f} | <strong>P&L:</strong> ${summary['total_pnl']:,.2f} ({summary['total_pnl_pct']:+.2f}%)</p>
    <p><strong>Current Regime:</strong> {emoji} {current_regime} | <strong>VIX:</strong> {market_stats['vix_current']:.1f}</p>
    <p style='font-size: 12px; margin-top: 10px;'>Built with DRIVER Framework | Regime-Aware Portfolio Management</p>
</div>
""", unsafe_allow_html=True)

# Educational tip
with st.expander("💡 How to Use This Dashboard"):
    st.markdown(f"""
    ## Daily Workflow

    **1. Check Market Regime (Top)**
    - Current regime: **{current_regime}**
    - Strategy: {regime_info['strategy']}
    - Adjust your mindset based on regime

    **2. Review Action Items**
    - Red (HIGH priority): Take action today
    - Yellow (MEDIUM): Address soon
    - Blue (LOW): Monitor
    - Green: All clear

    **3. Check Portfolio Health**
    - Total value and P&L trending correctly?
    - Winner/loser ratio acceptable?
    - Any concentration risk?

    **4. Review Top Positions**
    - Are your largest positions still strong?
    - Check scores in Section 3 if concerned

    **5. Take Action**
    - Follow high-priority recommendations
    - Use deep dive links for details
    - Execute trades if needed

    ---

    ## When to Act

    **Act immediately if:**
    - Regime changed (check Market Dashboard)
    - High-priority alerts (red)
    - VIX spike (>30)
    - Large drawdown on key position

    **Review weekly:**
    - Position scores (Section 3)
    - Rebalancing needs (Section 4)
    - Sector performance

    **Review monthly:**
    - Full optimization
    - Tax loss harvesting opportunities
    - Performance vs. benchmark

    ---

    ## This Dashboard Answers:

    - ✅ "What's the market environment?" → Check regime
    - ✅ "How's my portfolio doing?" → Health metrics
    - ✅ "What should I do today?" → Action items
    - ✅ "Which positions need attention?" → Alerts + scores
    - ✅ "Do I need to rebalance?" → Optimization section

    **Everything else is noise. Focus on these questions.**
    """)
