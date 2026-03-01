"""
Regime-Aware Portfolio Manager
Section 9: Portfolio Hedges — Options-Based Downside Protection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent))

from data.options_data import OptionsDataLoader
from calculations.options_analytics import OptionsAnalytics
from calculations.regime_detector import RegimeDetector
from data.market_data import MarketDataLoader
from utils.carbon_theme import apply_carbon_theme, carbon_plotly_layout, page_header

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Hedges",
    page_icon="🛡️",
    layout="wide"
)
apply_carbon_theme()

page_header("🛡️ Portfolio Hedge Analyzer", "Build options-based downside protection from the live chain, with full scenario analysis")

# ── Helpers ──────────────────────────────────────────────────────────────────
options_loader = OptionsDataLoader()
options_calc   = OptionsAnalytics()
market_loader  = MarketDataLoader()
regime_detector = RegimeDetector()


@st.cache_data(show_spinner="Fetching regime…")
def get_regime():
    spy_data  = market_loader.load_index_data('SPY', '1y')
    vix_data  = market_loader.load_vix_data('1y')
    spy_px, vix_px = market_loader.align_data(spy_data, vix_data)
    regime, _ = regime_detector.classify_regime(spy_px, vix_px)
    return regime.iloc[-1]


@st.cache_data(show_spinner="Loading SPY options chain…")
def get_spy_options(expiration):
    return options_loader.get_options_chain('SPY', expiration)


current_regime = get_regime()

regime_colors = {
    'Low Vol': '#22d3ee', 'High Vol': '#fb7185',
    'Trending': '#22d3ee', 'Mean Reversion': '#f59e0b', 'Unknown': '#888888'
}
rcolor = regime_colors.get(current_regime, '#888888')
st.markdown(
    f"""<div style="background:{rcolor}22;border:1px solid {rcolor}55;border-radius:8px;
    padding:10px 18px;margin-bottom:12px;font-size:15px;">
    🌍 <strong>Current Regime:</strong> {current_regime}
    — adjust hedge size accordingly.</div>""",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PORTFOLIO INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Portfolio Details")

col_pv, col_beta, col_spy = st.columns(3)

with col_pv:
    portfolio_value = st.number_input(
        "Total Portfolio Value ($)",
        min_value=1_000.0, max_value=100_000_000.0,
        value=100_000.0, step=5_000.0,
        format="%.0f"
    )

with col_beta:
    portfolio_beta = st.slider(
        "Portfolio Beta vs SPY",
        min_value=0.3, max_value=2.5, value=1.0, step=0.05,
        help="Beta = 1 means portfolio moves 1:1 with SPY"
    )

with col_spy:
    spy_summary = options_loader.get_options_summary('SPY')
    spy_price   = spy_summary.get('current_price', 580.0)
    st.metric("SPY Current Price", f"${spy_price:.2f}")

# Optional position detail table
with st.expander("📊 Optional: Detailed Portfolio Positions", expanded=False):
    default_positions = pd.DataFrame({
        'Ticker':  ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN'],
        'Shares':  [50,      30,      20,      15,      25],
        'Avg Cost':[175.00,  380.00,  650.00,  170.00,  185.00],
    })
    positions_df = st.data_editor(
        default_positions, num_rows="dynamic",
        use_container_width=True, hide_index=True,
        key="positions_editor"
    )
    if not positions_df.empty:
        positions_df['Value'] = positions_df['Shares'] * positions_df['Avg Cost']
        total_pos = positions_df['Value'].sum()
        st.metric("Total Position Value", f"${total_pos:,.0f}",
                  f"vs. manual input: ${portfolio_value:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — HEDGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("⚙️ Hedge Configuration")

col_exp, col_prot, col_budget = st.columns(3)

with col_exp:
    expirations = options_loader.get_options_expirations('SPY')
    exp_options = []
    for exp in expirations[:8]:
        dte = (datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days
        if dte >= 14:
            exp_options.append(f"{exp} ({dte}d)")
    selected_exp_str = st.selectbox("Hedge Expiration", exp_options,
                                    index=min(2, len(exp_options) - 1))
    selected_exp = selected_exp_str.split(' ')[0]
    exp_dte = int(selected_exp_str.split('(')[1].replace('d)', ''))
    T_hedge = max(exp_dte / 365, 0.003)

with col_prot:
    protection_pct = st.slider(
        "Protection Strike (% below spot)",
        min_value=2.0, max_value=20.0, value=5.0, step=0.5,
        help="E.g. 5% → buy put at SPY × 0.95"
    )

with col_budget:
    collar_call_pct = st.slider(
        "Collar Call Strike (% above spot)",
        min_value=2.0, max_value=20.0, value=5.0, step=0.5,
        help="Sell a call this far OTM to reduce hedge cost (collar only)"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Fetch live options & compute hedge structures
# ─────────────────────────────────────────────────────────────────────────────
calls_spy, puts_spy, spy_live = get_spy_options(selected_exp)

if puts_spy.empty or calls_spy.empty:
    st.error("Could not load SPY options chain. Please refresh.")
    st.stop()

spy_price = spy_live if spy_live > 0 else spy_price  # prefer live price

# Number of contracts needed to cover portfolio delta
contracts_needed = int(np.ceil(portfolio_value * portfolio_beta / (spy_price * 100)))

# Helper: find nearest strike in chain
def _nearest_put(target_K):
    diff = (puts_spy['strike'] - target_K).abs()
    row  = puts_spy.loc[diff.idxmin()]
    return float(row['strike']), float(row.get('lastPrice', 0) or row.get('midPrice', 0) or 0.01)

def _nearest_call(target_K):
    diff = (calls_spy['strike'] - target_K).abs()
    row  = calls_spy.loc[diff.idxmin()]
    return float(row['strike']), float(row.get('lastPrice', 0) or row.get('midPrice', 0) or 0.01)

# Retrieve live option IV (fallback to realized vol estimate)
def _get_iv(chain_df, strike, option_type):
    diff = (chain_df['strike'] - strike).abs()
    row  = chain_df.loc[diff.idxmin()]
    iv   = float(row.get('impliedVolatility', 0.20) or 0.20)
    return max(iv, 0.05)

# ── Protective Put ────────────────────────────────────────────────────────────
put_K,       put_mkt_prem  = _nearest_put(spy_price * (1 - protection_pct / 100))
put_iv                     = _get_iv(puts_spy, put_K, 'put')
put_bs_prem                = options_calc.black_scholes(spy_price, put_K, T_hedge, put_iv, 'put')
put_prem_used              = put_mkt_prem if put_mkt_prem > 0.01 else put_bs_prem
put_total_cost             = put_prem_used * contracts_needed * 100
put_annual_cost            = put_total_cost * (365 / exp_dte)

# ── Put Spread ────────────────────────────────────────────────────────────────
put_spread_short_K,  ps_short_prem = _nearest_put(spy_price * (1 - 2 * protection_pct / 100))
ps_short_iv                        = _get_iv(puts_spy, put_spread_short_K, 'put')
ps_short_bs                        = options_calc.black_scholes(
    spy_price, put_spread_short_K, T_hedge, ps_short_iv, 'put'
)
ps_short_used              = ps_short_prem if ps_short_prem > 0.01 else ps_short_bs
put_spread_net             = put_prem_used - ps_short_used
put_spread_total           = put_spread_net * contracts_needed * 100
put_spread_annual          = put_spread_total * (365 / exp_dte)

# ── Collar ────────────────────────────────────────────────────────────────────
call_K,       call_mkt_prem = _nearest_call(spy_price * (1 + collar_call_pct / 100))
call_iv                     = _get_iv(calls_spy, call_K, 'call')
call_bs_prem                = options_calc.black_scholes(spy_price, call_K, T_hedge, call_iv, 'call')
call_prem_used              = call_mkt_prem if call_mkt_prem > 0.01 else call_bs_prem
collar_net                  = put_prem_used - call_prem_used
collar_total                = collar_net * contracts_needed * 100
collar_annual               = collar_total * (365 / exp_dte)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — HEDGE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🛡️ Recommended Hedge Strategies")

# Regime-based urgency message
urgency = {
    'High Vol':       ("⚠️ HIGH URGENCY", "ef4444",
                       "Volatility is elevated — full protection recommended."),
    'Trending':       ("📈 MODERATE", "4a9eff",
                       "Directional trend in place — collar or put spread sufficient."),
    'Low Vol':        ("🟢 LOW URGENCY", "22c55e",
                       "Calm markets — light hedge or no hedge needed."),
    'Mean Reversion': ("🟡 MODERATE", "f59e0b",
                       "Choppy conditions — put spread recommended."),
    'Unknown':        ("❓ UNKNOWN", "6b7a8f",
                       "Insufficient regime data — maintain moderate protection."),
}
urg_label, urg_col, urg_msg = urgency.get(current_regime, urgency['Unknown'])
st.markdown(
    f"""<div style="background:#{urg_col}22;border:1px solid #{urg_col}55;
    border-radius:8px;padding:10px 18px;margin-bottom:16px;">
    <strong>{urg_label}</strong> — {urg_msg}</div>""",
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("#### 🔴 Protective Put")
    st.markdown(f"Buy **{contracts_needed} puts** @ ${put_K:.0f} strike")
    st.metric("Put Premium (mid)", f"${put_prem_used:.2f}/contract")
    st.metric("Total Cost", f"${put_total_cost:,.0f}")
    st.metric("Est. Annual Cost", f"${put_annual_cost:,.0f}",
              f"{put_annual_cost / portfolio_value * 100:.2f}% of portfolio")
    st.markdown(f"**Protects against:** drops below ${put_K:.0f}  \n"
                f"**Strike:** {protection_pct:.1f}% OTM  \n"
                f"**Break-even decline:** {(put_prem_used / spy_price * 100):.2f}%")

with c2:
    st.markdown("#### 🟡 Put Spread")
    st.markdown(f"Buy **${put_K:.0f}** put, Sell **${put_spread_short_K:.0f}** put")
    st.metric("Net Premium", f"${put_spread_net:.2f}/contract")
    st.metric("Total Cost", f"${put_spread_total:,.0f}")
    st.metric("Est. Annual Cost", f"${put_spread_annual:,.0f}",
              f"{put_spread_annual / portfolio_value * 100:.2f}% of portfolio")
    st.markdown(f"**Max protection:** ${(put_K - put_spread_short_K) * contracts_needed * 100:,.0f}  \n"
                f"**Protection zone:** ${put_spread_short_K:.0f} – ${put_K:.0f}  \n"
                f"**Cheaper than straight put by:** "
                f"{(put_prem_used - put_spread_net) / put_prem_used * 100:.0f}%")

with c3:
    st.markdown("#### 🔵 Collar")
    st.markdown(f"Buy **${put_K:.0f}** put, Sell **${call_K:.0f}** call")
    col_label = "Net Credit" if collar_net < 0 else "Net Debit"
    st.metric(col_label, f"${abs(collar_net):.2f}/contract")
    st.metric("Total Cost", f"${abs(collar_total):,.0f}",
              "credit received" if collar_net < 0 else "debit paid")
    st.metric("Est. Annual Cost", f"${abs(collar_annual):,.0f}",
              f"{abs(collar_annual) / portfolio_value * 100:.2f}% of portfolio")
    st.markdown(f"**Downside protected below:** ${put_K:.0f}  \n"
                f"**Upside capped at:** ${call_K:.0f}  \n"
                f"**Net position:** {'Slight income' if collar_net < 0 else 'Small cost'}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Scenario Analysis — Portfolio + Hedge P&L")
st.markdown(
    "Shows how your portfolio value changes under different market moves, "
    "with and without each hedge strategy."
)

moves = np.linspace(-0.30, 0.30, 61)   # -30% to +30% in 0.5% steps

unhedged_pnl    = portfolio_value * portfolio_beta * moves

# Protective put P&L at each scenario
put_payoff      = np.array([max(put_K - spy_price * (1 + m), 0) for m in moves])
pp_hedge_pnl    = (put_payoff - put_prem_used) * contracts_needed * 100
pp_total        = unhedged_pnl + pp_hedge_pnl

# Put spread P&L at each scenario
ps_long_payoff  = np.array([max(put_K                - spy_price * (1 + m), 0) for m in moves])
ps_short_payoff = np.array([max(put_spread_short_K   - spy_price * (1 + m), 0) for m in moves])
ps_hedge_pnl    = (ps_long_payoff - ps_short_payoff - put_spread_net) * contracts_needed * 100
ps_total        = unhedged_pnl + ps_hedge_pnl

# Collar P&L at each scenario
collar_put_pay  = np.array([max(put_K  - spy_price * (1 + m), 0) for m in moves])
collar_call_pay = np.array([max(spy_price * (1 + m) - call_K, 0) for m in moves])
collar_hedge_pnl= (collar_put_pay - collar_call_pay - collar_net) * contracts_needed * 100
collar_total    = unhedged_pnl + collar_hedge_pnl

move_pcts = moves * 100

fig_scen = go.Figure()

# Shaded region below 0
fig_scen.add_hrect(y0=-portfolio_value * 0.4, y1=0,
                   fillcolor='rgba(251,113,133,0.04)', line_width=0)

# Unhedged
fig_scen.add_trace(go.Scatter(
    x=move_pcts, y=unhedged_pnl,
    name='Unhedged Portfolio', mode='lines',
    line=dict(color='#888888', width=2, dash='dot')
))

# Protective Put
fig_scen.add_trace(go.Scatter(
    x=move_pcts, y=pp_total,
    name=f'Protective Put (${put_K:.0f})', mode='lines',
    line=dict(color='#fb7185', width=2.5)
))

# Put Spread
fig_scen.add_trace(go.Scatter(
    x=move_pcts, y=ps_total,
    name=f'Put Spread ({put_K:.0f}/{put_spread_short_K:.0f})', mode='lines',
    line=dict(color='#f59e0b', width=2.5)
))

# Collar
fig_scen.add_trace(go.Scatter(
    x=move_pcts, y=collar_total,
    name=f'Collar ({put_K:.0f}p / {call_K:.0f}c)', mode='lines',
    line=dict(color='#22d3ee', width=2.5)
))

fig_scen.add_hline(y=0, line_color='#ffffff', line_width=1, line_dash='dash')
fig_scen.add_vline(x=0,  line_color='#888888', line_width=1, line_dash='dot')

fig_scen.update_layout(
    **carbon_plotly_layout(
        height=520,
        title="Portfolio P&L vs SPY Move — All Hedge Strategies",
        xaxis_title="SPY Move (%)",
        yaxis_title="Portfolio P&L ($)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15)
    )
)
fig_scen.update_xaxes(
    tickformat="+.0f",
    ticksuffix="%",
    gridcolor='rgba(107,122,143,0.15)'
)
fig_scen.update_yaxes(
    tickprefix="$",
    tickformat=",.0f",
    gridcolor='rgba(107,122,143,0.15)'
)

st.plotly_chart(fig_scen, use_container_width=True)

# ── Scenario table ────────────────────────────────────────────────────────────
st.markdown("### Scenario Table")

key_moves = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]
key_idx   = [np.argmin(np.abs(moves - m)) for m in key_moves]

scen_rows = []
for i in key_idx:
    m = moves[i]
    scen_rows.append({
        'Market Move':      f"{m:+.0%}",
        'SPY Target':       f"${spy_price * (1 + m):.0f}",
        'Unhedged P&L':     unhedged_pnl[i],
        'Protective Put':   pp_total[i],
        'Put Spread':       ps_total[i],
        'Collar':           collar_total[i],
    })

scen_df = pd.DataFrame(scen_rows)

def _color_pnl(val):
    try:
        v = float(val.replace('$', '').replace(',', ''))
        return f'color: {"#22d3ee" if v >= 0 else "#fb7185"}'
    except Exception:
        return ''

pnl_cols = ['Unhedged P&L', 'Protective Put', 'Put Spread', 'Collar']
styled = scen_df.style.format({c: '${:+,.0f}' for c in pnl_cols}) \
                      .applymap(_color_pnl, subset=pnl_cols)

st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Hedge effectiveness summary ───────────────────────────────────────────────
st.subheader("📈 Hedge Effectiveness Summary")

worst_case_move = -0.20
wi = np.argmin(np.abs(moves - worst_case_move))

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Unhedged Loss (–20%)",
    f"${unhedged_pnl[wi]:+,.0f}",
    f"{unhedged_pnl[wi] / portfolio_value * 100:+.1f}%"
)
c2.metric(
    "Protective Put Loss (–20%)",
    f"${pp_total[wi]:+,.0f}",
    f"Saved ${pp_total[wi] - unhedged_pnl[wi]:+,.0f}"
)
c3.metric(
    "Put Spread Loss (–20%)",
    f"${ps_total[wi]:+,.0f}",
    f"Saved ${ps_total[wi] - unhedged_pnl[wi]:+,.0f}"
)
c4.metric(
    "Collar Loss (–20%)",
    f"${collar_total[wi]:+,.0f}",
    f"Saved ${collar_total[wi] - unhedged_pnl[wi]:+,.0f}"
)

# ── Regime-based recommendation ───────────────────────────────────────────────
with st.expander("💡 Regime-Based Hedge Recommendation"):
    recommendations = {
        'High Vol': {
            'rec': 'Protective Put',
            'reason': 'Volatility is elevated and downside risk is real. '
                      'Full put protection justifies the higher premium cost.',
            'sizing': '100% of calculated contracts',
        },
        'Trending': {
            'rec': 'Put Spread',
            'reason': 'Strong trend reduces crash risk. A put spread gives '
                      'affordable protection for the most likely drawdown zone.',
            'sizing': '75–100% of calculated contracts',
        },
        'Low Vol': {
            'rec': 'Collar (light)',
            'reason': 'Calm environment. A collar costs almost nothing in low vol '
                      'and provides a safety net without significant upside drag.',
            'sizing': '50–75% of calculated contracts',
        },
        'Mean Reversion': {
            'rec': 'Put Spread',
            'reason': 'Choppy markets can produce sudden drops. '
                      'A put spread balances cost and protection in range-bound conditions.',
            'sizing': '75% of calculated contracts',
        },
        'Unknown': {
            'rec': 'Protective Put (reduced size)',
            'reason': 'Regime unclear — maintain baseline protection at reduced notional.',
            'sizing': '50% of calculated contracts',
        },
    }
    rec = recommendations.get(current_regime, recommendations['Unknown'])

    st.markdown(f"""
    **Regime:** {current_regime}

    **Recommended Strategy:** {rec['rec']}

    **Why:** {rec['reason']}

    **Suggested Sizing:** {rec['sizing']} ({contracts_needed} contracts full size)

    ---

    **Cost comparison for {exp_dte}-day horizon:**
    | Strategy | Cost | Annual Cost | Portfolio % |
    |---|---|---|---|
    | Protective Put | ${put_total_cost:,.0f} | ${put_annual_cost:,.0f} | {put_annual_cost/portfolio_value*100:.2f}% |
    | Put Spread     | ${put_spread_total:,.0f} | ${put_spread_annual:,.0f} | {put_spread_annual/portfolio_value*100:.2f}% |
    | Collar         | ${abs(collar_total):,.0f} | ${abs(collar_annual):,.0f} | {abs(collar_annual)/portfolio_value*100:.2f}% |
    """)
