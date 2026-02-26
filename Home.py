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
import yfinance as yf

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data.portfolio_loader import PortfolioLoader
from calculations.performance import PerformanceAnalytics
from calculations.options_analytics import OptionsAnalytics
from utils.theme import apply_dark_theme, dark_plotly_layout
from utils.portfolio_store import (
    save_portfolio, load_portfolio, portfolio_file_exists, get_last_saved_time,
    save_options_positions, load_options_positions,
)

# ── Multi-leg strategy templates ─────────────────────────────────────────────
_STRAT_TEMPLATES = {
    'Iron Condor': {
        'desc': 'Sell OTM put spread + OTM call spread (net credit)',
        'calendar': False,
        'legs': [
            {'option_type': 'put',  'position': 'short', 'offset': -0.05, 'label': 'Short OTM Put'},
            {'option_type': 'put',  'position': 'long',  'offset': -0.10, 'label': 'Long Far-OTM Put'},
            {'option_type': 'call', 'position': 'short', 'offset':  0.05, 'label': 'Short OTM Call'},
            {'option_type': 'call', 'position': 'long',  'offset':  0.10, 'label': 'Long Far-OTM Call'},
        ],
    },
    'Bull Call Spread': {
        'desc': 'Buy ATM call, sell OTM call (net debit)',
        'calendar': False,
        'legs': [
            {'option_type': 'call', 'position': 'long',  'offset': 0.00, 'label': 'Long ATM Call'},
            {'option_type': 'call', 'position': 'short', 'offset': 0.05, 'label': 'Short OTM Call'},
        ],
    },
    'Bear Call Spread': {
        'desc': 'Sell OTM call, buy further OTM call (net credit)',
        'calendar': False,
        'legs': [
            {'option_type': 'call', 'position': 'short', 'offset': 0.05, 'label': 'Short OTM Call'},
            {'option_type': 'call', 'position': 'long',  'offset': 0.10, 'label': 'Long Far-OTM Call'},
        ],
    },
    'Bull Put Spread': {
        'desc': 'Sell OTM put, buy further OTM put (net credit)',
        'calendar': False,
        'legs': [
            {'option_type': 'put', 'position': 'short', 'offset': -0.05, 'label': 'Short OTM Put'},
            {'option_type': 'put', 'position': 'long',  'offset': -0.10, 'label': 'Long Far-OTM Put'},
        ],
    },
    'Bear Put Spread': {
        'desc': 'Buy ATM put, sell OTM put (net debit)',
        'calendar': False,
        'legs': [
            {'option_type': 'put', 'position': 'long',  'offset':  0.00, 'label': 'Long ATM Put'},
            {'option_type': 'put', 'position': 'short', 'offset': -0.05, 'label': 'Short OTM Put'},
        ],
    },
    'Long Straddle': {
        'desc': 'Buy ATM call + put — profits from big move (net debit)',
        'calendar': False,
        'legs': [
            {'option_type': 'call', 'position': 'long', 'offset': 0.00, 'label': 'Long ATM Call'},
            {'option_type': 'put',  'position': 'long', 'offset': 0.00, 'label': 'Long ATM Put'},
        ],
    },
    'Short Straddle': {
        'desc': 'Sell ATM call + put — profits from no move (net credit)',
        'calendar': False,
        'legs': [
            {'option_type': 'call', 'position': 'short', 'offset': 0.00, 'label': 'Short ATM Call'},
            {'option_type': 'put',  'position': 'short', 'offset': 0.00, 'label': 'Short ATM Put'},
        ],
    },
    'Long Strangle': {
        'desc': 'Buy OTM call + put — cheaper than straddle (net debit)',
        'calendar': False,
        'legs': [
            {'option_type': 'call', 'position': 'long', 'offset':  0.05, 'label': 'Long OTM Call'},
            {'option_type': 'put',  'position': 'long', 'offset': -0.05, 'label': 'Long OTM Put'},
        ],
    },
    'Short Strangle': {
        'desc': 'Sell OTM call + put (net credit)',
        'calendar': False,
        'legs': [
            {'option_type': 'call', 'position': 'short', 'offset':  0.05, 'label': 'Short OTM Call'},
            {'option_type': 'put',  'position': 'short', 'offset': -0.05, 'label': 'Short OTM Put'},
        ],
    },
    'Calendar Call Spread': {
        'desc': 'Sell near-term call, buy same-strike far-term call (net debit)',
        'calendar': True,
        'legs': [
            {'option_type': 'call', 'position': 'short', 'offset': 0.00, 'label': 'Short Near-Term Call', 'far': False},
            {'option_type': 'call', 'position': 'long',  'offset': 0.00, 'label': 'Long Far-Term Call',   'far': True},
        ],
    },
    'Calendar Put Spread': {
        'desc': 'Sell near-term put, buy same-strike far-term put (net debit)',
        'calendar': True,
        'legs': [
            {'option_type': 'put', 'position': 'short', 'offset': 0.00, 'label': 'Short Near-Term Put', 'far': False},
            {'option_type': 'put', 'position': 'long',  'offset': 0.00, 'label': 'Long Far-Term Put',   'far': True},
        ],
    },
    'Custom': {
        'desc': 'Build your own multi-leg strategy leg by leg',
        'calendar': False,
        'legs': [],
    },
}

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

# ── Options Positions Input ──────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("📊 Add Options Position")

# Initialise options state once per session
if 'options_positions' not in st.session_state:
    st.session_state['options_positions'] = []
if '_opts_loaded' not in st.session_state:
    _saved_opts = load_options_positions()
    if _saved_opts:
        st.session_state['options_positions'] = _saved_opts
    st.session_state['_opts_loaded'] = True

# ── Step 1: Ticker ───────────────────────────────────────────────────────────
_add_ul = st.sidebar.text_input(
    "Underlying Ticker", key="_add_opt_ul", placeholder="e.g. AAPL"
).upper().strip()

if st.sidebar.button("🔍 Fetch Options Chain", key="_fetch_chain_btn"):
    if _add_ul:
        try:
            _s = yf.Ticker(_add_ul)
            _exps = list(_s.options)
            if _exps:
                st.session_state['_opt_exps']      = _exps
                st.session_state['_opt_ul_loaded'] = _add_ul
                # Clear downstream state on new ticker fetch
                for _k in ['_opt_chain', '_opt_sel_exp', '_opt_sel_type']:
                    st.session_state.pop(_k, None)
            else:
                st.sidebar.error(f"No options found for {_add_ul}")
        except Exception as _e:
            st.sidebar.error(f"Error fetching chain: {_e}")
    else:
        st.sidebar.warning("Enter a ticker first.")

# ── Step 2: Expiry + Type + Load Strikes ────────────────────────────────────
if st.session_state.get('_opt_exps'):
    _sel_exp  = st.sidebar.selectbox(
        "Expiration Date", st.session_state['_opt_exps'], key="_sel_opt_exp"
    )
    _sel_type = st.sidebar.radio(
        "Option Type", ["call", "put"], horizontal=True, key="_sel_opt_type"
    )

    if st.sidebar.button("📋 Load Strikes", key="_load_strikes_btn"):
        try:
            _s2    = yf.Ticker(st.session_state['_opt_ul_loaded'])
            _chain = _s2.option_chain(_sel_exp)
            _df    = _chain.calls if _sel_type == 'call' else _chain.puts
            if not _df.empty:
                st.session_state['_opt_chain']    = _df.reset_index(drop=True).to_dict(orient='records')
                st.session_state['_opt_sel_exp']  = _sel_exp
                st.session_state['_opt_sel_type'] = _sel_type
            else:
                st.sidebar.error("No contracts found for this expiry/type.")
        except Exception as _e:
            st.sidebar.error(f"Error loading strikes: {_e}")

# ── Step 3: Strike selector + Entry form ────────────────────────────────────
if st.session_state.get('_opt_chain'):
    _chain_df = pd.DataFrame(st.session_state['_opt_chain'])

    def _strike_label(i):
        r   = _chain_df.iloc[i]
        bid = float(r.get('bid', 0) or 0)
        ask = float(r.get('ask', 0) or 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else float(r.get('lastPrice', 0) or 0)
        iv  = float(r.get('impliedVolatility', 0) or 0) * 100
        oi  = int(r.get('openInterest', 0) or 0)
        return f"${r['strike']:.2f}  mid ${mid:.2f}  IV {iv:.0f}%  OI {oi:,}"

    _sel_idx = st.sidebar.selectbox(
        "Strike", range(len(_chain_df)),
        format_func=_strike_label, key="_sel_strike_idx"
    )
    _sel_row  = _chain_df.iloc[_sel_idx]
    _bid      = float(_sel_row.get('bid', 0) or 0)
    _ask      = float(_sel_row.get('ask', 0) or 0)
    _last     = float(_sel_row.get('lastPrice', 0) or 0)
    _mid      = (_bid + _ask) / 2 if _bid > 0 and _ask > 0 else _last
    _iv       = float(_sel_row.get('impliedVolatility', 0.3) or 0.3)
    _oi       = int(_sel_row.get('openInterest', 0) or 0)
    _vol      = int(_sel_row.get('volume', 0) or 0)

    st.sidebar.info(
        f"**Market:** Mid ${_mid:.2f}  |  IV {_iv*100:.0f}%  |  "
        f"OI {_oi:,}  |  Vol {_vol:,}"
    )

    with st.sidebar.form("add_option_form", clear_on_submit=True):
        _contracts  = st.number_input("Contracts", min_value=1, value=1, step=1)
        _cost_basis = st.number_input(
            "Your Cost Basis ($/share)",
            min_value=0.0, value=round(_mid, 2), step=0.01,
            help="Price YOU paid per share. Each contract = 100 shares."
        )
        _submitted = st.form_submit_button("➕ Add Options Position")

        if _submitted:
            st.session_state['options_positions'].append({
                'underlying':  st.session_state['_opt_ul_loaded'],
                'option_type': st.session_state['_opt_sel_type'],
                'strike':      float(_sel_row['strike']),
                'expiration':  st.session_state['_opt_sel_exp'],
                'contracts':   int(_contracts),
                'cost_basis':  float(_cost_basis),
                'iv_at_entry': _iv,
                'status':      'open',
                'close_method': None,
                'close_price':  None,
                'close_date':   None,
            })
            save_options_positions(st.session_state['options_positions'])
            st.rerun()

    if st.sidebar.button("🔄 New Search", key="_reset_chain_btn"):
        for _k in ['_opt_exps', '_opt_ul_loaded', '_opt_chain',
                   '_opt_sel_exp', '_opt_sel_type']:
            st.session_state.pop(_k, None)
        st.rerun()

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

    # ── OPTIONS POSITIONS ────────────────────────────────────────────────────
    st.subheader("📊 Options Positions")

    _opts = st.session_state.get('options_positions', [])
    _oa   = OptionsAnalytics()

    # ── Helper: live price lookup for one contract ────────────────────────────
    def _live_option(underlying, option_type, strike, expiration, iv_fallback=0.3):
        try:
            _s  = yf.Ticker(underlying)
            _c  = _s.option_chain(expiration)
            _df = _c.calls if option_type == 'call' else _c.puts
            _df = _df.copy()
            _df['_d'] = abs(_df['strike'] - strike)
            _r  = _df.nsmallest(1, '_d').iloc[0]
            _b  = float(_r.get('bid', 0) or 0)
            _a  = float(_r.get('ask', 0) or 0)
            _l  = float(_r.get('lastPrice', 0) or 0)
            _m  = (_b + _a) / 2 if _b > 0 and _a > 0 else _l
            _iv = float(_r.get('impliedVolatility', iv_fallback) or iv_fallback)
            return (_m if _m > 0 else None), _iv
        except Exception:
            return None, iv_fallback

    # Separate single-leg vs strategy positions
    _singles    = [(i, p) for i, p in enumerate(_opts) if p.get('type', 'single') == 'single']
    _strategies = [(i, p) for i, p in enumerate(_opts) if p.get('type') == 'strategy']

    # Collect all open underlying tickers
    _open_uls = set()
    for _, _p in _singles:
        if _p['status'] == 'open':
            _open_uls.add(_p['underlying'])
    for _, _p in _strategies:
        if _p['status'] == 'open':
            _open_uls.add(_p['underlying'])
    _ul_prices: dict = loader.fetch_current_prices(list(_open_uls)) if _open_uls else {}

    _total_opt_val = 0.0
    _total_opt_pnl = 0.0

    # ── Strategy Builder toggle ───────────────────────────────────────────────
    if '_sb_show' not in st.session_state:
        st.session_state['_sb_show'] = False

    _hcol1, _hcol2 = st.columns([4, 1])
    with _hcol2:
        if st.button(
            "✖ Close Builder" if st.session_state['_sb_show'] else "🔀 Add Strategy",
            key="_strat_toggle"
        ):
            st.session_state['_sb_show'] = not st.session_state['_sb_show']
            if not st.session_state['_sb_show']:
                for _k in ['_sb_exps','_sb_ul_loaded','_sb_near_chain','_sb_far_chain',
                           '_sb_calls_chain','_sb_puts_chain','_sb_single_exp_loaded',
                           '_sb_near_exp_loaded','_sb_far_exp_loaded','_sb_custom_legs']:
                    st.session_state.pop(_k, None)
            st.rerun()

    # ── Strategy Builder UI ───────────────────────────────────────────────────
    if st.session_state['_sb_show']:
        with st.container(border=True):
            st.markdown("#### 🔧 Multi-Leg Strategy Builder")

            _sb_c1, _sb_c2, _sb_c3 = st.columns([2, 2, 1])
            with _sb_c1:
                _sb_tmpl = st.selectbox(
                    "Strategy Template", list(_STRAT_TEMPLATES.keys()), key="_sb_tmpl_sel"
                )
                st.caption(_STRAT_TEMPLATES[_sb_tmpl]['desc'])
            with _sb_c2:
                _sb_ul = st.text_input(
                    "Underlying Ticker", key="_sb_ul_in", placeholder="e.g. SPY"
                ).upper().strip()
            with _sb_c3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔍 Fetch Chain", key="_sb_fetch"):
                    if _sb_ul:
                        try:
                            _exps = list(yf.Ticker(_sb_ul).options)
                            if _exps:
                                st.session_state['_sb_exps']      = _exps
                                st.session_state['_sb_ul_loaded'] = _sb_ul
                                for _k in ['_sb_near_chain','_sb_far_chain',
                                           '_sb_calls_chain','_sb_puts_chain']:
                                    st.session_state.pop(_k, None)
                            else:
                                st.error(f"No options for {_sb_ul}")
                        except Exception as _e:
                            st.error(str(_e))

            _ti   = _STRAT_TEMPLATES[_sb_tmpl]
            _is_cal = _ti['calendar']

            if st.session_state.get('_sb_exps'):
                _exps = st.session_state['_sb_exps']
                st.markdown("---")

                # Expiry selection
                if _is_cal:
                    _ec1, _ec2, _ec3 = st.columns([2, 2, 1])
                    with _ec1:
                        _sb_near = st.selectbox(
                            "Near Expiry (short leg)", _exps, index=0, key="_sb_near_exp"
                        )
                    with _ec2:
                        _far_def = min(3, len(_exps) - 1)
                        _sb_far  = st.selectbox(
                            "Far Expiry (long leg)", _exps, index=_far_def, key="_sb_far_exp"
                        )
                    with _ec3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("📋 Load Chains", key="_sb_load_cal"):
                            try:
                                _ul2     = st.session_state['_sb_ul_loaded']
                                _ot      = _ti['legs'][0]['option_type']
                                _stk_obj = yf.Ticker(_ul2)
                                _nc      = _stk_obj.option_chain(_sb_near)
                                _fc      = _stk_obj.option_chain(_sb_far)
                                _nd      = _nc.calls if _ot == 'call' else _nc.puts
                                _fd      = _fc.calls if _ot == 'call' else _fc.puts
                                st.session_state['_sb_near_chain']       = _nd.reset_index(drop=True).to_dict(orient='records')
                                st.session_state['_sb_far_chain']        = _fd.reset_index(drop=True).to_dict(orient='records')
                                st.session_state['_sb_near_exp_loaded']  = _sb_near
                                st.session_state['_sb_far_exp_loaded']   = _sb_far
                            except Exception as _e:
                                st.error(str(_e))
                else:
                    _ec1, _ec2 = st.columns([3, 1])
                    with _ec1:
                        _sb_single_exp = st.selectbox(
                            "Expiration (all legs)", _exps,
                            index=min(2, len(_exps) - 1), key="_sb_single_exp"
                        )
                    with _ec2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("📋 Load Strikes", key="_sb_load_std"):
                            try:
                                _full = yf.Ticker(st.session_state['_sb_ul_loaded']).option_chain(_sb_single_exp)
                                st.session_state['_sb_calls_chain']       = _full.calls.reset_index(drop=True).to_dict(orient='records')
                                st.session_state['_sb_puts_chain']        = _full.puts.reset_index(drop=True).to_dict(orient='records')
                                st.session_state['_sb_single_exp_loaded'] = _sb_single_exp
                            except Exception as _e:
                                st.error(str(_e))

                # Check if chains are ready
                _chains_ready = (
                    (st.session_state.get('_sb_near_chain') and st.session_state.get('_sb_far_chain'))
                    if _is_cal else
                    bool(st.session_state.get('_sb_calls_chain'))
                )

                # Custom leg management
                if _sb_tmpl == 'Custom' and _chains_ready:
                    if '_sb_custom_legs' not in st.session_state:
                        st.session_state['_sb_custom_legs'] = []
                    if st.button("+ Add Leg", key="_sb_add_cl"):
                        st.session_state['_sb_custom_legs'].append({})
                        st.rerun()

                if _chains_ready:
                    st.markdown("---")
                    st.markdown("**Configure legs** — strike · contracts · your cost basis:")

                    _ul_px   = _ul_prices.get(st.session_state.get('_sb_ul_loaded', ''), 0)
                    _leg_cfg = []
                    _tmpl_legs = _ti['legs'] if _sb_tmpl != 'Custom' else \
                                 [{'option_type': 'call', 'position': 'long', 'offset': 0.0,
                                   'label': f'Leg {i+1}', 'far': False}
                                  for i in range(len(st.session_state.get('_sb_custom_legs', [])))]

                    for _li, _lt in enumerate(_tmpl_legs):
                        _ot2   = _lt['option_type']
                        _pos2  = _lt['position'] if _sb_tmpl != 'Custom' else \
                                 st.session_state.get(f'_sb_cl_pos_{_li}', 'long')
                        _lbl2  = _lt['label']
                        _is_far_leg = _lt.get('far', False)

                        # Pick the right chain
                        if _is_cal:
                            _cr = st.session_state['_sb_far_chain'] if _is_far_leg \
                                  else st.session_state['_sb_near_chain']
                            _exp_leg = st.session_state['_sb_far_exp_loaded'] if _is_far_leg \
                                       else st.session_state['_sb_near_exp_loaded']
                        else:
                            _cr = st.session_state.get('_sb_calls_chain' if _ot2 == 'call'
                                                        else '_sb_puts_chain', [])
                            _exp_leg = st.session_state.get('_sb_single_exp_loaded', '')

                        _cdf2 = pd.DataFrame(_cr)
                        if _cdf2.empty:
                            continue

                        _icon  = '▲' if _pos2 == 'long' else '▼'
                        _iclr  = '#00d4aa' if _pos2 == 'long' else '#FF6B6B'
                        st.markdown(
                            f"<small><span style='color:{_iclr};font-weight:bold;'>"
                            f"{_icon} {_pos2.upper()} {_ot2.upper()}</span> — "
                            f"{_lbl2}  |  exp {_exp_leg}</small>",
                            unsafe_allow_html=True
                        )

                        # Suggest nearest strike to offset
                        _target = _ul_px * (1 + _lt.get('offset', 0.0))
                        _cdf2   = _cdf2.copy()
                        _cdf2['_d2'] = abs(_cdf2['strike'] - _target)
                        _def_idx = int(_cdf2['_d2'].idxmin())

                        def _slbl(i, df=_cdf2):
                            r  = df.iloc[i]
                            b  = float(r.get('bid', 0) or 0)
                            a  = float(r.get('ask', 0) or 0)
                            m  = (b + a) / 2 if b > 0 and a > 0 else float(r.get('lastPrice', 0) or 0)
                            iv = float(r.get('impliedVolatility', 0) or 0) * 100
                            return f"${r['strike']:.2f}  mid ${m:.2f}  IV {iv:.0f}%"

                        _lc1, _lc2, _lc3 = st.columns([3, 1, 1])
                        with _lc1:
                            _sel_si = st.selectbox(
                                "Strike", range(len(_cdf2)), index=_def_idx,
                                format_func=_slbl, key=f"_sb_str_{_li}",
                                label_visibility='collapsed'
                            )
                        _sel_sr = _cdf2.iloc[_sel_si]
                        _sb2    = float(_sel_sr.get('bid', 0) or 0)
                        _sa2    = float(_sel_sr.get('ask', 0) or 0)
                        _sm2    = (_sb2 + _sa2) / 2 if _sb2 > 0 and _sa2 > 0 \
                                  else float(_sel_sr.get('lastPrice', 0) or 0)
                        _siv2   = float(_sel_sr.get('impliedVolatility', 0.3) or 0.3)

                        with _lc2:
                            _leg_contracts = st.number_input(
                                "Qty", min_value=1, value=1, step=1,
                                key=f"_sb_qty_{_li}", label_visibility='collapsed'
                            )
                        with _lc3:
                            _leg_cost = st.number_input(
                                "Cost/sh", min_value=0.0, value=round(_sm2, 2), step=0.01,
                                key=f"_sb_cost_{_li}", label_visibility='collapsed'
                            )

                        # Custom: also let user pick option type + position
                        if _sb_tmpl == 'Custom':
                            _cc1, _cc2 = st.columns(2)
                            with _cc1:
                                _ot2 = st.selectbox(
                                    "Type", ['call', 'put'],
                                    key=f"_sb_cl_type_{_li}", label_visibility='collapsed'
                                )
                            with _cc2:
                                _pos2 = st.selectbox(
                                    "Position", ['long', 'short'],
                                    key=f"_sb_cl_pos_{_li}", label_visibility='collapsed'
                                )
                            if st.button(f"🗑️ Remove leg {_li+1}", key=f"_sb_rm_cl_{_li}"):
                                st.session_state['_sb_custom_legs'].pop(_li)
                                st.rerun()

                        _leg_cfg.append({
                            'option_type': _ot2,
                            'position':    _pos2,
                            'strike':      float(_sel_sr['strike']),
                            'expiration':  _exp_leg,
                            'contracts':   int(st.session_state.get(f"_sb_qty_{_li}", 1)),
                            'cost_basis':  float(st.session_state.get(f"_sb_cost_{_li}", _sm2)),
                            'iv_at_entry': _siv2,
                            'label':       _lbl2,
                        })

                    # Net debit/credit preview
                    if _leg_cfg:
                        _net = sum(
                            l['cost_basis'] * l['contracts'] * 100 *
                            (1 if l['position'] == 'long' else -1)
                            for l in _leg_cfg
                        )
                        _net_lbl = f"Net Credit received: ${abs(_net):,.2f}" \
                                   if _net < 0 else f"Net Debit paid: ${_net:,.2f}"
                        st.info(f"📊 {_net_lbl}")

                    # Name + submit
                    _auto_nm = f"{_sb_tmpl} — {st.session_state.get('_sb_ul_loaded','')}"
                    _strat_nm = st.text_input("Strategy Name", value=_auto_nm, key="_sb_name")

                    _sc1, _sc2 = st.columns(2)
                    with _sc1:
                        if st.button("➕ Add Strategy", key="_sb_submit", type="primary"):
                            if _leg_cfg:
                                st.session_state['options_positions'].append({
                                    'type':         'strategy',
                                    'name':         _strat_nm,
                                    'underlying':   st.session_state.get('_sb_ul_loaded', ''),
                                    'status':       'open',
                                    'close_method': None,
                                    'close_price':  None,
                                    'close_date':   None,
                                    'legs':         _leg_cfg,
                                })
                                save_options_positions(st.session_state['options_positions'])
                                for _k in ['_sb_exps','_sb_ul_loaded','_sb_near_chain',
                                           '_sb_far_chain','_sb_calls_chain','_sb_puts_chain',
                                           '_sb_single_exp_loaded','_sb_near_exp_loaded',
                                           '_sb_far_exp_loaded','_sb_custom_legs']:
                                    st.session_state.pop(_k, None)
                                st.session_state['_sb_show'] = False
                                st.rerun()
                    with _sc2:
                        if st.button("Cancel", key="_sb_cancel"):
                            st.session_state['_sb_show'] = False
                            st.rerun()

    # ── Single-leg positions table ────────────────────────────────────────────
    if _singles:
        st.markdown("#### Single-Leg Positions")
        with st.spinner("Fetching live prices for single-leg positions…"):
            for _, _p in _singles:
                if _p['status'] == 'open':
                    _lp, _liv = _live_option(
                        _p['underlying'], _p['option_type'],
                        _p['strike'], _p['expiration'], _p.get('iv_at_entry', 0.3)
                    )
                    _p['_live_price'] = _lp
                    _p['_live_iv']    = _liv

        _sl_rows     = []
        _sl_open_cnt = 0
        _sl_cls_cnt  = 0
        for _sidx, _p in _singles:
            _exp_dt   = datetime.strptime(_p['expiration'], '%Y-%m-%d')
            _dte      = (_exp_dt - datetime.now()).days
            _cost_tot = _p['cost_basis'] * _p['contracts'] * 100
            if _p['status'] == 'closed':
                _cm = _p.get('close_method', '')
                _cp = float(_p.get('close_price') or 0)
                _pnl = (-_cost_tot if _cm == 'expired_worthless'
                        else _cp * _p['contracts'] * 100 - _cost_tot)
                _pnl_pct = (_pnl / _cost_tot * 100) if _cost_tot else 0
                _delta = _theta = None
                _iv_d  = _p.get('iv_at_entry', 0.3)
                _sl_cls_cnt += 1
            else:
                _cp    = _p.get('_live_price')
                _iv_d  = _p.get('_live_iv', _p.get('iv_at_entry', 0.3))
                if _cp is not None:
                    _pnl     = _cp * _p['contracts'] * 100 - _cost_tot
                    _pnl_pct = (_pnl / _cost_tot * 100) if _cost_tot else 0
                    _total_opt_val += _cp * _p['contracts'] * 100
                    _total_opt_pnl += _pnl
                else:
                    _pnl = _pnl_pct = None
                _ul_p2 = _ul_prices.get(_p['underlying'], 0)
                _T2    = max(_dte / 365, 0.001)
                _delta = _theta = None
                if _ul_p2 > 0:
                    try:
                        _g2    = _oa.calculate_greeks(S=_ul_p2, K=_p['strike'], T=_T2,
                                                       sigma=_iv_d, option_type=_p['option_type'], r=0.045)
                        _delta = _g2.get('delta')
                        _theta = _g2.get('theta')
                    except Exception:
                        pass
                _sl_open_cnt += 1

            _sl_rows.append({
                '_idx':       _sidx,
                'Underlying': _p['underlying'],
                'Type':       _p['option_type'].upper(),
                'Strike':     f"${_p['strike']:.2f}",
                'Expiry':     _p['expiration'],
                'DTE':        str(_dte) if _p['status'] == 'open' else '—',
                'Contracts':  _p['contracts'],
                'Cost/sh':    f"${_p['cost_basis']:.2f}",
                'Current':    f"${_cp:.2f}" if _cp is not None else 'N/A',
                'P&L $':      f"${_pnl:,.2f}" if _pnl is not None else 'N/A',
                'P&L %':      f"{_pnl_pct:+.1f}%" if _pnl_pct is not None else 'N/A',
                'Delta':      f"{_delta:.3f}" if _delta is not None else '—',
                'Theta':      f"{_theta:.3f}" if _theta is not None else '—',
                'IV':         f"{_iv_d*100:.1f}%" if _iv_d else '—',
                'Status':     '✅ Open' if _p['status'] == 'open'
                              else f"⭕ {(_p.get('close_method') or 'closed').replace('_',' ').title()}",
            })

        st.dataframe(
            pd.DataFrame([{k: v for k, v in r.items() if k != '_idx'} for r in _sl_rows]),
            use_container_width=True, hide_index=True
        )

        # Manage single-leg positions
        _open_count  = _sl_open_cnt
        _closed_count = _sl_cls_cnt
        with st.expander("Manage Single-Leg Positions"):
            for _row in _sl_rows:
                _sidx2 = _row['_idx']
                _p2    = _opts[_sidx2]
                _lbl2  = (f"{_p2['option_type'].upper()} {_p2['underlying']} "
                          f"${_p2['strike']:.0f} exp {_p2['expiration']}")
                if _p2['status'] == 'open':
                    st.markdown(f"**✅ {_lbl2}**")
                    _clm = st.selectbox(
                        "Close method", ['sold','expired_worthless','exercised'],
                        format_func=lambda x: {'sold':'Sold','expired_worthless':'Expired worthless',
                                                'exercised':'Exercised'}[x],
                        key=f"sl_cm_{_sidx2}"
                    )
                    _clp = None
                    if _clm == 'sold':
                        _clp = st.number_input("Sale price ($/sh)", min_value=0.0, step=0.01,
                                               key=f"sl_cp_{_sidx2}")
                    elif _clm == 'exercised':
                        _ulpx3 = _ul_prices.get(_p2['underlying'], 0)
                        _intr2 = (max(_ulpx3 - _p2['strike'], 0) if _p2['option_type'] == 'call'
                                  else max(_p2['strike'] - _ulpx3, 0))
                        _clp = st.number_input("Net value at exercise ($/sh)", min_value=0.0,
                                               value=round(_intr2, 2), step=0.01, key=f"sl_ep_{_sidx2}")
                    if st.button("Confirm Close", key=f"sl_cls_{_sidx2}"):
                        st.session_state['options_positions'][_sidx2].update({
                            'status': 'closed', 'close_method': _clm,
                            'close_price': _clp, 'close_date': datetime.now().strftime('%Y-%m-%d'),
                        })
                        save_options_positions(st.session_state['options_positions'])
                        st.rerun()
                else:
                    _cm3  = _p2.get('close_method', '')
                    _cst3 = _p2['cost_basis'] * _p2['contracts'] * 100
                    _pr3  = float(_p2.get('close_price') or 0)
                    _pnl3 = (-_cst3 if _cm3 == 'expired_worthless'
                             else _pr3 * _p2['contracts'] * 100 - _cst3)
                    _clr3 = '#00d4aa' if _pnl3 >= 0 else '#FF6B6B'
                    st.markdown(
                        f"⭕ **{_lbl2}** — {_cm3.replace('_',' ').title()} | "
                        f"<span style='color:{_clr3};'>${_pnl3:,.2f}</span>",
                        unsafe_allow_html=True
                    )
                    if st.button("🗑️ Remove", key=f"sl_rm_{_sidx2}"):
                        st.session_state['options_positions'].pop(_sidx2)
                        save_options_positions(st.session_state['options_positions'])
                        st.rerun()
                st.divider()
    else:
        _open_count = _closed_count = 0

    # ── Strategies display ────────────────────────────────────────────────────
    if _strategies:
        st.markdown("#### Multi-Leg Strategies")

        with st.spinner("Fetching live prices for strategy legs…"):
            for _, _p in _strategies:
                if _p['status'] != 'open':
                    continue
                for _leg in _p.get('legs', []):
                    _lp2, _liv2 = _live_option(
                        _p['underlying'], _leg['option_type'],
                        _leg['strike'], _leg['expiration'], _leg.get('iv_at_entry', 0.3)
                    )
                    _leg['_live_price'] = _lp2
                    _leg['_live_iv']    = _liv2

        for _stidx, _p in _strategies:
            _legs   = _p.get('legs', [])
            _is_open = _p['status'] == 'open'

            # Compute combined metrics
            _net_entry = sum(
                l['cost_basis'] * l['contracts'] * 100 *
                (1 if l['position'] == 'long' else -1)
                for l in _legs
            )
            _net_delta = _net_theta = _net_vega = 0.0

            if _is_open:
                _net_cur = 0.0
                _all_ok  = True
                for _leg in _legs:
                    _lp3 = _leg.get('_live_price')
                    if _lp3 is None:
                        _all_ok = False
                        continue
                    _sign3 = 1 if _leg['position'] == 'long' else -1
                    _net_cur += _lp3 * _leg['contracts'] * 100 * _sign3
                    # Greeks
                    _exp_dt3 = datetime.strptime(_leg['expiration'], '%Y-%m-%d')
                    _dte3    = max((_exp_dt3 - datetime.now()).days, 0)
                    _T3      = max(_dte3 / 365, 0.001)
                    _ulpx4   = _ul_prices.get(_p['underlying'], 0)
                    _iv3     = _leg.get('_live_iv', _leg.get('iv_at_entry', 0.3))
                    if _ulpx4 > 0:
                        try:
                            _g3 = _oa.calculate_greeks(
                                S=_ulpx4, K=_leg['strike'], T=_T3,
                                sigma=_iv3, option_type=_leg['option_type'], r=0.045
                            )
                            _mult = _leg['contracts'] * _sign3
                            _net_delta += (_g3.get('delta') or 0) * _mult
                            _net_theta += (_g3.get('theta') or 0) * _mult
                            _net_vega  += (_g3.get('vega')  or 0) * _mult
                        except Exception:
                            pass
                _strat_pnl     = _net_cur - _net_entry
                _strat_pnl_pct = (_strat_pnl / abs(_net_entry) * 100) if _net_entry else 0
                if _all_ok:
                    _total_opt_val += _net_cur
                    _total_opt_pnl += _strat_pnl
                _pnl_str = f"P&L ${_strat_pnl:+,.2f} ({_strat_pnl_pct:+.1f}%)"
                _entry_str = (f"Credit ${abs(_net_entry):,.2f}" if _net_entry < 0
                              else f"Debit ${_net_entry:,.2f}")
                _stat_icon = "✅"
            else:
                _cm4 = _p.get('close_method', '')
                _cp4 = float(_p.get('close_price') or 0)
                _pnl3b = (_cp4 - _net_entry) if _cm4 != 'expired_worthless' else -abs(_net_entry)
                _pnl_str   = f"Realised P&L ${_pnl3b:+,.2f}"
                _entry_str = f"Entry {('Credit' if _net_entry < 0 else 'Debit')} ${abs(_net_entry):,.2f}"
                _stat_icon = "⭕"
                _open_count  -= 0   # already counted above for singles; don't double

            _strat_header = (
                f"{_stat_icon} **{_p['name']}** — {len(_legs)} legs | "
                f"{_entry_str} | {_pnl_str}"
            )
            if _is_open:
                _strat_header += (
                    f" | Δ {_net_delta:+.3f} | Θ {_net_theta:+.3f}"
                )

            with st.expander(_strat_header, expanded=False):
                # Leg details table
                _leg_rows = []
                for _leg in _legs:
                    _sign4 = 1 if _leg['position'] == 'long' else -1
                    _lp4   = _leg.get('_live_price')
                    _lpnl  = ((_lp4 - _leg['cost_basis']) * _leg['contracts'] * 100 * _sign4
                              if _lp4 is not None else None)
                    _leg_rows.append({
                        'Label':     _leg.get('label', ''),
                        'Type':      _leg['option_type'].upper(),
                        'Position':  _leg['position'].upper(),
                        'Strike':    f"${_leg['strike']:.2f}",
                        'Expiry':    _leg['expiration'],
                        'Contracts': _leg['contracts'],
                        'Cost/sh':   f"${_leg['cost_basis']:.2f}",
                        'Current':   f"${_lp4:.2f}" if _lp4 is not None else 'N/A',
                        'Leg P&L':   f"${_lpnl:,.2f}" if _lpnl is not None else 'N/A',
                        'IV':        f"{_leg.get('_live_iv', _leg.get('iv_at_entry',0))*100:.1f}%",
                    })
                st.dataframe(pd.DataFrame(_leg_rows), use_container_width=True, hide_index=True)

                # Close / Remove controls
                if _is_open:
                    st.markdown("**Close entire strategy:**")
                    _scm = st.selectbox(
                        "Close method",
                        ['bought_to_close', 'expired_worthless', 'partial_close'],
                        format_func=lambda x: {
                            'bought_to_close':  'Bought to close (enter net credit/debit received)',
                            'expired_worthless': 'All legs expired worthless',
                            'partial_close':    'Partial / custom close (enter net P&L)',
                        }[x],
                        key=f"strat_cm_{_stidx}"
                    )
                    _scp = None
                    if _scm in ('bought_to_close', 'partial_close'):
                        _scp = st.number_input(
                            "Net amount received to close (negative = paid to close, $/total)",
                            step=1.0, key=f"strat_cp_{_stidx}"
                        )
                    if st.button("Confirm Strategy Close", key=f"strat_cls_{_stidx}"):
                        st.session_state['options_positions'][_stidx].update({
                            'status':       'closed',
                            'close_method': _scm,
                            'close_price':  _scp,
                            'close_date':   datetime.now().strftime('%Y-%m-%d'),
                        })
                        save_options_positions(st.session_state['options_positions'])
                        st.rerun()
                else:
                    if st.button("🗑️ Remove Strategy", key=f"strat_rm_{_stidx}"):
                        st.session_state['options_positions'].pop(_stidx)
                        save_options_positions(st.session_state['options_positions'])
                        st.rerun()

    if not _singles and not _strategies:
        st.info("No options positions yet. Use the sidebar **(single legs)** or **🔀 Add Strategy** button above.")

    # ── Options summary metrics ───────────────────────────────────────────────
    if _singles or _strategies:
        _s_open  = sum(1 for _, p in _singles    if p['status'] == 'open')
        _s_cls   = sum(1 for _, p in _singles    if p['status'] == 'closed')
        _st_open = sum(1 for _, p in _strategies if p['status'] == 'open')
        _st_cls  = sum(1 for _, p in _strategies if p['status'] == 'closed')
        _ocm1, _ocm2, _ocm3, _ocm4 = st.columns(4)
        with _ocm1:
            st.metric("Open Options Value", f"${_total_opt_val:,.2f}")
        with _ocm2:
            st.metric("Unrealised P&L", f"${_total_opt_pnl:,.2f}")
        with _ocm3:
            st.metric("Open Positions", f"{_s_open} single · {_st_open} strategies")
        with _ocm4:
            st.metric("Closed", f"{_s_cls} single · {_st_cls} strategies")

        if (_s_cls + _st_cls) > 0:
            if st.button("🗑️ Clear All Closed Options"):
                st.session_state['options_positions'] = [
                    p for p in _opts if p['status'] == 'open'
                ]
                save_options_positions(st.session_state['options_positions'])
                st.rerun()

    # ── COMBINED PORTFOLIO TOTALS ────────────────────────────────────────────
    if _singles or _strategies:
        st.subheader("🎯 Combined Portfolio")
        _stk_val  = summary['total_value']
        _stk_pnl  = summary['total_pnl']
        _comb_val = _stk_val + _total_opt_val
        _comb_pnl = _stk_pnl + _total_opt_pnl
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            st.markdown("**📈 Stock Portfolio**")
            st.metric("Value", f"${_stk_val:,.2f}")
            st.metric("P&L",   f"${_stk_pnl:,.2f}", f"{summary['total_pnl_pct']:+.2f}%")
        with _c2:
            st.markdown("**📊 Options Portfolio**")
            st.metric("Value (Open)",   f"${_total_opt_val:,.2f}")
            st.metric("Unrealised P&L", f"${_total_opt_pnl:,.2f}")
        with _c3:
            st.markdown("**🎯 Combined**")
            st.metric("Total Value", f"${_comb_val:,.2f}")
            st.metric("Total P&L",   f"${_comb_pnl:,.2f}")

    # Correlation Matrix & Diversification Score
    st.subheader("🔗 Correlation & Diversification")

    if len(tickers) >= 2:
        with st.spinner("Calculating 60-day correlations..."):
            try:
                import yfinance as yf
                corr_start = datetime.now() - timedelta(days=90)
                raw_corr = loader.fetch_historical_data(tickers, start_date=corr_start)
                prices_corr = pd.DataFrame({
                    t: raw_corr[t]['Close']
                    for t in tickers
                    if raw_corr.get(t) is not None and not raw_corr[t].empty
                })

                if prices_corr.shape[1] >= 2:
                    rets_corr = prices_corr.pct_change().dropna()
                    corr_mtx  = rets_corr.corr()
                    n         = len(corr_mtx)

                    off_diag = [
                        abs(corr_mtx.iloc[i, j])
                        for i in range(n) for j in range(i + 1, n)
                    ]
                    avg_corr       = sum(off_diag) / len(off_diag) if off_diag else 0
                    diversif_score = (1 - avg_corr) * 100
                    score_color    = "#00d4aa" if diversif_score >= 60 else "#FFB366" if diversif_score >= 40 else "#FF6B6B"
                    score_label    = "Good" if diversif_score >= 60 else "Moderate" if diversif_score >= 40 else "Poor"

                    high_corr_pairs = [
                        (corr_mtx.index[i], corr_mtx.columns[j], corr_mtx.iloc[i, j])
                        for i in range(n) for j in range(i + 1, n)
                        if abs(corr_mtx.iloc[i, j]) >= 0.85
                    ]

                    col_div, col_heat = st.columns([1, 3])

                    with col_div:
                        st.markdown(
                            f"<div style='text-align:center; padding:20px;'>"
                            f"<p style='margin:0; font-size:13px; color:#aaa;'>Diversification Score</p>"
                            f"<p style='margin:0; font-size:56px; font-weight:bold; color:{score_color};'>{diversif_score:.0f}</p>"
                            f"<p style='margin:0; font-size:13px; color:{score_color};'>{score_label}</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        st.caption(f"Avg pairwise correlation: {avg_corr:.2f}")
                        if high_corr_pairs:
                            st.markdown("**Highly correlated pairs (≥0.85):**")
                            for t1, t2, val in high_corr_pairs:
                                st.warning(f"{t1} ↔ {t2}: {val:.2f}", icon="⚠️")
                        else:
                            st.success("No pairs with correlation ≥ 0.85", icon="✅")

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
                        fig_corr.update_layout(
                            title="60-Day Return Correlations",
                            height=380,
                            **dark_plotly_layout()
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Could not fetch sufficient data for correlation analysis.")
            except Exception as e:
                st.warning(f"Correlation calculation failed: {e}")
    else:
        st.info("Add at least 2 positions to see correlation analysis.")

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
