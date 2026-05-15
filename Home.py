"""
Portfolio Intelligence — Command Center
Merges: Position Tracker · Action Dashboard · Trade Suggestions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent))

from data.portfolio_loader import PortfolioLoader
from data.market_data import MarketDataLoader
from data.options_data import OptionsDataLoader
from calculations.performance import PerformanceAnalytics
from calculations.options_analytics import OptionsAnalytics
from calculations.regime_detector import RegimeDetector
from calculations.scoring_engine import ScoringEngine
from calculations.optimizer import RegimeAwareOptimizer
from calculations.strategy_builder import StrategyBuilder
from calculations.options_recommender import OptionsRecommender
from calculations.probability_utils import probability_of_profit, expected_value
from utils.carbon_theme import (
    apply_carbon_theme, carbon_plotly_layout, top_nav, metric_card, flex_table,
    GAIN, LOSS, ACCENT, BG, CARD, CARD2, BORDER, BORDER2, FG, DIM, SUBTLE, AMBER, GREEN, PURPLE,
    regime_color, page_header, section_header, _SANS, _SERIF,
    regime_alert_banner, position_alert_banners, macro_event_banner,
)
from utils.portfolio_store import (
    save_portfolio, load_portfolio, portfolio_file_exists, get_last_saved_time,
    save_options_positions, load_options_positions,
)
from data.trade_journal import (
    log_trade, get_trades, delete_trade,
    save_daily_snapshot, get_portfolio_history,
    compute_equity_curve, compute_trade_pnl,
)

# ── Multi-leg strategy templates ──────────────────────────────────────────────
_STRAT_TEMPLATES = {
    'Iron Condor':        {'desc': 'Sell OTM put spread + OTM call spread (net credit)', 'calendar': False,
                           'legs': [{'option_type':'put','position':'short','offset':-0.05,'label':'Short OTM Put'},
                                    {'option_type':'put','position':'long','offset':-0.10,'label':'Long Far-OTM Put'},
                                    {'option_type':'call','position':'short','offset':0.05,'label':'Short OTM Call'},
                                    {'option_type':'call','position':'long','offset':0.10,'label':'Long Far-OTM Call'}]},
    'Bull Call Spread':   {'desc': 'Buy ATM call, sell OTM call (net debit)', 'calendar': False,
                           'legs': [{'option_type':'call','position':'long','offset':0.00,'label':'Long ATM Call'},
                                    {'option_type':'call','position':'short','offset':0.05,'label':'Short OTM Call'}]},
    'Bear Call Spread':   {'desc': 'Sell OTM call, buy further OTM call (net credit)', 'calendar': False,
                           'legs': [{'option_type':'call','position':'short','offset':0.05,'label':'Short OTM Call'},
                                    {'option_type':'call','position':'long','offset':0.10,'label':'Long Far-OTM Call'}]},
    'Bull Put Spread':    {'desc': 'Sell OTM put, buy further OTM put (net credit)', 'calendar': False,
                           'legs': [{'option_type':'put','position':'short','offset':-0.05,'label':'Short OTM Put'},
                                    {'option_type':'put','position':'long','offset':-0.10,'label':'Long Far-OTM Put'}]},
    'Bear Put Spread':    {'desc': 'Buy ATM put, sell OTM put (net debit)', 'calendar': False,
                           'legs': [{'option_type':'put','position':'long','offset':0.00,'label':'Long ATM Put'},
                                    {'option_type':'put','position':'short','offset':-0.05,'label':'Short OTM Put'}]},
    'Long Straddle':      {'desc': 'Buy ATM call + put (net debit)', 'calendar': False,
                           'legs': [{'option_type':'call','position':'long','offset':0.00,'label':'Long ATM Call'},
                                    {'option_type':'put','position':'long','offset':0.00,'label':'Long ATM Put'}]},
    'Short Straddle':     {'desc': 'Sell ATM call + put (net credit)', 'calendar': False,
                           'legs': [{'option_type':'call','position':'short','offset':0.00,'label':'Short ATM Call'},
                                    {'option_type':'put','position':'short','offset':0.00,'label':'Short ATM Put'}]},
    'Long Strangle':      {'desc': 'Buy OTM call + put (net debit)', 'calendar': False,
                           'legs': [{'option_type':'call','position':'long','offset':0.05,'label':'Long OTM Call'},
                                    {'option_type':'put','position':'long','offset':-0.05,'label':'Long OTM Put'}]},
    'Short Strangle':     {'desc': 'Sell OTM call + put (net credit)', 'calendar': False,
                           'legs': [{'option_type':'call','position':'short','offset':0.05,'label':'Short OTM Call'},
                                    {'option_type':'put','position':'short','offset':-0.05,'label':'Short OTM Put'}]},
    'Calendar Call Spread':{'desc': 'Sell near-term call, buy far-term call (net debit)', 'calendar': True,
                            'legs': [{'option_type':'call','position':'short','offset':0.00,'label':'Short Near-Term Call','far':False},
                                     {'option_type':'call','position':'long','offset':0.00,'label':'Long Far-Term Call','far':True}]},
    'Calendar Put Spread': {'desc': 'Sell near-term put, buy far-term put (net debit)', 'calendar': True,
                            'legs': [{'option_type':'put','position':'short','offset':0.00,'label':'Short Near-Term Put','far':False},
                                     {'option_type':'put','position':'long','offset':0.00,'label':'Long Far-Term Put','far':True}]},
    'Custom':             {'desc': 'Build your own multi-leg strategy', 'calendar': False, 'legs': []},
}

_HOLDING_COLORS = ['#22d3ee','#67e8f9','#a5f3fc','#0891b2','#06b6d4',
                   '#38bdf8','#7dd3fc','#0e7490','#155e75','#cffafe']
_OPT_COLORS     = ['#f59e0b','#fbbf24','#fcd34d','#f97316','#fb923c',
                   '#fdba74','#d97706','#b45309','#a16207','#ca8a04']

# ── Trade signal constants ────────────────────────────────────────────────────
_STRATEGY_PROFILES = {
    'Long Call':        {'bias':'bullish',  'premium_type':'buy',  'risk_defined':True},
    'Long Put':         {'bias':'bearish',  'premium_type':'buy',  'risk_defined':True},
    'Bull Call Spread': {'bias':'bullish',  'premium_type':'buy',  'risk_defined':True},
    'Bear Put Spread':  {'bias':'bearish',  'premium_type':'buy',  'risk_defined':True},
    'Iron Condor':      {'bias':'neutral',  'premium_type':'sell', 'risk_defined':True},
    'Short Strangle':   {'bias':'neutral',  'premium_type':'sell', 'risk_defined':False},
    'Long Straddle':    {'bias':'volatile', 'premium_type':'buy',  'risk_defined':True},
    'Long Strangle':    {'bias':'volatile', 'premium_type':'buy',  'risk_defined':True},
    'Iron Butterfly':   {'bias':'neutral',  'premium_type':'sell', 'risk_defined':True},
    'Call Butterfly':   {'bias':'bullish',  'premium_type':'buy',  'risk_defined':True},
}
_REGIME_PREFERRED = {
    'Risk-On':         ['Bull Call Spread','Long Call','Call Butterfly','Iron Condor'],
    'Caution':         ['Bull Call Spread','Bear Put Spread','Iron Condor','Short Strangle'],
    'High Volatility': ['Long Straddle','Long Strangle','Bull Call Spread','Bear Put Spread'],
    'Stagflation':     ['Bear Put Spread','Long Put','Iron Condor','Short Strangle'],
    'Recession':       ['Bear Put Spread','Long Put','Long Straddle','Iron Condor'],
    'Mean Reversion':  ['Short Strangle','Iron Butterfly','Iron Condor'],
    'Uncertain':       ['Iron Condor','Short Strangle','Bull Call Spread'],
    # legacy
    'Low Vol':         ['Iron Condor','Iron Butterfly','Call Butterfly','Bull Call Spread'],
    'High Vol':        ['Long Straddle','Long Strangle','Bull Call Spread','Bear Put Spread'],
    'Trending':        ['Bull Call Spread','Bear Put Spread','Long Call','Long Put','Call Butterfly'],
    'Unknown':         ['Bull Call Spread','Iron Condor'],
}
_RF = 0.045

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Command Center", page_icon="◈", layout="wide")
apply_carbon_theme()
top_nav("Command Center")

# ── Object initialisation ─────────────────────────────────────────────────────
loader          = PortfolioLoader()
analytics       = PerformanceAnalytics()
market_loader   = MarketDataLoader()
options_loader  = OptionsDataLoader()
detector        = RegimeDetector()
scorer          = ScoringEngine()
optimizer       = RegimeAwareOptimizer()
recommender     = OptionsRecommender()
_oa             = OptionsAnalytics()

# ── Cached market context ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _get_market_context():
    spy_data  = market_loader.load_index_data('SPY', '2y')
    vix_data  = market_loader.load_vix_data('2y')
    spy_px, vix_px = market_loader.align_data(spy_data, vix_data)
    regime, signals = detector.classify_regime(spy_px, vix_px)
    current_regime = regime.iloc[-1]
    current_vix    = float(vix_px.iloc[-1])
    market_stats   = market_loader.get_current_market_stats(spy_data, current_vix)
    regime_info    = detector.get_regime_description(current_regime)
    return current_regime, regime_info, market_stats

@st.cache_data(ttl=1800, show_spinner=False)
def _compute_iv_rank(ticker: str, current_iv: float) -> float:
    try:
        from data.alpaca_client import get_bars as _alpaca_bars
        hist = _alpaca_bars(ticker, period='1y')
        if hist.empty or len(hist) < 30:
            return 0.5
        lr   = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        hv   = lr.rolling(20).std().dropna() * np.sqrt(252)
        if len(hv) < 2 or hv.max() <= hv.min():
            return 0.5
        return float(np.clip((current_iv - hv.min()) / (hv.max() - hv.min()), 0, 1))
    except Exception:
        return 0.5

def _score_strategy(name, view, iv_rank, regime, risk_tol):
    if name not in _STRATEGY_PROFILES:
        return 0.0, []
    p       = _STRATEGY_PROFILES[name]
    score   = 0.0
    reasons = []
    bias_map = {'bullish':['Bullish'],'bearish':['Bearish'],'neutral':['Neutral'],
                'volatile':['Volatile / Expecting Big Move']}
    if view in bias_map.get(p['bias'], []):
        score += 40
        reasons.append(f"Direction aligned: {p['bias']} matches {view.lower()}")
    is_sell = p['premium_type'] == 'sell'
    if iv_rank >= 0.70 and is_sell:
        score += 25; reasons.append(f"IV rank {iv_rank:.0%} — selling premium advantageous")
    elif iv_rank <= 0.35 and not is_sell:
        score += 25; reasons.append(f"IV rank {iv_rank:.0%} — options relatively cheap")
    elif 0.35 < iv_rank < 0.70:
        score += 12; reasons.append(f"IV rank {iv_rank:.0%} moderate")
    if name in _REGIME_PREFERRED.get(regime, []):
        score += 20; reasons.append(f"Preferred strategy in '{regime}' regime")
    conservatism = 4 - risk_tol
    if p['risk_defined']:
        score += 15 * (conservatism / 3) if risk_tol < 3 else 15
        if conservatism >= 2:
            reasons.append("Risk-defined matches conservative tolerance")
    else:
        if risk_tol < 2:
            score -= 25; reasons.append("Undefined risk penalised for conservative tolerance")
    return float(np.clip(score, 0, 100)), reasons

def _live_option(underlying, option_type, strike, expiration, iv_fallback=0.3):
    try:
        df  = (yf.Ticker(underlying).option_chain(expiration).calls
               if option_type == 'call'
               else yf.Ticker(underlying).option_chain(expiration).puts).copy()
        df['_d'] = abs(df['strike'] - strike)
        r  = df.nsmallest(1,'_d').iloc[0]
        b  = float(r.get('bid',0) or 0); a = float(r.get('ask',0) or 0)
        m  = (b+a)/2 if b>0 and a>0 else float(r.get('lastPrice',0) or 0)
        iv = float(r.get('impliedVolatility', iv_fallback) or iv_fallback)
        return (m if m > 0 else None), iv
    except Exception:
        return None, iv_fallback

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO INPUT PANEL (inline — no sidebar)
# ══════════════════════════════════════════════════════════════════════════════

# Auto-load saved portfolio on first visit
if 'options_positions' not in st.session_state:
    st.session_state['options_positions'] = []
if '_opts_loaded' not in st.session_state:
    _so = load_options_positions()
    if _so:
        st.session_state['options_positions'] = _so
    st.session_state['_opts_loaded'] = True

# Auto-restore equity positions from disk (so other pages see them immediately)
if st.session_state.get('positions') is None:
    _saved = load_portfolio()
    if _saved is not None and not _saved.empty:
        st.session_state['positions'] = _saved

positions_df = None

# Determine label for the expander
_port_loaded = 'positions' in st.session_state and st.session_state['positions'] is not None
_port_label = (
    f"◈ Portfolio — {len(st.session_state['positions'])} positions loaded  (click to update)"
    if _port_loaded else "◈ Load Portfolio  ▸ click to expand"
)

with st.expander(_port_label, expanded=not _port_loaded):
    _pi_c1, _pi_c2 = st.columns([2, 1])

    with _pi_c1:
        input_method = st.radio(
            "Load positions via:", ["Manual Entry", "Use Sample Portfolio", "Upload CSV"],
            horizontal=True, key="input_method_radio",
        )

    with _pi_c2:
        _cash = st.number_input(
            "Cash Balance ($)", min_value=0.0, step=100.0, format="%.2f",
            value=float(st.session_state.get('cash_balance', 0.0)), key="cash_balance_input",
        )
        st.session_state['cash_balance'] = _cash

        if portfolio_file_exists():
            _last = get_last_saved_time()
            if _last:
                try:
                    st.caption(f"Last saved: {datetime.fromisoformat(_last).strftime('%Y-%m-%d %H:%M')}")
                except Exception:
                    pass
            if st.button("Load Saved Portfolio", key="load_saved_btn"):
                loaded = load_portfolio()
                if loaded is not None:
                    st.session_state['positions'] = loaded
                    st.session_state['manual_positions'] = loaded.to_dict(orient='records')
                    _ro = load_options_positions()
                    if _ro:
                        st.session_state['options_positions'] = _ro
                    st.success(f"Loaded {len(loaded)} positions!")
                    st.rerun()
                else:
                    st.error("Could not load saved portfolio.")

    st.divider()

    if input_method == "Manual Entry":
        if 'manual_positions' not in st.session_state:
            st.session_state['manual_positions'] = []

        with st.form("add_position_form", clear_on_submit=True):
            _f1, _f2, _f3, _f4, _f5 = st.columns([1.5, 1, 1, 1, 0.8])
            with _f1:
                ticker_in = st.text_input("Ticker", placeholder="AAPL").upper()
            with _f2:
                shares_in = st.number_input("Shares", min_value=0.0, step=1.0, format="%.2f")
            with _f3:
                cost_in = st.number_input("Cost Basis ($)", min_value=0.0, step=0.01, format="%.2f")
            with _f4:
                date_in = st.date_input("Purchase Date", value=datetime.now())
            with _f5:
                st.write("")
                st.write("")
                _submitted = st.form_submit_button("Add Position", use_container_width=True)

            if _submitted:
                if ticker_in and shares_in > 0 and cost_in > 0:
                    st.session_state['manual_positions'].append({
                        'ticker': ticker_in, 'shares': shares_in,
                        'cost_basis': cost_in, 'purchase_date': date_in.strftime('%Y-%m-%d'),
                    })
                    st.success(f"Added {shares_in:.0f} × {ticker_in}")
                else:
                    st.error("Fill all required fields.")

        if st.session_state.get('manual_positions'):
            _mp_cols = st.columns([3, 1])
            with _mp_cols[0]:
                for idx, pos in enumerate(st.session_state['manual_positions']):
                    _mc1, _mc2 = st.columns([5, 1])
                    _mc1.markdown(
                        f'<span style="color:{ACCENT};font-weight:500;">{pos["ticker"]}</span>'
                        f'<span style="color:{DIM};font-size:0.82rem;"> — {pos["shares"]} shares @ ${pos["cost_basis"]:.2f}</span>',
                        unsafe_allow_html=True,
                    )
                    if _mc2.button("✕", key=f"del_{idx}"):
                        st.session_state['manual_positions'].pop(idx)
                        st.rerun()
            with _mp_cols[1]:
                if st.button("Clear All", key="clear_all_btn"):
                    st.session_state['manual_positions'] = []
                    st.session_state.pop('positions', None)
                    st.rerun()

            positions_df = pd.DataFrame(st.session_state['manual_positions'])
            st.session_state['positions'] = positions_df
        else:
            st.info("No positions yet. Add one above.")

    elif input_method == "Use Sample Portfolio":
        if st.button("Load Sample Portfolio", type="primary"):
            positions_df = loader.create_sample_portfolio()
            st.session_state['positions'] = positions_df
            st.session_state['manual_positions'] = []
            st.success("Sample portfolio loaded!")
            st.rerun()

    elif input_method == "Upload CSV":
        st.caption("**Required columns:** `ticker, shares, cost_basis, purchase_date`")
        uploaded = st.file_uploader("Upload CSV", type=['csv'], key="csv_uploader")
        if uploaded:
            try:
                positions_df = loader.load_from_csv(uploaded)
                st.session_state['positions'] = positions_df
                st.session_state['manual_positions'] = []
                st.success(f"Loaded {len(positions_df)} positions!")
                st.rerun()
            except Exception as e:
                st.error(str(e))

# ── Inline options entry (separate expander) ──────────────────────────────────
with st.expander("Add Options Position", expanded=False):
    _oe1, _oe2, _oe3 = st.columns([2, 1, 1])
    with _oe1:
        _add_ul = st.text_input("Underlying Ticker", key="_add_opt_ul", placeholder="e.g. AAPL").upper().strip()
    with _oe2:
        st.write("")
        st.write("")
        _fetch_btn = st.button("Fetch Chain", key="_fetch_chain_btn")
    with _oe3:
        st.write("")

    if _fetch_btn:
        if _add_ul:
            try:
                _exps = list(yf.Ticker(_add_ul).options)
                if _exps:
                    st.session_state['_opt_exps'] = _exps
                    st.session_state['_opt_ul_loaded'] = _add_ul
                    for k in ['_opt_chain', '_opt_sel_exp', '_opt_sel_type']:
                        st.session_state.pop(k, None)
                else:
                    st.error(f"No options for {_add_ul}")
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("Enter a ticker first.")

    if st.session_state.get('_opt_exps'):
        _oc1, _oc2, _oc3 = st.columns([2, 1, 1])
        with _oc1:
            _sel_exp = st.selectbox("Expiration", st.session_state['_opt_exps'], key="_sel_opt_exp")
        with _oc2:
            _sel_type = st.radio("Type", ["call", "put"], horizontal=True, key="_sel_opt_type")
        with _oc3:
            st.write("")
            st.write("")
            _load_strikes = st.button("Load Strikes", key="_load_strikes_btn")

        if _load_strikes:
            try:
                _ch = yf.Ticker(st.session_state['_opt_ul_loaded']).option_chain(_sel_exp)
                _df = (_ch.calls if _sel_type == 'call' else _ch.puts).reset_index(drop=True)
                if not _df.empty:
                    st.session_state['_opt_chain']    = _df.to_dict(orient='records')
                    st.session_state['_opt_sel_exp']  = _sel_exp
                    st.session_state['_opt_sel_type'] = _sel_type
                else:
                    st.error("No contracts found.")
            except Exception as e:
                st.error(str(e))

    if st.session_state.get('_opt_chain'):
        _cdf = pd.DataFrame(st.session_state['_opt_chain'])
        def _slbl(i):
            r  = _cdf.iloc[i]
            b  = float(r.get('bid', 0) or 0); a = float(r.get('ask', 0) or 0)
            m  = (b+a)/2 if b>0 and a>0 else float(r.get('lastPrice', 0) or 0)
            iv = float(r.get('impliedVolatility', 0) or 0) * 100
            return f"${r['strike']:.2f}  mid ${m:.2f}  IV {iv:.0f}%"

        _sel_idx = st.selectbox("Strike", range(len(_cdf)), format_func=_slbl, key="_sel_strike_idx")
        _sel_row = _cdf.iloc[_sel_idx]
        _b2  = float(_sel_row.get('bid', 0) or 0); _a2 = float(_sel_row.get('ask', 0) or 0)
        _mid = (_b2+_a2)/2 if _b2>0 and _a2>0 else float(_sel_row.get('lastPrice', 0) or 0)
        _iv2 = float(_sel_row.get('impliedVolatility', 0.3) or 0.3)
        st.info(f"Mid **${_mid:.2f}** · IV {_iv2*100:.0f}% · OI {int(_sel_row.get('openInterest', 0) or 0):,}")

        with st.form("add_option_form", clear_on_submit=True):
            _of1, _of2, _of3 = st.columns([1, 1, 1])
            with _of1:
                _contracts = st.number_input("Contracts", min_value=1, value=1, step=1)
            with _of2:
                _cost_opt = st.number_input("Cost Basis ($/share)", min_value=0.0,
                                            value=round(_mid, 2), step=0.01)
            with _of3:
                st.write("")
                st.write("")
                _add_opt = st.form_submit_button("Add Options Position", use_container_width=True)
            if _add_opt:
                st.session_state['options_positions'].append({
                    'underlying':  st.session_state['_opt_ul_loaded'],
                    'option_type': st.session_state['_opt_sel_type'],
                    'strike':      float(_sel_row['strike']),
                    'expiration':  st.session_state['_opt_sel_exp'],
                    'contracts':   int(_contracts),
                    'cost_basis':  float(_cost_opt),
                    'iv_at_entry': _iv2, 'status': 'open',
                    'close_method': None, 'close_price': None, 'close_date': None,
                })
                save_options_positions(st.session_state['options_positions'])
                st.rerun()

        if st.button("Reset Search", key="_reset_chain_btn"):
            for k in ['_opt_exps', '_opt_ul_loaded', '_opt_chain', '_opt_sel_exp', '_opt_sel_type']:
                st.session_state.pop(k, None)
            st.rerun()

# Restore positions from session state if not set by input widget
if 'positions' in st.session_state:
    positions_df = st.session_state['positions']

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
page_header("Command Center", datetime.now().strftime('%A, %B %-d, %Y'))

tab_overview, tab_positions, tab_options, tab_signals, tab_journal = st.tabs([
    "Overview", "Positions", "Options", "Trade Signals", "Trade Journal"
])

# ── Shared data (loaded once, used across tabs) ───────────────────────────────
summary          = None
position_metrics = None
current_prices   = {}
tickers          = []

if positions_df is not None and not positions_df.empty:
    tickers = positions_df['ticker'].tolist()
    with st.spinner("Fetching market data…"):
        current_prices   = loader.fetch_current_prices(tickers)
    position_metrics = loader.calculate_position_metrics(positions_df, current_prices)
    summary          = loader.get_portfolio_summary(position_metrics)
    _cash_balance    = st.session_state.get('cash_balance', 0.0)
    st.session_state['current_prices'] = current_prices
    st.session_state['total_value']    = summary['total_value'] + _cash_balance

    # Day's P&L
    _day_pnl = _day_pnl_pct = 0.0
    try:
        from data.alpaca_client import get_close_prices as _alpaca_close
        _h2 = _alpaca_close(tickers, period="5d")
        if len(_h2) >= 2:
            _prev = _h2.iloc[-2]; _curr = _h2.iloc[-1]
            for _, _row in positions_df.iterrows():
                t = _row['ticker'].upper()
                if t in _curr.index and t in _prev.index:
                    _day_pnl += (_curr[t] - _prev[t]) * float(_row['shares'])
            _cb_total = (positions_df['shares'] * positions_df['cost_basis']).sum()
            _day_pnl_pct = _day_pnl / _cb_total * 100 if _cb_total else 0.0
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    if summary is None:
        st.info("Load a portfolio using the panel above to get started.")
        st.markdown("""
**Manual Entry** — add positions one by one · **Sample Portfolio** — instant demo · **Upload CSV** — `ticker, shares, cost_basis, purchase_date`
        """)
    else:
        _cash_balance = st.session_state.get('cash_balance', 0.0)

        # ── Market context ────────────────────────────────────────────────────
        try:
            current_regime, regime_info, market_stats = _get_market_context()
        except Exception:
            current_regime, regime_info, market_stats = "Unknown", {}, {}

        rc = regime_color(current_regime)

        # ── Alert banners ─────────────────────────────────────────────────────
        regime_alert_banner(current_regime)
        if current_prices:
            position_alert_banners(positions_df, current_prices)
        # Macro events from calendar_data if available
        try:
            from data.calendar_data import get_upcoming_events as _get_evts
            _raw_evts = _get_evts(days_ahead=7)
            _evts = [{"name": e["label"], "days_away": e["days"]} for e in _raw_evts]
            if _evts:
                macro_event_banner(_evts)
        except Exception:
            pass

        # ── Top KPI row ───────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Portfolio Value",
                  f"${summary['total_value'] + _cash_balance:,.0f}",
                  f"{summary['total_pnl_pct']:+.2f}% all-time")
        k2.metric("Cash",
                  f"${_cash_balance:,.0f}",
                  f"{_cash_balance/(summary['total_value']+_cash_balance)*100:.1f}% of portfolio"
                  if (summary['total_value']+_cash_balance)>0 else "—")
        k3.metric("Day's P&L",  f"${_day_pnl:,.0f}", f"{_day_pnl_pct:+.2f}%")
        k4.metric("Positions",  summary['num_positions'],
                  f"{summary['winners']}W / {summary['losers']}L")
        vix_now = market_stats.get('vix_current', 0)
        k5.metric("VIX", f"{vix_now:.1f}",
                  "Elevated" if vix_now > 25 else "Normal" if vix_now > 15 else "Low")

        # ── Regime status strip ───────────────────────────────────────────────
        _spy1d   = market_stats.get("returns_1d", 0)
        _spy_col = GAIN if _spy1d >= 0 else LOSS
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:1.75rem;'
            f'background:{CARD};border-radius:12px;'
            f'border-top:3px solid {rc};'
            f'padding:1rem 1.5rem;margin:0.75rem 0 1.25rem 0;'
            f'box-shadow:0 4px 20px rgba(0,0,0,0.4);">'
            f'<div style="flex-shrink:0;">'
            f'<div style="font-family:{_SANS};font-size:0.6rem;font-weight:600;'
            f'color:{DIM};text-transform:uppercase;letter-spacing:0.14em;">Market Regime</div>'
            f'<div style="font-family:{_SERIF};font-size:1.5rem;font-style:italic;'
            f'color:{rc};margin-top:4px;line-height:1;">{current_regime}</div>'
            f'</div>'
            f'<div style="width:1px;height:44px;background:linear-gradient(to bottom,'
            f'transparent,{BORDER},transparent);flex-shrink:0;"></div>'
            f'<div style="font-family:{_SANS};font-size:0.84rem;color:{DIM};flex:1;line-height:1.5;">'
            f'{regime_info.get("strategy","") if regime_info else ""}</div>'
            f'<div style="text-align:right;flex-shrink:0;">'
            f'<div style="font-family:{_SANS};font-size:0.68rem;color:{DIM};'
            f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">SPY 1D</div>'
            f'<div style="font-family:{_SERIF};font-size:1.1rem;font-weight:300;'
            f'color:{_spy_col};">{_spy1d:+.2f}%</div>'
            f'<div style="font-family:{_SANS};font-size:0.72rem;color:{DIM};margin-top:2px;">'
            f'52W: {market_stats.get("distance_from_high",0):+.1f}%</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Action items ──────────────────────────────────────────────────────
        section_header("Action Items")
        actions = []
        regime_constraints = optimizer.regime_constraints.get(current_regime, {})
        max_equity = regime_constraints.get('max_equity_exposure', 1.0)
        if max_equity < 1.0:
            excess = (1.0 - max_equity) * summary['total_value']
            actions.append({'priority':'HIGH','type':'Reduce Exposure',
                'action': f"{current_regime} regime: max {max_equity*100:.0f}% equity. "
                          f"Consider raising ~${excess:,.0f} cash.", 'score':90})
        largest_w = summary.get('largest_position_weight', 0)
        if largest_w > 30:
            actions.append({'priority':'MEDIUM','type':'Concentration Risk',
                'action': f"{summary.get('largest_position_ticker','')} at {largest_w:.1f}% — "
                          f"trim to under 30%.", 'score':70})
        if vix_now > 30:
            actions.append({'priority':'HIGH','type':'Elevated Volatility',
                'action': f"VIX {vix_now:.1f} — elevated fear. "
                          f"Consider defensive positioning.", 'score':95})
        dist_high = market_stats.get('distance_from_high', -100)
        if dist_high > -5:
            actions.append({'priority':'LOW','type':'Near 52W High',
                'action': f"SPY within 5% of 52W high. Monitor for overbought signals.", 'score':25})
        if not actions:
            actions.append({'priority':'NONE','type':'All Clear',
                'action': f"Portfolio looks healthy for {current_regime} regime.", 'score':0})
        actions.sort(key=lambda x: x['score'], reverse=True)
        _pri_colors = {'HIGH':LOSS,'MEDIUM':AMBER,'LOW':ACCENT,'NONE':GREEN}
        for act in actions:
            col = _pri_colors.get(act['priority'], SUBTLE)
            st.markdown(
                f'<div style="background:{CARD};border-radius:10px;'
                f'border-top:2px solid {col};'
                f'padding:0.75rem 1.1rem;margin-bottom:0.5rem;'
                f'box-shadow:0 2px 10px rgba(0,0,0,0.35);">'
                f'<div style="font-family:{_SANS};font-size:0.6rem;font-weight:700;'
                f'letter-spacing:0.14em;text-transform:uppercase;color:{col};">'
                f'{act["priority"]} &nbsp;·&nbsp; {act["type"]}</div>'
                f'<div style="font-family:{_SANS};font-size:0.84rem;color:{DIM};'
                f'margin-top:0.3rem;line-height:1.5;">{act["action"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── Earnings calendar ─────────────────────────────────────────────────
        section_header("Earnings — Next 30 Days")
        with st.spinner("Checking earnings…"):
            earnings_data = []
            today_d  = datetime.now().date()
            cutoff_d = today_d + timedelta(days=30)
            for t in tickers:
                try:
                    cal = yf.Ticker(t).calendar
                    ed  = None
                    if isinstance(cal, dict):
                        _ed = cal.get('Earnings Date')
                        ed  = (_ed[0] if isinstance(_ed,(list,tuple)) and _ed else _ed)
                    elif isinstance(cal, pd.DataFrame) and not cal.empty:
                        if 'Earnings Date' in cal.index:
                            ed = cal.loc['Earnings Date'].iloc[0]
                    if ed is not None:
                        if hasattr(ed,'date'): ed = ed.date()
                        elif isinstance(ed,str):
                            try: ed = datetime.strptime(ed[:10],'%Y-%m-%d').date()
                            except: ed = None
                        if ed and today_d <= ed <= cutoff_d:
                            days_left = (ed - today_d).days
                            earnings_data.append({
                                'Ticker': t,
                                'Date':   ed.strftime('%Y-%m-%d'),
                                'Days':   days_left,
                                'Urgency': 'This week' if days_left<=7 else '2 weeks' if days_left<=14 else 'This month',
                            })
                except Exception:
                    pass
        if earnings_data:
            flex_table(
                pd.DataFrame(earnings_data).sort_values('Days'),
                columns=[
                    {"key": "Ticker",  "label": "Ticker",  "width": "20%", "align": "left"},
                    {"key": "Date",    "label": "Date",    "width": "28%", "align": "left"},
                    {"key": "Days",    "label": "Days Out", "width": "22%", "align": "right", "numeric": True},
                    {"key": "Urgency", "label": "Urgency", "width": "30%", "align": "left"},
                ],
                key="earnings",
            )
            imminent = [e['Ticker'] for e in earnings_data if e['Days'] <= 7]
            if imminent:
                st.warning(f"Earnings this week: **{', '.join(imminent)}**")
        else:
            st.success("No earnings in the next 30 days for your holdings.")

        # ── Top 3 positions ───────────────────────────────────────────────────
        section_header("Top Positions by Value")
        top3 = position_metrics.nlargest(3,'current_value')
        cols = st.columns(min(3, len(top3)))
        _pos_accents = [ACCENT, PURPLE, AMBER]
        for idx, (_, pos) in enumerate(top3.iterrows()):
            pnl    = float(pos['total_pnl'])
            pc     = GAIN if pnl >= 0 else LOSS
            _ac    = _pos_accents[idx % len(_pos_accents)]
            with cols[idx]:
                st.markdown(
                    metric_card(
                        label=pos["ticker"],
                        value=f'${float(pos["current_value"]):,.0f}',
                        delta=f'P&L ${pnl:,.0f} ({float(pos["total_pnl_pct"]):+.1f}%)  ·  {float(pos["weight_pct"]):.1f}% of portfolio',
                        accent=_ac,
                        accent2=PURPLE if _ac == ACCENT else ACCENT,
                    ),
                    unsafe_allow_html=True,
                )

        # ── Mini charts ───────────────────────────────────────────────────────
        section_header("Portfolio Snapshot")
        ch1, ch2 = st.columns(2)
        with ch1:
            fig_bar = go.Figure(data=[go.Bar(
                x=position_metrics['ticker'],
                y=position_metrics['total_pnl'],
                marker_color=position_metrics['total_pnl'].apply(lambda x: GAIN if x>=0 else LOSS),
                text=position_metrics['total_pnl'],
                texttemplate='$%{text:,.0f}', textposition='outside',
            )])
            fig_bar.update_layout(**carbon_plotly_layout(title="Position P&L", height=280,
                                                         xaxis_title="", yaxis_title="P&L ($)", showlegend=False))
            fig_bar.add_hline(y=0, line_dash="dash", line_color=BORDER)
            st.plotly_chart(fig_bar, use_container_width=True)
        with ch2:
            _pie_labels = list(position_metrics['ticker'])
            _pie_vals   = list(position_metrics['current_value'])
            _pie_colors = [_HOLDING_COLORS[i % len(_HOLDING_COLORS)] for i in range(len(_pie_labels))]
            if _cash_balance > 0:
                _pie_labels.append("CASH"); _pie_vals.append(_cash_balance); _pie_colors.append(GREEN)
            _pie_total = sum(_pie_vals) or 1
            _pie_text  = [
                f"<b>{lbl}</b><br>{v/_pie_total*100:.1f}%"
                if v / _pie_total >= 0.05 else ""
                for lbl, v in zip(_pie_labels, _pie_vals)
            ]
            fig_pie = go.Figure(data=[go.Pie(
                labels=_pie_labels,
                values=_pie_vals,
                hole=0.44,
                text=_pie_text,
                textinfo='text',
                textposition='outside',
                automargin=False,
                marker=dict(colors=_pie_colors, line=dict(color=BG, width=1.5)),
                textfont=dict(size=12),
                domain=dict(x=[0.0, 0.72], y=[0.0, 1.0]),
            )])
            _pie_layout = carbon_plotly_layout(title="Allocation", height=480)
            _pie_layout.update(dict(
                showlegend=True,
                legend=dict(
                    x=0.75, y=0.5,
                    xanchor='left', yanchor='middle',
                    font=dict(size=11),
                    bgcolor='rgba(0,0,0,0)',
                    itemsizing='constant',
                ),
                margin=dict(l=20, r=20, t=48, b=20),
            ))
            fig_pie.update_layout(**_pie_layout)
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Rebalancing panel ─────────────────────────────────────────────────
        section_header("Rebalance Positions")
        _rb_tab_exist, _rb_tab_new = st.tabs(["Adjust Existing Position", "Add New Position"])

        with _rb_tab_exist:
            _rb_ticker = st.selectbox(
                "Select position",
                options=list(position_metrics['ticker']),
                key="_rb_sel_ticker",
            )
            if _rb_ticker:
                _rb_row = position_metrics[position_metrics['ticker'] == _rb_ticker].iloc[0]
                _rb_sh  = float(_rb_row['shares'])
                _rb_cb  = float(_rb_row['cost_basis'])
                _rb_px  = float(_rb_row['current_price'])
                _rb_val = float(_rb_row['current_value'])
                _rb_pnl = float(_rb_row['total_pnl'])
                _rb_pct = float(_rb_row['total_pnl_pct'])
                _rb_pc  = GAIN if _rb_pnl >= 0 else LOSS

                # Current position summary cards
                _rb_c1, _rb_c2, _rb_c3, _rb_c4 = st.columns(4)
                _rb_c1.markdown(
                    f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:0.8rem 1rem;">'
                    f'<div style="font-size:0.62rem;color:{DIM};text-transform:uppercase;letter-spacing:0.1em;">Shares</div>'
                    f'<div style="font-size:1.3rem;color:{FG};font-weight:500;">{_rb_sh:,.4g}</div></div>',
                    unsafe_allow_html=True)
                _rb_c2.markdown(
                    f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:0.8rem 1rem;">'
                    f'<div style="font-size:0.62rem;color:{DIM};text-transform:uppercase;letter-spacing:0.1em;">Avg Cost</div>'
                    f'<div style="font-size:1.3rem;color:{FG};font-weight:500;">${_rb_cb:.2f}</div></div>',
                    unsafe_allow_html=True)
                _rb_c3.markdown(
                    f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:0.8rem 1rem;">'
                    f'<div style="font-size:0.62rem;color:{DIM};text-transform:uppercase;letter-spacing:0.1em;">Market Value</div>'
                    f'<div style="font-size:1.3rem;color:{FG};font-weight:500;">${_rb_val:,.0f}</div></div>',
                    unsafe_allow_html=True)
                _rb_c4.markdown(
                    f'<div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:0.8rem 1rem;">'
                    f'<div style="font-size:0.62rem;color:{DIM};text-transform:uppercase;letter-spacing:0.1em;">Unrealised P&L</div>'
                    f'<div style="font-size:1.3rem;color:{_rb_pc};font-weight:500;">'
                    f'{"+" if _rb_pnl>=0 else ""}${_rb_pnl:,.0f} ({_rb_pct:+.1f}%)</div></div>',
                    unsafe_allow_html=True)

                st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

                _act_col, _qty_col, _px_col, _btn_col = st.columns([2, 2, 2, 1])
                _rb_action = _act_col.selectbox(
                    "Action", ["Buy (add to position)", "Sell (reduce position)", "Close entire position"],
                    key="_rb_action", label_visibility="visible")
                _rb_qty = _qty_col.number_input(
                    "Shares", min_value=0.0001, max_value=_rb_sh if "Sell" in _rb_action else 1e9,
                    value=round(_rb_sh * 0.1, 4) if "Sell" in _rb_action else 1.0,
                    step=1.0, format="%.4g", key="_rb_qty",
                    disabled=_rb_action == "Close entire position")
                _rb_exec_px = _px_col.number_input(
                    "Execution price ($)", min_value=0.0001,
                    value=_rb_px, step=0.01, format="%.2f", key="_rb_exec_px",
                    disabled=_rb_action == "Close entire position")

                # Live preview
                if _rb_action == "Buy (add to position)":
                    _new_sh   = _rb_sh + _rb_qty
                    _new_cb   = (_rb_sh * _rb_cb + _rb_qty * _rb_exec_px) / _new_sh
                    _trade_val = _rb_qty * _rb_exec_px
                    st.caption(f"Preview: {_new_sh:,.4g} shares @ avg ${_new_cb:.2f} cost  ·  Trade notional ${_trade_val:,.2f}")
                elif _rb_action == "Sell (reduce position)":
                    _new_sh   = _rb_sh - _rb_qty
                    _realised = _rb_qty * (_rb_exec_px - _rb_cb)
                    _real_col = GAIN if _realised >= 0 else LOSS
                    _trade_val = _rb_qty * _rb_exec_px
                    _rem_lbl  = f"{_new_sh:,.4g} shares remain" if _new_sh > 0 else "position fully closed"
                    st.caption(f"Preview: {_rem_lbl}  ·  Realised P&L "
                               f"{'+'if _realised>=0 else ''}${_realised:,.2f}  ·  Proceeds ${_trade_val:,.2f}")
                else:
                    _realised = _rb_sh * (_rb_px - _rb_cb)
                    st.caption(f"Preview: close all {_rb_sh:,.4g} shares  ·  Realised P&L "
                               f"{'+'if _realised>=0 else ''}${_realised:,.2f}")

                if _btn_col.button("Apply", key="_rb_apply", type="primary"):
                    _manual = st.session_state.get('manual_positions', [])
                    if _rb_action == "Buy (add to position)":
                        for _mp in _manual:
                            if _mp['ticker'] == _rb_ticker:
                                _new_sh2  = float(_mp['shares']) + _rb_qty
                                _mp['cost_basis'] = (float(_mp['shares']) * float(_mp['cost_basis']) + _rb_qty * _rb_exec_px) / _new_sh2
                                _mp['shares'] = _new_sh2
                                break
                    elif _rb_action == "Sell (reduce position)":
                        for _mp in _manual:
                            if _mp['ticker'] == _rb_ticker:
                                _rem2 = float(_mp['shares']) - _rb_qty
                                if _rem2 <= 0:
                                    _manual = [p for p in _manual if p['ticker'] != _rb_ticker]
                                else:
                                    _mp['shares'] = _rem2
                                break
                    else:  # close entire
                        _manual = [p for p in _manual if p['ticker'] != _rb_ticker]

                    st.session_state['manual_positions'] = _manual
                    _udf2 = pd.DataFrame(_manual) if _manual else pd.DataFrame()
                    if not _udf2.empty:
                        st.session_state['positions'] = _udf2
                        save_portfolio(_udf2)
                    else:
                        st.session_state.pop('positions', None)
                    st.success(f"Updated {_rb_ticker}.")
                    st.rerun()

        with _rb_tab_new:
            st.caption("Add a new ticker to your portfolio.")
            _n1, _n2, _n3, _n4, _n5 = st.columns([2, 2, 2, 2, 1])
            _new_t  = _n1.text_input("Ticker", placeholder="TSLA", key="_rb_new_t").upper().strip()
            _new_sh = _n2.number_input("Shares", min_value=0.0001, step=1.0, format="%.4g", key="_rb_new_sh")
            _new_cb = _n3.number_input("Avg cost ($)", min_value=0.0001, step=0.01, format="%.2f", key="_rb_new_cb")
            _new_dt = _n4.date_input("Purchase date", value=datetime.today(), key="_rb_new_dt")
            if _n5.button("Add", key="_rb_new_add", type="primary"):
                if _new_t and _new_sh > 0 and _new_cb > 0:
                    _manual2 = st.session_state.get('manual_positions', [])
                    _existing_t = [p['ticker'] for p in _manual2]
                    if _new_t in _existing_t:
                        st.warning(f"{_new_t} already in portfolio — use 'Adjust Existing Position' to add shares.")
                    else:
                        _manual2.append({'ticker': _new_t, 'shares': _new_sh,
                                         'cost_basis': _new_cb, 'purchase_date': str(_new_dt)})
                        st.session_state['manual_positions'] = _manual2
                        _udf3 = pd.DataFrame(_manual2)
                        st.session_state['positions'] = _udf3
                        save_portfolio(_udf3)
                        st.success(f"Added {_new_t} to portfolio.")
                        st.rerun()
                else:
                    st.warning("Fill in ticker, shares, and cost before adding.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_positions:
    if summary is None:
        st.info("Load a portfolio to view positions.")
    else:
        _cash_balance = st.session_state.get('cash_balance', 0.0)
        _sv, _sp = st.columns([5,1])
        with _sp:
            if st.button("Save Portfolio"):
                save_portfolio(positions_df)
                st.success("Saved!")

        # Position details table
        section_header("Position Details")
        _pm_display = position_metrics.copy()
        _pm_display['Shares']   = _pm_display['shares'].apply(lambda v: f"{float(v):,.4g}")
        _pm_display['Avg Cost'] = _pm_display['cost_basis'].apply(lambda v: f"${float(v):.2f}")
        _pm_display['Price']    = _pm_display['current_price'].apply(
            lambda v: f"${float(v):.2f}" if v == v and v is not None else "—")
        _pm_display['Value']    = _pm_display['current_value'].apply(
            lambda v: f"${float(v):,.2f}" if v == v and v is not None else "—")
        if _cash_balance > 0:
            import pandas as _pd_pos
            _cash_row = _pd_pos.DataFrame([{
                'ticker': 'CASH', 'Shares': '—', 'Avg Cost': '—',
                'Price': '—', 'Value': f'${_cash_balance:,.2f}', 'total_pnl': 0,
                'total_pnl_pct': 0, 'shares': None, 'cost_basis': None,
                'current_price': None, 'current_value': _cash_balance,
            }])
            _pm_display = _pd_pos.concat([_pm_display, _cash_row], ignore_index=True)
        flex_table(
            _pm_display,
            columns=[
                {"key": "ticker",        "label": "Ticker",   "width": "15%", "align": "left"},
                {"key": "Shares",        "label": "Shares",   "width": "13%", "align": "right"},
                {"key": "Avg Cost",      "label": "Avg Cost", "width": "15%", "align": "right"},
                {"key": "Price",         "label": "Price",    "width": "15%", "align": "right"},
                {"key": "Value",         "label": "Value",    "width": "17%", "align": "right"},
                {"key": "total_pnl",     "label": "P&L ($)",  "width": "13%", "align": "right",
                 "numeric": True, "color_scale": "rg",
                 "fmt": lambda v: f'{"+" if v>=0 else ""}${abs(v):,.2f}'},
                {"key": "total_pnl_pct", "label": "P&L %",   "width": "12%", "align": "right",
                 "numeric": True, "color_scale": "rg",
                 "fmt": lambda v: f"{v:+.2f}%"},
            ],
            key="positions",
        )

        # Update positions
        with st.expander("Update Positions"):
            for _up_idx, _up_row in positions_df.iterrows():
                _up_t = _up_row['ticker']; _up_sh = float(_up_row['shares']); _up_cb = float(_up_row['cost_basis'])
                st.markdown(f"<span style='color:{ACCENT};font-style:italic;'>{_up_t}</span>"
                            f"<span style='color:{DIM};font-size:0.78rem;'> — {_up_sh:,.4g} @ ${_up_cb:.2f}</span>",
                            unsafe_allow_html=True)
                _uc1,_uc2,_uc3,_uc4 = st.columns([2,2,2,1])
                _upact = _uc1.selectbox("Action",["Buy More","Sell Shares"],key=f"_ua_{_up_t}_{_up_idx}",label_visibility="collapsed")
                _upqty = _uc2.number_input("Qty",min_value=0.0001,step=1.0,format="%.4g",key=f"_uq_{_up_t}_{_up_idx}",label_visibility="collapsed")
                _uppx  = _uc3.number_input("Price",min_value=0.0001,step=0.01,value=float(_up_cb),format="%.2f",key=f"_up_{_up_t}_{_up_idx}",label_visibility="collapsed")
                if _uc4.button("Apply",key=f"_uapp_{_up_t}_{_up_idx}",type="primary"):
                    _manual = st.session_state.get('manual_positions',[])
                    for _mp in _manual:
                        if _mp['ticker'] == _up_t:
                            if _upact == "Buy More":
                                _new = _up_sh + _upqty
                                _mp['cost_basis'] = (_up_sh*_up_cb + _upqty*_uppx)/_new
                                _mp['shares'] = _new
                            else:
                                _rem = _up_sh - _upqty
                                if _rem <= 0: _manual = [p for p in _manual if p['ticker']!=_up_t]
                                else: _mp['shares'] = _rem
                            break
                    st.session_state['manual_positions'] = _manual
                    _udf = pd.DataFrame(_manual) if _manual else pd.DataFrame()
                    if not _udf.empty: st.session_state['positions'] = _udf; save_portfolio(_udf)
                    else: st.session_state.pop('positions',None)
                    st.rerun()
                st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:0.5rem 0;'>",unsafe_allow_html=True)

        # Holdings charts
        section_header("Holdings Breakdown")
        _hc1, _hc2 = st.columns(2)
        with _hc1:
            _pv = st.radio("Pie",["Holdings","Allocation"],horizontal=True,key="_pv2",label_visibility="collapsed")
            if _pv == "Holdings":
                _p2_labels = list(position_metrics['ticker'])
                _p2_vals   = list(position_metrics['current_value'])
                _p2_colors = [_HOLDING_COLORS[i % len(_HOLDING_COLORS)] for i in range(len(_p2_labels))]
                _p2_total  = sum(_p2_vals) or 1
                _p2_text   = [
                    f"<b>{lbl}</b><br>{v/_p2_total*100:.1f}%"
                    if v / _p2_total >= 0.05 else ""
                    for lbl, v in zip(_p2_labels, _p2_vals)
                ]
                fig_p2 = go.Figure(data=[go.Pie(
                    labels=_p2_labels, values=_p2_vals,
                    hole=0.44, text=_p2_text,
                    textinfo='text', textposition='outside',
                    automargin=False,
                    marker=dict(colors=_p2_colors, line=dict(color=BG, width=1.5)),
                    textfont=dict(size=12),
                    domain=dict(x=[0.0, 0.72], y=[0.0, 1.0]),
                )])
                _p2_layout = carbon_plotly_layout(title="Holdings by Value", height=480)
                _p2_layout.update(dict(
                    showlegend=True,
                    legend=dict(
                        x=0.75, y=0.5,
                        xanchor='left', yanchor='middle',
                        font=dict(size=11),
                        bgcolor='rgba(0,0,0,0)',
                        itemsizing='constant',
                    ),
                    margin=dict(l=20, r=20, t=48, b=20),
                ))
                fig_p2.update_layout(**_p2_layout)
                st.plotly_chart(fig_p2, use_container_width=True)
            else:
                _wl = list(position_metrics['ticker']); _wv = list(position_metrics['weight_pct'])
                fig_w = go.Figure(data=[go.Bar(x=_wl,y=_wv,
                    marker_color=[_HOLDING_COLORS[i%len(_HOLDING_COLORS)] for i in range(len(_wl))])])
                fig_w.update_layout(**carbon_plotly_layout(title="Allocation %",height=360,
                                                           xaxis_title="",yaxis_title="Weight %"))
                st.plotly_chart(fig_w, use_container_width=True)
        with _hc2:
            fig_b2 = go.Figure(data=[go.Bar(
                x=position_metrics['ticker'], y=position_metrics['total_pnl'],
                marker_color=position_metrics['total_pnl'].apply(lambda x: GAIN if x>=0 else LOSS),
                text=position_metrics['total_pnl'], texttemplate='$%{text:,.0f}', textposition='outside',
            )])
            fig_b2.update_layout(**carbon_plotly_layout(title="Position P&L ($)",height=360,
                                                        xaxis_title="",yaxis_title="P&L ($)",showlegend=False))
            fig_b2.add_hline(y=0,line_dash="dash",line_color=BORDER)
            st.plotly_chart(fig_b2, use_container_width=True)

        # Correlation
        section_header("Correlation & Diversification")
        if len(tickers) >= 2:
            with st.spinner("Computing correlations…"):
                try:
                    raw_c = loader.fetch_historical_data(tickers, start_date=datetime.now()-timedelta(days=90))
                    px_c  = pd.DataFrame({t:raw_c[t]['Close'] for t in tickers
                                          if raw_c.get(t) is not None and not raw_c[t].empty})
                    if px_c.shape[1] >= 2:
                        rets_c = px_c.pct_change().dropna(); corr = rets_c.corr(); n = len(corr)
                        off = [abs(corr.iloc[i,j]) for i in range(n) for j in range(i+1,n)]
                        avg_corr = sum(off)/len(off) if off else 0
                        d_score  = (1-avg_corr)*100
                        d_col    = ACCENT if d_score>=60 else SUBTLE if d_score>=40 else LOSS
                        d_lbl    = "Good" if d_score>=60 else "Moderate" if d_score>=40 else "Poor"
                        high_pairs = [(corr.index[i],corr.columns[j],corr.iloc[i,j])
                                      for i in range(n) for j in range(i+1,n) if abs(corr.iloc[i,j])>=0.85]
                        cc1,cc2 = st.columns([1,3])
                        with cc1:
                            st.markdown(
                                f'<div style="text-align:center;padding:1.5rem;">'
                                f'<p style="margin:0;font-size:0.7rem;color:{SUBTLE};text-transform:uppercase;'
                                f'letter-spacing:0.1em;">Diversification</p>'
                                f'<p style="margin:0;font-size:3rem;font-weight:300;color:{d_col};">{d_score:.0f}</p>'
                                f'<p style="margin:0;font-size:0.8rem;color:{d_col};">{d_lbl}</p>'
                                f'</div>', unsafe_allow_html=True)
                            st.caption(f"Avg corr: {avg_corr:.2f}")
                            for t1,t2,val in high_pairs:
                                st.warning(f"{t1} ↔ {t2}: {val:.2f}")
                        with cc2:
                            fig_h = go.Figure(data=go.Heatmap(
                                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                                colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
                                text=corr.round(2).values, texttemplate='%{text}',
                                textfont=dict(size=11), showscale=True))
                            fig_h.update_layout(**carbon_plotly_layout(title="60-Day Correlations",height=360))
                            st.plotly_chart(fig_h, use_container_width=True)
                except Exception as e:
                    st.warning(f"Correlation failed: {e}")
        else:
            st.info("Add at least 2 positions for correlation analysis.")

        # Performance analytics
        section_header("Performance Analytics")
        benchmark_ticker = st.selectbox("Benchmark",['SPY','QQQ','DIA','IWM','None'],key='bench_sel')
        with st.spinner("Calculating performance…"):
            _start = datetime.now()-timedelta(days=365)
            hist_d = loader.fetch_historical_data(tickers, start_date=_start)
            bench_data = None
            if benchmark_ticker != 'None':
                _bd = loader.fetch_historical_data([benchmark_ticker], start_date=_start)
                if _bd.get(benchmark_ticker) is not None and not _bd[benchmark_ticker].empty:
                    bench_data = _bd[benchmark_ticker]['Close']
            _weights = {r['ticker']:r['current_value']/position_metrics['current_value'].sum()
                        for _,r in position_metrics.iterrows()}
            port_rets = analytics.calculate_returns(hist_d, _weights)
            metrics   = {}
            if not port_rets.empty and len(port_rets)>20:
                metrics = analytics.calculate_all_metrics(port_rets)

        tp1,tp2,tp3 = st.tabs(["Performance","P&L History","Risk"])
        with tp1:
            if metrics:
                mc1,mc2,mc3 = st.columns(3)
                mc1.metric("Ann. Return",   f"{metrics['cagr']:.2f}%")
                mc1.metric("Total Return",  f"{metrics['total_return']:.2f}%")
                mc2.metric("Volatility",    f"{metrics['volatility']:.2f}%")
                mc2.metric("Sharpe Ratio",  f"{metrics['sharpe_ratio']:.2f}")
                mc3.metric("Max Drawdown",  f"{metrics['max_drawdown']:.2f}%")
                mc3.metric("Win Rate",      f"{metrics['win_rate']:.1f}%")
                if bench_data is not None and not bench_data.empty:
                    br = bench_data.pct_change().dropna()
                    ci = port_rets.index.intersection(br.index)
                    if len(ci)>20:
                        pr2 = port_rets.loc[ci]; br2 = br.loc[ci]
                        beta  = pr2.cov(br2)/br2.var() if br2.var()>0 else 0
                        alpha = (pr2.mean()-beta*br2.mean())*252*100
                        ba1,ba2 = st.columns(2)
                        ba1.metric(f"Beta vs {benchmark_ticker}",f"{beta:.2f}")
                        ba2.metric(f"Alpha (Ann.)",f"{alpha:+.2f}%")
                cum = (1+port_rets).cumprod()
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(x=cum.index,y=(cum-1)*100,name='Portfolio',
                                             line=dict(color=ACCENT,width=2),fill='tozeroy',
                                             fillcolor='rgba(34,211,238,0.08)'))
                if bench_data is not None:
                    bc = (1+bench_data.pct_change().dropna()).cumprod()
                    fig_cum.add_trace(go.Scatter(x=bc.index,y=(bc-1)*100,name=benchmark_ticker,
                                                 line=dict(color=SUBTLE,width=1.5,dash='dash')))
                fig_cum.update_layout(**carbon_plotly_layout(title="Cumulative Return vs Benchmark",
                                                             height=380,hovermode='x unified',
                                                             xaxis_title="Date",yaxis_title="Return (%)"))
                st.plotly_chart(fig_cum, use_container_width=True)
                dd = (cum - cum.expanding().max())/cum.expanding().max()*100
                fig_dd = go.Figure(go.Scatter(x=dd.index,y=dd,name='Drawdown',
                                              line=dict(color=LOSS,width=2),fill='tozeroy',
                                              fillcolor='rgba(251,113,133,0.1)'))
                fig_dd.update_layout(**carbon_plotly_layout(title="Drawdown",height=260,
                                                            xaxis_title="Date",yaxis_title="Drawdown (%)"))
                st.plotly_chart(fig_dd, use_container_width=True)
            else:
                st.warning("Insufficient data (need >20 days).")

        with tp2:
            lb = st.slider("Lookback (days)",30,365,180,30,key='pnl_lb')
            hd = loader.fetch_historical_data(tickers, start_date=datetime.now()-timedelta(days=lb))
            fig_ph = go.Figure()
            for i,t in enumerate(tickers):
                if hd.get(t) is not None and not hd[t].empty:
                    px = hd[t]['Close']; row = positions_df[positions_df['ticker']==t].iloc[0]
                    fig_ph.add_trace(go.Scatter(x=px.index,y=px*float(row['shares'])-float(row['cost_basis'])*float(row['shares']),
                                                name=t,line=dict(color=_HOLDING_COLORS[i%len(_HOLDING_COLORS)],width=1.5)))
            fig_ph.add_hline(y=0,line_dash="dash",line_color=BORDER)
            fig_ph.update_layout(**carbon_plotly_layout(title="Unrealised P&L History ($)",
                                                        height=420,hovermode='x unified'))
            st.plotly_chart(fig_ph, use_container_width=True)

        with tp3:
            if metrics and not port_rets.empty:
                pv       = summary['total_value']; dv = port_rets.std()
                v1,v2,v3,v4 = st.columns(4)
                v1.metric("VaR 95% (Param)",  f"${1.645*dv*pv:,.0f}")
                v2.metric("VaR 99% (Param)",  f"${2.326*dv*pv:,.0f}")
                v3.metric("VaR 95% (Hist)",   f"${abs(port_rets.quantile(0.05))*pv:,.0f}")
                v4.metric("VaR 99% (Hist)",   f"${abs(port_rets.quantile(0.01))*pv:,.0f}")
                stress = pd.DataFrame([{'SPY Move':f"{s:+.0f}%",
                    'Portfolio Impact':f"${pv*(s/100):,.0f}",
                    'Est. Value':f"${pv+pv*(s/100):,.0f}"} for s in [-10,-20,-30,-40]])
                flex_table(
                    stress,
                    columns=[
                        {"key": "SPY Move",         "label": "SPY Move",         "width": "33%", "align": "left"},
                        {"key": "Portfolio Impact", "label": "Portfolio Impact", "width": "34%", "align": "right"},
                        {"key": "Est. Value",       "label": "Est. Value",       "width": "33%", "align": "right"},
                    ],
                    key="stress",
                )

        # Export
        st.download_button("Download CSV", position_metrics.to_csv(index=False),
                           f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_options:
    _opts       = st.session_state.get('options_positions', [])
    _singles    = [(i,p) for i,p in enumerate(_opts) if p.get('type','single')=='single']
    _strategies = [(i,p) for i,p in enumerate(_opts) if p.get('type')=='strategy']
    _open_uls   = set()
    for _,p in _singles+_strategies:
        if p['status']=='open': _open_uls.add(p['underlying'])
    _sb_ul_loaded = st.session_state.get('_sb_ul_loaded','')
    if _sb_ul_loaded: _open_uls.add(_sb_ul_loaded)
    _ul_prices = loader.fetch_current_prices(list(_open_uls)) if _open_uls else {}
    _total_opt_val = _total_opt_pnl = 0.0

    # ── Strategy builder ──────────────────────────────────────────────────────
    if '_sb_show' not in st.session_state:
        st.session_state['_sb_show'] = False
    _tlbl  = "Close Builder" if st.session_state['_sb_show'] else "Add Multi-Leg Strategy"
    _ttype = "secondary" if st.session_state['_sb_show'] else "primary"
    if st.button(_tlbl, key="_strat_toggle", type=_ttype):
        st.session_state['_sb_show'] = not st.session_state['_sb_show']
        if not st.session_state['_sb_show']:
            for k in ['_sb_exps','_sb_ul_loaded','_sb_near_chain','_sb_far_chain',
                      '_sb_calls_chain','_sb_puts_chain','_sb_single_exp_loaded',
                      '_sb_near_exp_loaded','_sb_far_exp_loaded','_sb_custom_legs']:
                st.session_state.pop(k,None)
        st.rerun()

    if st.session_state['_sb_show']:
        with st.container(border=True):
            st.markdown("#### Multi-Leg Strategy Builder")
            _sb_c1,_sb_c2,_sb_c3 = st.columns([2,2,1])
            with _sb_c1:
                _sb_tmpl = st.selectbox("Template",list(_STRAT_TEMPLATES.keys()),key="_sb_tmpl_sel")
                st.caption(_STRAT_TEMPLATES[_sb_tmpl]['desc'])
            with _sb_c2:
                _sb_ul = st.text_input("Underlying",key="_sb_ul_in",placeholder="e.g. SPY").upper().strip()
            with _sb_c3:
                st.markdown("<br>",unsafe_allow_html=True)
                if st.button("Fetch Chain",key="_sb_fetch"):
                    if _sb_ul:
                        try:
                            _exps2 = list(yf.Ticker(_sb_ul).options)
                            if _exps2:
                                st.session_state['_sb_exps']      = _exps2
                                st.session_state['_sb_ul_loaded'] = _sb_ul
                                for k in ['_sb_near_chain','_sb_far_chain','_sb_calls_chain','_sb_puts_chain']:
                                    st.session_state.pop(k,None)
                            else: st.error(f"No options for {_sb_ul}")
                        except Exception as e: st.error(str(e))

            _ti   = _STRAT_TEMPLATES[_sb_tmpl]; _is_cal = _ti['calendar']
            if st.session_state.get('_sb_exps'):
                _exps2 = st.session_state['_sb_exps']
                st.markdown("---")
                if _is_cal:
                    _ec1,_ec2,_ec3 = st.columns([2,2,1])
                    _sb_near = _ec1.selectbox("Near Expiry",_exps2,index=0,key="_sb_near_exp")
                    _sb_far  = _ec2.selectbox("Far Expiry",_exps2,index=min(3,len(_exps2)-1),key="_sb_far_exp")
                    _ec3.markdown("<br>",unsafe_allow_html=True)
                    if _ec3.button("Load",key="_sb_load_cal"):
                        try:
                            _ul2 = st.session_state['_sb_ul_loaded']; _ot0 = _ti['legs'][0]['option_type']
                            _stk0 = yf.Ticker(_ul2)
                            _nc = _stk0.option_chain(_sb_near); _fc = _stk0.option_chain(_sb_far)
                            st.session_state['_sb_near_chain']      = (_nc.calls if _ot0=='call' else _nc.puts).reset_index(drop=True).to_dict(orient='records')
                            st.session_state['_sb_far_chain']       = (_fc.calls if _ot0=='call' else _fc.puts).reset_index(drop=True).to_dict(orient='records')
                            st.session_state['_sb_near_exp_loaded'] = _sb_near
                            st.session_state['_sb_far_exp_loaded']  = _sb_far
                        except Exception as e: st.error(str(e))
                else:
                    _ec1,_ec2 = st.columns([3,1])
                    _sb_sexp = _ec1.selectbox("Expiration",_exps2,index=min(2,len(_exps2)-1),key="_sb_single_exp")
                    _ec2.markdown("<br>",unsafe_allow_html=True)
                    if _ec2.button("Load",key="_sb_load_std"):
                        try:
                            _full = yf.Ticker(st.session_state['_sb_ul_loaded']).option_chain(_sb_sexp)
                            st.session_state['_sb_calls_chain']       = _full.calls.reset_index(drop=True).to_dict(orient='records')
                            st.session_state['_sb_puts_chain']        = _full.puts.reset_index(drop=True).to_dict(orient='records')
                            st.session_state['_sb_single_exp_loaded'] = _sb_sexp
                        except Exception as e: st.error(str(e))

                _chains_ready = ((st.session_state.get('_sb_near_chain') and st.session_state.get('_sb_far_chain'))
                                 if _is_cal else bool(st.session_state.get('_sb_calls_chain')))
                if _sb_tmpl=='Custom' and _chains_ready:
                    if '_sb_custom_legs' not in st.session_state: st.session_state['_sb_custom_legs'] = []
                    if st.button("+ Add Leg",key="_sb_add_cl"): st.session_state['_sb_custom_legs'].append({}); st.rerun()

                if _chains_ready:
                    st.markdown("---"); st.markdown("**Configure legs:**")
                    _ul_px3 = _ul_prices.get(st.session_state.get('_sb_ul_loaded',''),0)
                    _leg_cfg = []
                    _tmpl_legs = _ti['legs'] if _sb_tmpl!='Custom' else \
                        [{'option_type':'call','position':'long','offset':0.0,'label':f'Leg {i+1}','far':False}
                         for i in range(len(st.session_state.get('_sb_custom_legs',[])))]
                    for _li,_lt in enumerate(_tmpl_legs):
                        _ot3  = _lt['option_type']; _pos3 = _lt['position']; _lbl3 = _lt['label']
                        _is_far3 = _lt.get('far',False)
                        if _is_cal:
                            _cr3 = st.session_state['_sb_far_chain'] if _is_far3 else st.session_state['_sb_near_chain']
                            _exp3 = st.session_state['_sb_far_exp_loaded'] if _is_far3 else st.session_state['_sb_near_exp_loaded']
                        else:
                            _cr3 = st.session_state.get('_sb_calls_chain' if _ot3=='call' else '_sb_puts_chain',[])
                            _exp3 = st.session_state.get('_sb_single_exp_loaded','')
                        _cdf3 = pd.DataFrame(_cr3)
                        if _cdf3.empty: continue
                        _icon3 = '▲' if _pos3=='long' else '▼'
                        _icol3 = GAIN if _pos3=='long' else LOSS
                        st.markdown(f"<small><span style='color:{_icol3};font-weight:bold;'>"
                                    f"{_icon3} {_pos3.upper()} {_ot3.upper()}</span> — {_lbl3} · exp {_exp3}</small>",
                                    unsafe_allow_html=True)
                        _target3 = _ul_px3*(1+_lt.get('offset',0.0))
                        _cdf3 = _cdf3.copy(); _cdf3['_d3'] = abs(_cdf3['strike']-_target3)
                        _def3 = int(_cdf3['_d3'].idxmin())
                        def _slbl3(i,df=_cdf3):
                            r = df.iloc[i]; b=float(r.get('bid',0) or 0); a=float(r.get('ask',0) or 0)
                            m=(b+a)/2 if b>0 and a>0 else float(r.get('lastPrice',0) or 0)
                            return f"${r['strike']:.2f} mid ${m:.2f} IV {float(r.get('impliedVolatility',0) or 0)*100:.0f}%"
                        _lc1,_lc2,_lc3 = st.columns([3,1,1])
                        _sel_si3 = _lc1.selectbox("Strike",range(len(_cdf3)),index=_def3,format_func=_slbl3,
                                                   key=f"_sb_str_{_li}",label_visibility='collapsed')
                        _ssr3 = _cdf3.iloc[_sel_si3]
                        _sb3 = float(_ssr3.get('bid',0) or 0); _sa3 = float(_ssr3.get('ask',0) or 0)
                        _sm3 = (_sb3+_sa3)/2 if _sb3>0 and _sa3>0 else float(_ssr3.get('lastPrice',0) or 0)
                        _siv3 = float(_ssr3.get('impliedVolatility',0.3) or 0.3)
                        _lqty = _lc2.number_input("Qty",min_value=1,value=1,step=1,key=f"_sb_qty_{_li}",label_visibility='collapsed')
                        _lcst = _lc3.number_input("Cost",min_value=0.0,value=round(_sm3,2),step=0.01,key=f"_sb_cost_{_li}",label_visibility='collapsed')
                        if _sb_tmpl=='Custom':
                            _cc1,_cc2 = st.columns(2)
                            _ot3  = _cc1.selectbox("Type",['call','put'],key=f"_sb_cl_type_{_li}",label_visibility='collapsed')
                            _pos3 = _cc2.selectbox("Pos",['long','short'],key=f"_sb_cl_pos_{_li}",label_visibility='collapsed')
                            if st.button(f"Remove Leg {_li+1}",key=f"_sb_rm_cl_{_li}"):
                                st.session_state['_sb_custom_legs'].pop(_li); st.rerun()
                        _leg_cfg.append({'option_type':_ot3,'position':_pos3,'strike':float(_ssr3['strike']),
                                         'expiration':_exp3,'contracts':int(st.session_state.get(f"_sb_qty_{_li}",1)),
                                         'cost_basis':float(st.session_state.get(f"_sb_cost_{_li}",_sm3)),
                                         'iv_at_entry':_siv3,'label':_lbl3})
                    if _leg_cfg:
                        _net3 = sum(l['cost_basis']*l['contracts']*100*(1 if l['position']=='long' else -1) for l in _leg_cfg)
                        st.info(f"{'Net Credit: $' if _net3<0 else 'Net Debit: $'}{abs(_net3):,.2f}")
                    _auto_nm3 = f"{_sb_tmpl} — {st.session_state.get('_sb_ul_loaded','')}"
                    _strat_nm3 = st.text_input("Strategy Name",value=_auto_nm3,key="_sb_name")
                    _sc1,_sc2 = st.columns(2)
                    with _sc1:
                        if st.button("Add Strategy",key="_sb_submit",type="primary"):
                            if _leg_cfg:
                                st.session_state['options_positions'].append({
                                    'type':'strategy','name':_strat_nm3,
                                    'underlying':st.session_state.get('_sb_ul_loaded',''),
                                    'status':'open','close_method':None,'close_price':None,'close_date':None,
                                    'legs':_leg_cfg})
                                save_options_positions(st.session_state['options_positions'])
                                for k in ['_sb_exps','_sb_ul_loaded','_sb_near_chain','_sb_far_chain',
                                          '_sb_calls_chain','_sb_puts_chain','_sb_single_exp_loaded',
                                          '_sb_near_exp_loaded','_sb_far_exp_loaded','_sb_custom_legs']:
                                    st.session_state.pop(k,None)
                                st.session_state['_sb_show'] = False; st.rerun()
                    with _sc2:
                        if st.button("Cancel",key="_sb_cancel"): st.session_state['_sb_show']=False; st.rerun()

    # ── Single-leg table ──────────────────────────────────────────────────────
    if _singles:
        section_header("Single-Leg Positions")
        with st.spinner("Fetching live option prices…"):
            for _,p in _singles:
                if p['status']=='open':
                    p['_live_price'],p['_live_iv'] = _live_option(p['underlying'],p['option_type'],p['strike'],p['expiration'],p.get('iv_at_entry',0.3))
        _sl_rows = []
        for _sidx,p in _singles:
            _exp_dt = datetime.strptime(p['expiration'],'%Y-%m-%d'); _dte = (_exp_dt-datetime.now()).days
            _ct = p['cost_basis']*p['contracts']*100
            if p['status']=='closed':
                _cm=p.get('close_method',''); _cp=float(p.get('close_price') or 0)
                _pnl = (-_ct if _cm=='expired_worthless' else _cp*p['contracts']*100-_ct)
                _pct = _pnl/_ct*100 if _ct else 0; _delta=_theta=None; _iv_d=p.get('iv_at_entry',0.3)
            else:
                _cp=p.get('_live_price'); _iv_d=p.get('_live_iv',p.get('iv_at_entry',0.3))
                if _cp is not None:
                    _pnl = _cp*p['contracts']*100-_ct; _pct = _pnl/_ct*100 if _ct else 0
                    _total_opt_val += _cp*p['contracts']*100; _total_opt_pnl += _pnl
                else: _pnl=_pct=None
                _ulp = _ul_prices.get(p['underlying'],0); _T = max(_dte/365,0.001)
                _delta=_theta=None
                if _ulp>0:
                    try:
                        _g = _oa.calculate_greeks(S=_ulp,K=p['strike'],T=_T,sigma=_iv_d,option_type=p['option_type'],r=0.045)
                        _delta=_g.get('delta'); _theta=_g.get('theta')
                    except Exception: pass
            _sl_rows.append({'_idx':_sidx,'Ul':p['underlying'],'Type':p['option_type'].upper(),
                             'Strike':f"${p['strike']:.2f}",'Expiry':p['expiration'],
                             'DTE':str(_dte) if p['status']=='open' else '—',
                             'Qty':p['contracts'],'Cost':f"${p['cost_basis']:.2f}",
                             'Cur':f"${_cp:.2f}" if _cp is not None else 'N/A',
                             'P&L':f"${_pnl:,.2f}" if _pnl is not None else 'N/A',
                             'P&L%':f"{_pct:+.1f}%" if _pct is not None else 'N/A',
                             'Δ':f"{_delta:.3f}" if _delta else '—',
                             'Θ':f"{_theta:.3f}" if _theta else '—',
                             'IV':f"{_iv_d*100:.1f}%",
                             'Status':'Open' if p['status']=='open' else 'Closed'})
        flex_table(
            pd.DataFrame(_sl_rows),
            columns=[
                {"key": "Ul",     "label": "Underlying", "width": "9%",  "align": "left"},
                {"key": "Type",   "label": "Type",       "width": "7%",  "align": "center"},
                {"key": "Strike", "label": "Strike",     "width": "8%",  "align": "right"},
                {"key": "Expiry", "label": "Expiry",     "width": "10%", "align": "right"},
                {"key": "DTE",    "label": "DTE",        "width": "6%",  "align": "right"},
                {"key": "Qty",    "label": "Qty",        "width": "5%",  "align": "right"},
                {"key": "Cost",   "label": "Cost",       "width": "8%",  "align": "right"},
                {"key": "Cur",    "label": "Last",       "width": "8%",  "align": "right"},
                {"key": "P&L",    "label": "P&L",        "width": "10%", "align": "right"},
                {"key": "P&L%",   "label": "P&L %",      "width": "8%",  "align": "right"},
                {"key": "Δ",      "label": "Delta",      "width": "7%",  "align": "right"},
                {"key": "Θ",      "label": "Theta",      "width": "7%",  "align": "right"},
                {"key": "IV",     "label": "IV",         "width": "7%",  "align": "right"},
                {"key": "Status", "label": "Status",     "width": "8%",  "align": "center"},
            ],
            key="singles",
            row_height=40,
        )
        with st.expander("Manage Single-Leg Positions"):
            for r3 in _sl_rows:
                _si2=r3['_idx']; _p3=_opts[_si2]
                _lbl4=(f"{_p3['option_type'].upper()} {_p3['underlying']} ${_p3['strike']:.0f} exp {_p3['expiration']}")
                if _p3['status']=='open':
                    st.markdown(f"**{_lbl4}**")
                    _clm4=st.selectbox("Close method",['sold','expired_worthless','exercised'],
                        format_func=lambda x:{'sold':'Sold','expired_worthless':'Expired worthless','exercised':'Exercised'}[x],
                        key=f"sl_cm_{_si2}")
                    _clp4=None
                    if _clm4=='sold': _clp4=st.number_input("Sale price ($/sh)",min_value=0.0,step=0.01,key=f"sl_cp_{_si2}")
                    elif _clm4=='exercised':
                        _ulpx5=_ul_prices.get(_p3['underlying'],0)
                        _intr3=(max(_ulpx5-_p3['strike'],0) if _p3['option_type']=='call' else max(_p3['strike']-_ulpx5,0))
                        _clp4=st.number_input("Net value at exercise",min_value=0.0,value=round(_intr3,2),step=0.01,key=f"sl_ep_{_si2}")
                    if st.button("Confirm Close",key=f"sl_cls_{_si2}"):
                        st.session_state['options_positions'][_si2].update({'status':'closed','close_method':_clm4,'close_price':_clp4,'close_date':datetime.now().strftime('%Y-%m-%d')})
                        save_options_positions(st.session_state['options_positions']); st.rerun()
                else:
                    _cst5=_p3['cost_basis']*_p3['contracts']*100; _pr5=float(_p3.get('close_price') or 0)
                    _pnl5=(-_cst5 if _p3.get('close_method')=='expired_worthless' else _pr5*_p3['contracts']*100-_cst5)
                    _clr5=GAIN if _pnl5>=0 else LOSS
                    st.markdown(f"⭕ **{_lbl4}** — <span style='color:{_clr5};'>${_pnl5:,.2f}</span>",unsafe_allow_html=True)
                    if st.button("Remove",key=f"sl_rm_{_si2}"):
                        st.session_state['options_positions'].pop(_si2)
                        save_options_positions(st.session_state['options_positions']); st.rerun()
                st.divider()

    # ── Strategies ────────────────────────────────────────────────────────────
    if _strategies:
        section_header("Multi-Leg Strategies")
        with st.spinner("Fetching strategy leg prices…"):
            for _,p in _strategies:
                if p['status']!='open': continue
                for leg in p.get('legs',[]):
                    leg['_live_price'],leg['_live_iv'] = _live_option(p['underlying'],leg['option_type'],leg['strike'],leg['expiration'],leg.get('iv_at_entry',0.3))
        for _stidx,p in _strategies:
            _legs6=p.get('legs',[]); _is_open6=p['status']=='open'
            _net_entry6=sum(l['cost_basis']*l['contracts']*100*(1 if l['position']=='long' else -1) for l in _legs6)
            _nd6=_nt6=_nv6=0.0
            if _is_open6:
                _nc6=0.0; _ok6=True
                for leg in _legs6:
                    _lp6=leg.get('_live_price')
                    if _lp6 is None: _ok6=False; continue
                    _s6=1 if leg['position']=='long' else -1; _nc6+=_lp6*leg['contracts']*100*_s6
                    _dte6=max((datetime.strptime(leg['expiration'],'%Y-%m-%d')-datetime.now()).days,0)
                    _ulp6=_ul_prices.get(p['underlying'],0); _iv6=leg.get('_live_iv',leg.get('iv_at_entry',0.3))
                    if _ulp6>0:
                        try:
                            _g6=_oa.calculate_greeks(S=_ulp6,K=leg['strike'],T=max(_dte6/365,0.001),sigma=_iv6,option_type=leg['option_type'],r=0.045)
                            _nd6+=(_g6.get('delta') or 0)*leg['contracts']*_s6
                            _nt6+=(_g6.get('theta') or 0)*leg['contracts']*_s6
                            _nv6+=(_g6.get('vega')  or 0)*leg['contracts']*_s6
                        except Exception: pass
                _sp6=_nc6-_net_entry6
                if _ok6: _total_opt_val+=_nc6; _total_opt_pnl+=_sp6
                _hdr6=(f"**{p['name']}** | {len(_legs6)} legs | "
                       f"{'Credit' if _net_entry6<0 else 'Debit'} ${abs(_net_entry6):,.2f} | "
                       f"P&L ${_sp6:+,.2f} | Δ {_nd6:+.3f} Θ {_nt6:+.3f}")
            else:
                _cm7=p.get('close_method',''); _cp7=float(p.get('close_price') or 0)
                _pnl7=(_cp7-_net_entry6) if _cm7!='expired_worthless' else -abs(_net_entry6)
                _hdr6=f"**{p['name']}** [Closed] | Realised ${_pnl7:+,.2f}"
            with st.expander(_hdr6,expanded=False):
                _leg_rows7=[{'Label':l.get('label',''),'Type':l['option_type'].upper(),
                    'Pos':l['position'].upper(),'Strike':f"${l['strike']:.2f}",'Expiry':l['expiration'],
                    'Qty':l['contracts'],'Cost':f"${l['cost_basis']:.2f}",
                    'Cur':f"${l.get('_live_price'):.2f}" if l.get('_live_price') else 'N/A',
                    'IV':f"{l.get('_live_iv',l.get('iv_at_entry',0))*100:.1f}%"} for l in _legs6]
                flex_table(
                    pd.DataFrame(_leg_rows7),
                    columns=[
                        {"key": "Label",  "label": "Leg",    "width": "22%", "align": "left"},
                        {"key": "Type",   "label": "Type",   "width": "8%",  "align": "center"},
                        {"key": "Pos",    "label": "Pos",    "width": "8%",  "align": "center"},
                        {"key": "Strike", "label": "Strike", "width": "12%", "align": "right"},
                        {"key": "Expiry", "label": "Expiry", "width": "14%", "align": "left"},
                        {"key": "Qty",    "label": "Qty",    "width": "7%",  "align": "right"},
                        {"key": "Cost",   "label": "Cost",   "width": "12%", "align": "right"},
                        {"key": "Cur",    "label": "Last",   "width": "10%", "align": "right"},
                        {"key": "IV",     "label": "IV",     "width": "7%",  "align": "right"},
                    ],
                    key=f"legs_{_stidx}",
                )
                if _is_open6:
                    _scm7=st.selectbox("Close method",['bought_to_close','expired_worthless','partial_close'],
                        format_func=lambda x:{'bought_to_close':'Bought to close','expired_worthless':'All expired worthless','partial_close':'Partial close'}[x],
                        key=f"strat_cm_{_stidx}")
                    _scp7=None
                    if _scm7 in ('bought_to_close','partial_close'):
                        _scp7=st.number_input("Net amount received",step=1.0,key=f"strat_cp_{_stidx}")
                    if st.button("Confirm Close",key=f"strat_cls_{_stidx}"):
                        st.session_state['options_positions'][_stidx].update({'status':'closed','close_method':_scm7,'close_price':_scp7,'close_date':datetime.now().strftime('%Y-%m-%d')})
                        save_options_positions(st.session_state['options_positions']); st.rerun()
                else:
                    if st.button("Remove",key=f"strat_rm_{_stidx}"):
                        st.session_state['options_positions'].pop(_stidx)
                        save_options_positions(st.session_state['options_positions']); st.rerun()

    if not _singles and not _strategies:
        st.info("No options positions. Use the sidebar to add single-leg positions, or click **Add Multi-Leg Strategy** above.")

    # ── Options summary ───────────────────────────────────────────────────────
    if _singles or _strategies:
        section_header("Options Summary")
        _so2=sum(1 for _,p in _singles if p['status']=='open')
        _sc2=sum(1 for _,p in _singles if p['status']=='closed')
        _sto2=sum(1 for _,p in _strategies if p['status']=='open')
        _stc2=sum(1 for _,p in _strategies if p['status']=='closed')
        om1,om2,om3,om4 = st.columns(4)
        om1.metric("Open Options Value",   f"${_total_opt_val:,.2f}")
        om2.metric("Unrealised P&L",       f"${_total_opt_pnl:,.2f}")
        om3.metric("Open Positions",       f"{_so2} single · {_sto2} strat")
        om4.metric("Closed",               f"{_sc2} single · {_stc2} strat")
        if (_sc2+_stc2)>0:
            if st.button("Clear All Closed Options"):
                st.session_state['options_positions']=[p for p in _opts if p['status']=='open']
                save_options_positions(st.session_state['options_positions']); st.rerun()

    # ── Combined portfolio totals ─────────────────────────────────────────────
    if (_singles or _strategies) and summary is not None:
        section_header("Combined Portfolio")
        _cv1,_cv2,_cv3 = st.columns(3)
        _cv1.metric("Equities", f"${summary['total_value']:,.2f}")
        _cv2.metric("Options (Open)", f"${_total_opt_val:,.2f}", f"P&L ${_total_opt_pnl:+,.2f}")
        _cv3.metric("Combined Value", f"${summary['total_value']+_total_opt_val:,.2f}",
                    f"Total P&L ${summary['total_pnl']+_total_opt_pnl:+,.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRADE SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tab_signals:
    section_header("Algorithm-Scored Strategy Recommendations")
    # Controls row
    ts_c1,ts_c2,ts_c3,ts_c4,ts_c5 = st.columns([2,2,2,2,1])
    ts_ticker  = ts_c1.text_input("Ticker",value="SPY",key="_ts_ticker").upper()
    ts_view    = ts_c2.selectbox("Directional View",
        ['Let Algorithm Decide','Bullish','Bearish','Neutral','Volatile / Expecting Big Move'],key="_ts_view")
    ts_acct    = ts_c3.number_input("Account ($)",min_value=5000,max_value=5_000_000,value=50000,step=5000,key="_ts_acct")
    ts_risk    = ts_c4.select_slider("Risk",options=[1,2,3],value=2,
        format_func=lambda x:{1:"Conservative",2:"Moderate",3:"Aggressive"}[x],key="_ts_risk")
    ts_dte     = ts_c5.selectbox("DTE",[14,21,30,45,60,90,120,180,365],index=2,key="_ts_dte",
        format_func=lambda d:f"{d}d")
    if st.button("Generate Suggestions",type="primary",key="_ts_go"):
        st.session_state['_ts_loaded'] = True

    if not st.session_state.get('_ts_loaded',False):
        st.info("Configure inputs above and click **Generate Suggestions**.")
        st.stop()

    # Load regime + chain
    try:
        _ts_regime,_,_ = _get_market_context()
    except Exception:
        _ts_regime = "Unknown"
    rc2 = regime_color(_ts_regime)

    with st.spinner(f"Loading {ts_ticker} options…"):
        _ts_exps = options_loader.get_options_expirations(ts_ticker)
        if not _ts_exps:
            st.error(f"No options data for {ts_ticker}"); st.stop()
        _now2 = datetime.now(); _best_exp2 = _ts_exps[0]; _best_diff2=9999; _actual_dte2=ts_dte
        for exp in _ts_exps:
            try:
                _d = (datetime.strptime(exp,'%Y-%m-%d')-_now2).days
                if abs(_d-ts_dte)<_best_diff2: _best_diff2=abs(_d-ts_dte); _best_exp2=exp; _actual_dte2=_d
            except ValueError: continue
        _ts_calls,_ts_puts,_ts_spot = options_loader.get_options_chain(ts_ticker,_best_exp2)

    if _ts_calls.empty or _ts_puts.empty:
        st.error("Could not fetch options chain."); st.stop()

    _ts_T   = max(_actual_dte2/365,0.003)
    _ts_atm_iv = float(_ts_calls['impliedVolatility'].iloc[len(_ts_calls)//2]) if not _ts_calls.empty else 0.25
    _ts_atm_iv = max(_ts_atm_iv,0.05)
    _ts_iv_rank = _compute_iv_rank(ts_ticker,_ts_atm_iv)
    try:
        _ts_hpx = yf.Ticker(ts_ticker).history(period='3mo')
        _ts_hv20 = float(np.log(_ts_hpx['Close']/_ts_hpx['Close'].shift(1)).dropna().rolling(20).std().iloc[-1]*np.sqrt(252))
    except Exception:
        _ts_hv20 = _ts_atm_iv*0.85

    # Context strip
    ctx1,ctx2,ctx3 = st.columns([2,2,3])
    with ctx1:
        st.markdown(f'<div style="background:{rc2}22;border:1px solid {rc2}55;border-radius:8px;'
                    f'padding:12px 16px;text-align:center;">'
                    f'<div style="font-size:10px;color:{SUBTLE};text-transform:uppercase;letter-spacing:0.1em;">Regime</div>'
                    f'<div style="font-size:20px;font-weight:600;color:{rc2};margin-top:4px;">{_ts_regime}</div>'
                    f'</div>',unsafe_allow_html=True)
    with ctx2:
        _ivc = LOSS if _ts_iv_rank>=0.70 else ACCENT if _ts_iv_rank<=0.35 else AMBER
        _ivl = "High — Sell" if _ts_iv_rank>=0.70 else "Low — Buy" if _ts_iv_rank<=0.35 else "Moderate"
        st.markdown(f'<div style="background:{_ivc}15;border:1px solid {_ivc}55;border-radius:8px;'
                    f'padding:12px 16px;text-align:center;">'
                    f'<div style="font-size:10px;color:{SUBTLE};text-transform:uppercase;letter-spacing:0.1em;">IV Rank</div>'
                    f'<div style="font-size:20px;font-weight:600;color:{_ivc};margin-top:4px;">{_ts_iv_rank:.0%}</div>'
                    f'<div style="font-size:10px;color:{_ivc};margin-top:2px;">{_ivl}</div>'
                    f'</div>',unsafe_allow_html=True)
    with ctx3:
        m1,m2,m3 = st.columns(3)
        m1.metric(f"{ts_ticker}",f"${_ts_spot:.2f}")
        m2.metric("ATM IV",f"{_ts_atm_iv*100:.1f}%")
        m3.metric("20d HV",f"{_ts_hv20*100:.1f}%")

    st.divider()

    # Infer view
    _ts_inferred = None; _ts_reasons = []
    _view_final  = ts_view
    if ts_view == 'Let Algorithm Decide':
        if _ts_regime in ('High Volatility','High Vol'):  _ts_inferred='Volatile / Expecting Big Move'; _ts_reasons.append(f"Regime **{_ts_regime}**")
        elif _ts_regime in ('Recession','Stagflation'):   _ts_inferred='Bearish';  _ts_reasons.append(f"Regime **{_ts_regime}** → bearish tilt")
        elif _ts_regime in ('Low Vol','Risk-On','Mean Reversion','Caution','Uncertain'): _ts_inferred='Neutral'; _ts_reasons.append(f"Regime **{_ts_regime}** → neutral")
        else:
            try:
                _ppx = yf.Ticker(ts_ticker).history(period='3mo')['Close']
                _r20 = float(_ppx.iloc[-1]/_ppx.iloc[-20]-1) if len(_ppx)>=20 else 0
                _ma50 = float(_ppx.rolling(50).mean().iloc[-1]) if len(_ppx)>=50 else float(_ppx.mean())
                _above = _ppx.iloc[-1] > _ma50
            except Exception: _r20=0; _above=True
            if _r20>0.02 and _above: _ts_inferred='Bullish'; _ts_reasons.append(f"20d return {_r20:+.1%}, above 50MA")
            elif _r20<-0.02 and not _above: _ts_inferred='Bearish'; _ts_reasons.append(f"20d return {_r20:+.1%}, below 50MA")
            else: _ts_inferred='Neutral'; _ts_reasons.append("Mixed signals — Neutral")
        if _ts_iv_rank>=0.75 and _ts_inferred in ('Bullish','Bearish'):
            _ts_reasons.append(f"IV rank {_ts_iv_rank:.0%} — prefer spreads over naked options")
        elif _ts_iv_rank<=0.30: _ts_reasons.append(f"IV rank {_ts_iv_rank:.0%} — cheap options favour buying")
        _view_final = _ts_inferred

    if _ts_inferred:
        _vc = ACCENT if _ts_inferred=='Bullish' else LOSS if _ts_inferred=='Bearish' else AMBER if 'Volatile' in _ts_inferred else ACCENT
        st.markdown(f'<div style="background:{_vc}12;border:1px solid {_vc}44;border-radius:8px;padding:12px 16px;margin-bottom:12px;">'
                    f'<span style="font-size:10px;color:{SUBTLE};text-transform:uppercase;letter-spacing:0.1em;">Inferred View</span>'
                    f'<div style="font-size:18px;font-weight:600;color:{_vc};margin:4px 0;">{_ts_inferred}</div>'
                    + "".join(f"<div style='font-size:0.82rem;color:#cccccc;'>• {r}</div>" for r in _ts_reasons)
                    + '</div>',unsafe_allow_html=True)

    # Score strategies
    _scored = sorted([(n, *_score_strategy(n,_view_final,_ts_iv_rank,_ts_regime,ts_risk))
                      for n in _STRATEGY_PROFILES], key=lambda x:x[1], reverse=True)
    _top3 = _scored[:3]

    st.markdown(f"**Top 3 Strategies** — {ts_ticker} · View: {_view_final} · Regime: {_ts_regime} · IV Rank: {_ts_iv_rank:.0%} · DTE: {_actual_dte2}d")

    for rank,(strat_name,score,reasons) in enumerate(_top3,1):
        _rclrs = {1:AMBER,2:'#9ca3af',3:'#cd7f32'}; _rc2=_rclrs.get(rank,DIM)
        st.markdown(f'<div style="background:{CARD};border:1px solid {_rc2}55;border-radius:10px;'
                    f'padding:4px 16px 2px;margin-bottom:4px;">'
                    f' <span style="font-size:17px;font-weight:700;color:{_rc2};">#{rank}</span>'
                    f' <span style="font-size:17px;font-weight:700;color:{FG};margin-left:8px;">{strat_name}</span>'
                    f'</div>',unsafe_allow_html=True)
        st.progress(int(score),text=f"Score: {score:.0f}/100")
        with st.expander("Strategy Details",expanded=(rank==1)):
            st.markdown("**Why selected:**")
            for r in reasons: st.markdown(f"- {r}")
            builder = StrategyBuilder()
            ok = builder.load_template(strat_name,_ts_spot,_ts_calls,_ts_puts)
            if not ok or not builder.legs:
                st.warning(f"Could not build {strat_name} from current chain."); continue
            _summ = builder.get_strategy_summary(_ts_spot,_ts_T)
            if 'error' in _summ: st.warning(_summ['error']); continue
            try: _pop=probability_of_profit(_ts_spot,builder,_ts_atm_iv,_RF,_ts_T); _ev=expected_value(_ts_spot,builder,_ts_atm_iv,_RF,_ts_T)
            except Exception: _pop=0.5; _ev=0.0
            _legs_df = builder.get_legs_dataframe()
            flex_table(
                _legs_df,
                columns=[
                    {"key": "#",         "label": "#",        "width": "6%",  "align": "right", "numeric": True},
                    {"key": "Type",      "label": "Type",     "width": "9%",  "align": "center"},
                    {"key": "Position",  "label": "Pos",      "width": "9%",  "align": "center"},
                    {"key": "Strike",    "label": "Strike",   "width": "14%", "align": "right",
                     "fmt": lambda v: f"${v:.2f}" if v == v else "—"},
                    {"key": "Contracts", "label": "Qty",      "width": "8%",  "align": "right"},
                    {"key": "Premium",   "label": "Premium",  "width": "14%", "align": "right",
                     "fmt": lambda v: f"${v:.2f}" if v == v else "—"},
                    {"key": "IV",        "label": "IV",       "width": "12%", "align": "right",
                     "fmt": lambda v: f"{v:.2%}" if v == v else "—"},
                    {"key": "Cost",      "label": "Cost",     "width": "14%", "align": "right",
                     "fmt": lambda v: f"${v:+,.2f}" if v == v else "—"},
                ],
                key=f"opt_legs_{rank}",
            )
            _cost2=_summ['initial_cost']; _mp2=_summ['max_profit']; _ml2=_summ['max_loss']
            km1,km2,km3,km4,km5 = st.columns(5)
            km1.metric("Net",f"{'Credit' if _cost2<0 else 'Debit'} ${abs(_cost2):,.0f}")
            km2.metric("Max Profit",f"${_mp2:,.0f}" if _mp2<1e9 else "Unlimited")
            km3.metric("Max Loss",  f"${_ml2:,.0f}" if _ml2>-1e9 else "Unlimited")
            km4.metric("PoP",f"{_pop:.1%}"); km5.metric("EV",f"${_ev:,.0f}")
            _pr4,_pnl4 = builder.calculate_payoff(_ts_spot)
            fig_pay = go.Figure()
            fig_pay.add_trace(go.Scatter(x=_pr4,y=_pnl4,fill='tozeroy',fillcolor='rgba(34,211,238,0.08)',
                                         line=dict(color=ACCENT,width=2),name='P&L at Expiry'))
            fig_pay.add_hline(y=0,line_color=SUBTLE,line_dash='dash',line_width=1)
            fig_pay.add_vline(x=_ts_spot,line_color='#4a9eff',line_dash='dot',line_width=1.5,
                              annotation_text=f"${_ts_spot:.2f}",annotation_font_color='#4a9eff')
            for be in _summ['breakevens']:
                fig_pay.add_vline(x=be,line_color=AMBER,line_dash='dot',line_width=1,
                                  annotation_text=f"BE ${be:.0f}",annotation_font_color=AMBER)
            fig_pay.update_layout(**carbon_plotly_layout(height=300,
                title=f"{strat_name} Payoff at Expiry ({_actual_dte2}d)",
                xaxis_title="Price",yaxis_title="P&L ($)",hovermode="x unified"))
            st.plotly_chart(fig_pay,use_container_width=True)
            _sz = recommender.get_position_sizing_guide(_ts_regime,ts_acct)
            st.caption(f"Sizing ({_ts_regime}): max risk ${_sz['max_dollar_risk']:,.0f} "
                       f"({_sz['max_risk_per_trade']*100:.1f}% of account) · {_sz['notes']}")
        st.markdown("")

    with st.expander("Full Strategy Rankings"):
        _rdf = pd.DataFrame([{'Rank':i+1,'Strategy':n,'Score':s,
            'Bias':_STRATEGY_PROFILES[n]['bias'],'Premium':_STRATEGY_PROFILES[n]['premium_type'],
            'Risk Defined':'Yes' if _STRATEGY_PROFILES[n]['risk_defined'] else 'No'}
            for i,(n,s,_) in enumerate(_scored)])
        flex_table(
            _rdf,
            columns=[
                {"key": "Rank",         "label": "Rank",         "width": "8%",  "align": "right",  "numeric": True},
                {"key": "Strategy",     "label": "Strategy",     "width": "27%", "align": "left"},
                {"key": "Score",        "label": "Score",        "width": "13%", "align": "right",  "numeric": True,
                 "color_scale": "rg",   "fmt": lambda v: f"{v:.1f}"},
                {"key": "Bias",         "label": "Bias",         "width": "17%", "align": "left"},
                {"key": "Premium",      "label": "Premium Type", "width": "20%", "align": "left"},
                {"key": "Risk Defined", "label": "Risk Defined", "width": "15%", "align": "center"},
            ],
            key="rankings",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TRADE JOURNAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_journal:
    section_header("Trade Journal & Portfolio History")

    # ── Auto-snapshot today's portfolio value ─────────────────────────────────
    if positions_df is not None and not positions_df.empty and current_prices:
        try:
            _snap_val = sum(
                float(row.get("shares", 0)) * current_prices.get(row.get("ticker", ""), 0)
                for _, row in positions_df.iterrows()
            )
            _snap_spy = current_prices.get("SPY", 0)
            if _snap_spy == 0:
                try:
                    import yfinance as _yf2
                    _snap_spy = float(_yf2.Ticker("SPY").fast_info.get("last_price", 0))
                except Exception:
                    pass
            _snap_regime = "Unknown"
            try:
                _snap_regime, _, _ = _get_market_context()
            except Exception:
                pass
            save_daily_snapshot(_snap_val, _snap_spy, positions_df, regime=_snap_regime)
        except Exception:
            pass

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    jt_equity, jt_log, jt_add = st.tabs(["Equity Curve", "Trade Log", "Log a Trade"])

    # ── Equity Curve ──────────────────────────────────────────────────────────
    with jt_equity:
        _history = get_portfolio_history(days=365)
        if _history.empty:
            st.info("No daily snapshots yet. Come back after the app has been open for a few days — it auto-saves a snapshot each visit.")
        else:
            _eq = compute_equity_curve(_history)
            if not _eq.empty:
                # Summary metrics
                _pnorm_last = float(_eq["portfolio_norm"].iloc[-1])
                _bnorm_last = float(_eq["benchmark_norm"].iloc[-1])
                _alpha_last  = float(_eq["alpha"].iloc[-1])
                _port_ret    = (_pnorm_last / 100 - 1) * 100
                _bench_ret   = (_bnorm_last / 100 - 1) * 100

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Portfolio Return",  f"{_port_ret:+.2f}%")
                mc2.metric("SPY Return",        f"{_bench_ret:+.2f}%")
                mc3.metric("Alpha (vs SPY)",    f"{_alpha_last:+.2f}%",
                           delta_color="normal")
                mc4.metric("Days Tracked",      len(_eq))

                # Equity curve chart
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=_eq.index, y=_eq["portfolio_norm"],
                    name="Portfolio", line=dict(color=ACCENT, width=2),
                    fill="tozeroy", fillcolor="rgba(34,211,238,0.06)",
                ))
                fig_eq.add_trace(go.Scatter(
                    x=_eq.index, y=_eq["benchmark_norm"],
                    name="SPY", line=dict(color=DIM, width=1.5, dash="dash"),
                ))
                fig_eq.add_hline(y=100, line_color=BORDER, line_dash="dot", line_width=1)
                fig_eq.update_layout(**carbon_plotly_layout(
                    height=380,
                    title="Portfolio vs SPY (Normalised to 100)",
                    hovermode="x unified",
                ))
                fig_eq.update_layout(yaxis=dict(title="Indexed (Base=100)"))
                st.plotly_chart(fig_eq, use_container_width=True)

                # Alpha area chart
                fig_al = go.Figure()
                _alpha_pos = _eq["alpha"].clip(lower=0)
                _alpha_neg = _eq["alpha"].clip(upper=0)
                fig_al.add_trace(go.Scatter(
                    x=_eq.index, y=_alpha_pos, name="Outperformance",
                    fill="tozeroy", fillcolor="rgba(34,197,94,0.18)",
                    line=dict(color=GAIN, width=1),
                ))
                fig_al.add_trace(go.Scatter(
                    x=_eq.index, y=_alpha_neg, name="Underperformance",
                    fill="tozeroy", fillcolor="rgba(239,68,68,0.18)",
                    line=dict(color=LOSS, width=1),
                ))
                fig_al.add_hline(y=0, line_color=BORDER, line_width=1)
                fig_al.update_layout(**carbon_plotly_layout(
                    height=200, title="Alpha vs SPY (Portfolio − Benchmark)",
                ))
                fig_al.update_layout(yaxis=dict(title="Alpha (pp)"))
                st.plotly_chart(fig_al, use_container_width=True)

    # ── Trade Log ─────────────────────────────────────────────────────────────
    with jt_log:
        _jl_col1, _jl_col2, _jl_col3 = st.columns([2, 2, 1])
        _jl_ticker_filter = _jl_col1.text_input("Filter by Ticker", "", key="_jl_tf").upper().strip()
        _jl_start = _jl_col2.date_input("From", value=None, key="_jl_sd")
        _jl_end   = _jl_col3.date_input("To",   value=None, key="_jl_ed")

        _trades = get_trades(
            ticker=_jl_ticker_filter or None,
            start_date=str(_jl_start) if _jl_start else None,
            end_date=str(_jl_end)     if _jl_end   else None,
        )

        if _trades.empty:
            st.info("No trades logged yet. Use the **Log a Trade** tab to record trades.")
        else:
            _trades = compute_trade_pnl(_trades)

            # Summary P&L
            _realised = _trades["realised_pnl"].dropna().sum()
            _n_trades = len(_trades)
            _winners  = (_trades["realised_pnl"] > 0).sum()
            _losers   = (_trades["realised_pnl"] < 0).sum()
            _wr       = _winners / max(_winners + _losers, 1) * 100

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total Trades",    _n_trades)
            sc2.metric("Realised P&L",    f"${_realised:+,.2f}",
                       delta_color="normal")
            sc3.metric("Win Rate",        f"{_wr:.1f}%")
            sc4.metric("Wins / Losses",   f"{_winners} / {_losers}")

            # Table
            _display = _trades[[
                "id","date","ticker","action","shares","price",
                "total_value","cost_basis","realised_pnl","realised_pnl_pct",
                "regime","source","notes"
            ]].copy()
            _display.columns = [
                "ID","Date","Ticker","Action","Shares","Price",
                "Total","Cost Basis","Real. P&L","P&L %",
                "Regime","Source","Notes"
            ]
            flex_table(
                _display,
                columns=[
                    {"key": "ID",        "label": "ID",       "width": "4%",  "align": "right",  "numeric": True},
                    {"key": "Date",      "label": "Date",     "width": "9%",  "align": "left"},
                    {"key": "Ticker",    "label": "Ticker",   "width": "7%",  "align": "left"},
                    {"key": "Action",    "label": "Action",   "width": "10%", "align": "center"},
                    {"key": "Shares",    "label": "Shares",   "width": "7%",  "align": "right",  "numeric": True,
                     "fmt": lambda v: f"{v:.4g}" if pd.notna(v) else "—"},
                    {"key": "Price",     "label": "Price",    "width": "8%",  "align": "right",  "numeric": True,
                     "fmt": lambda v: f"${v:.2f}" if pd.notna(v) else "—"},
                    {"key": "Total",     "label": "Total $",  "width": "9%",  "align": "right",  "numeric": True,
                     "fmt": lambda v: f"${v:,.2f}" if pd.notna(v) else "—"},
                    {"key": "Real. P&L", "label": "Real. P&L","width": "9%",  "align": "right",  "numeric": True,
                     "color_scale": "rg",
                     "fmt": lambda v: f"${v:+,.2f}" if pd.notna(v) else "—"},
                    {"key": "P&L %",     "label": "P&L %",   "width": "7%",  "align": "right",  "numeric": True,
                     "color_scale": "rg",
                     "fmt": lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"},
                    {"key": "Regime",    "label": "Regime",   "width": "10%", "align": "left"},
                    {"key": "Source",    "label": "Source",   "width": "7%",  "align": "center"},
                    {"key": "Notes",     "label": "Notes",    "width": "13%", "align": "left"},
                ],
                key="trade_log_tbl",
            )

            # Delete trade
            with st.expander("Delete a Trade"):
                _del_id = st.number_input("Trade ID to delete", min_value=1, step=1, key="_del_tid")
                if st.button("Delete", type="secondary", key="_del_btn"):
                    if delete_trade(int(_del_id)):
                        st.success(f"Trade #{_del_id} deleted.")
                        st.rerun()
                    else:
                        st.error("Could not delete trade.")

    # ── Log a Trade ───────────────────────────────────────────────────────────
    with jt_add:
        st.markdown("#### Manually Record a Trade")
        with st.form("log_trade_form", clear_on_submit=True):
            _fa, _fb, _fc = st.columns(3)
            _lt_ticker  = _fa.text_input("Ticker", placeholder="AAPL").upper().strip()
            _lt_action  = _fb.selectbox("Action", ["BUY", "SELL", "REBALANCE", "OPTION_BUY", "OPTION_SELL"])
            _lt_source  = _fc.selectbox("Source", ["manual", "paper", "imported"])

            _fd, _fe, _ff = st.columns(3)
            _lt_shares  = _fd.number_input("Shares / Contracts", min_value=0.0, step=1.0)
            _lt_price   = _fe.number_input("Price ($)", min_value=0.0, step=0.01)
            _lt_cb      = _ff.number_input("Cost Basis / Avg Price ($)", min_value=0.0, step=0.01,
                                            help="Used to compute realised P&L on sells")

            _fg, _fh = st.columns([2, 3])
            _lt_regime = _fg.text_input("Regime at trade", value="")
            _lt_notes  = _fh.text_input("Notes", placeholder="Optional rationale")

            # Options fields (optional)
            with st.expander("Options fields (leave blank for stock trades)"):
                _fo1, _fo2, _fo3 = st.columns(3)
                _lt_strategy  = _fo1.text_input("Strategy", placeholder="Long Call")
                _lt_opt_type  = _fo2.selectbox("Option Type", ["", "call", "put"])
                _lt_strike    = _fo3.number_input("Strike ($)", min_value=0.0, step=0.5)
                _lt_exp       = st.text_input("Expiration (YYYY-MM-DD)", "")

            _submitted = st.form_submit_button("Log Trade", type="primary")

        if _submitted:
            if not _lt_ticker:
                st.warning("Ticker is required.")
            elif _lt_shares <= 0 or _lt_price <= 0:
                st.warning("Shares and price must be > 0.")
            else:
                _new_id = log_trade(
                    ticker=_lt_ticker,
                    action=_lt_action,
                    shares=_lt_shares,
                    price=_lt_price,
                    cost_basis=_lt_cb,
                    regime=_lt_regime,
                    notes=_lt_notes,
                    source=_lt_source,
                    strategy=_lt_strategy,
                    option_type=_lt_opt_type,
                    strike=_lt_strike,
                    expiration=_lt_exp,
                )
                st.success(f"Trade logged (ID #{_new_id}): {_lt_action} {_lt_shares} {_lt_ticker} @ ${_lt_price:.2f}")
                st.rerun()
