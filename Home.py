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

# Distinct color palette for holdings — cyan-anchored shades
_HOLDING_COLORS = [
    '#22d3ee', '#67e8f9', '#a5f3fc', '#0891b2', '#06b6d4',
    '#38bdf8', '#7dd3fc', '#0e7490', '#155e75', '#cffafe',
    '#164e63', '#bae6fd', '#0284c7', '#93c5fd', '#0ea5e9',
]

# Options color family — amber / warm-orange shades
_OPT_COLORS = [
    '#f59e0b', '#fbbf24', '#fcd34d', '#f97316', '#fb923c',
    '#fdba74', '#d97706', '#b45309', '#a16207', '#ca8a04',
]

# Page config
st.set_page_config(
    page_title="Portfolio Position Tracker",
    page_icon="💼",
    layout="wide"
)
apply_dark_theme()

# ── Carbon bg · Cyan accent · Paper Trading typography ───────────────────
_H_BG     = "#111111"
_H_CARD   = "#1a1a1a"
_H_BORDER = "#2a2a2a"
_H_FG     = "#ffffff"
_H_DIM    = "#555555"
_H_SUBTLE = "#888888"
_H_ACCENT = "#22d3ee"   # Carbon gold
_H_GAIN   = "#22d3ee"   # positive P&L → gold
_H_LOSS   = "#fb7185"   # negative P&L → red

_HOME_CSS = f"""
/* ══ HOME — CARBON THEME (Theme #6) ══════════════════════════════════════
   #111111 bg · #1a1a1a card · #2a2a2a border · #22d3ee gold accent
   Helvetica Neue throughout · flat / no shadows / no rounded excesses
   ══════════════════════════════════════════════════════════════════════ */

/* ── Global & App background ────────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stMain"],
section[data-testid="stAppViewContainer"] > div:first-child {{
    background-color: {_H_BG} !important;
    box-shadow: none !important;
}}
.main .block-container,
[data-testid="stMainBlockContainer"] {{
    background-color: {_H_BG} !important;
    padding-top: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
    width: 100% !important;
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 14px !important;
    color: {_H_FG} !important;
    box-shadow: none !important;
    border: none !important;
}}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {_H_CARD} !important;
    border-right: 1px solid {_H_BORDER} !important;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
    color: {_H_SUBTLE} !important;
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.68rem !important;
    font-style: normal !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.40em !important;
    border-left: none !important;
    padding-left: 0 !important;
}}

/* ── Paper Trading: Palatino serif throughout ───────────────────────── */
* {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
}}

/* ── Page title (h1) — Paper Trading: large italic serif, accent ─────── */
h1 {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 2.2rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    color: {_H_ACCENT} !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}}

/* ── Subheaders (h2) — Paper Trading: italic serif, accent color ─────── */
[data-testid="stHeadingWithActionElements"] h2 {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 1.15rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    color: {_H_ACCENT} !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    border-left: none !important;
    padding-left: 0 !important;
    margin-top: 2rem !important;
    margin-bottom: 0.8rem !important;
}}

/* ── Section headers (h3) — italic serif, cyan ──────────────────────── */
[data-testid="stHeadingWithActionElements"] h3 {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 1.1rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    color: {_H_ACCENT} !important;
    border-left: none !important;
    padding-left: 0 !important;
    margin-top: 1.8rem !important;
    margin-bottom: 0.8rem !important;
}}

/* ── Inline markdown h4 headers — italic serif, cyan, slightly smaller ── */
[data-testid="stMarkdownContainer"] h4 {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    color: {_H_ACCENT} !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.5rem !important;
}}

/* ── Metric cards ────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {_H_CARD} !important;
    border: 1px solid {_H_BORDER} !important;
    border-radius: 8px !important;
    padding: 1.2rem 1.4rem 1rem !important;
    box-shadow: none !important;
}}
[data-testid="stMetricLabel"] > div {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    color: {_H_DIM} !important;
    font-size: 0.68rem !important;
    font-style: normal !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    font-weight: 500 !important;
}}
[data-testid="stMetricValue"] > div {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    color: {_H_FG} !important;
    font-weight: 300 !important;
    font-size: 1.75rem !important;
    letter-spacing: -0.01em !important;
}}
[data-testid="stMetricDelta"] svg {{ display: none !important; }}
[data-testid="stMetricDelta"] > div {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    color: {_H_ACCENT} !important;
}}

/* ── Tabs ─────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {_H_CARD} !important;
    border-radius: 8px !important;
    padding: 4px 6px !important;
    gap: 4px !important;
    border: 1px solid {_H_BORDER} !important;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent !important;
    color: {_H_DIM} !important;
    border-radius: 6px !important;
    border: none !important;
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    font-style: normal !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    padding: 0.38rem 1rem !important;
    transition: all 0.15s ease !important;
}}
.stTabs [aria-selected="true"] {{
    background-color: rgba(34, 211, 238, 0.10) !important;
    color: {_H_ACCENT} !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {{
    color: {_H_SUBTLE} !important;
    background-color: rgba(255,255,255,0.04) !important;
}}

/* ── Expanders ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border: none !important;
    border-top: 1px solid {_H_BORDER} !important;
    border-radius: 8px !important;
    background-color: transparent !important;
    overflow: hidden !important;
    box-shadow: none !important;
}}
[data-testid="stExpander"] > div:first-child {{
    background-color: transparent !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    font-style: normal !important;
    color: {_H_DIM} !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
}}
[data-testid="stExpander"] summary:hover span {{
    color: {_H_ACCENT} !important;
}}

/* ── Layout block wrappers — transparent, no implicit card box ─────────── */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] {{
    background-color: transparent !important;
    box-shadow: none !important;
}}
/* ── Explicit border containers only (st.container(border=True)) ────────── */
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    border: 1px solid {_H_BORDER} !important;
    border-radius: 8px !important;
    background-color: {_H_CARD} !important;
    box-shadow: none !important;
}}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton > button {{
    border-radius: 8px !important;
    border: 1px solid {_H_BORDER} !important;
    background-color: transparent !important;
    color: {_H_DIM} !important;
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    font-style: normal !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
    box-shadow: none !important;
}}
.stButton > button:hover {{
    border-color: {_H_ACCENT} !important;
    color: {_H_ACCENT} !important;
    background-color: transparent !important;
    box-shadow: none !important;
}}
.stButton > button[kind="primary"] {{
    background: {_H_ACCENT} !important;
    border: none !important;
    color: {_H_BG} !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: #38e0f8 !important;
    box-shadow: none !important;
    transform: none !important;
    color: {_H_BG} !important;
}}
.stButton > button[kind="secondary"] {{
    border: 1px solid {_H_ACCENT} !important;
    color: {_H_ACCENT} !important;
    background-color: transparent !important;
}}
.stButton > button[kind="secondary"]:hover {{
    background-color: rgba(34, 211, 238, 0.08) !important;
}}

/* ── Input widgets ────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stDateInput > div > div > input {{
    background-color: {_H_CARD} !important;
    border-color: {_H_BORDER} !important;
    border-radius: 8px !important;
    color: {_H_FG} !important;
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.82rem !important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {{
    border-color: {_H_ACCENT} !important;
    box-shadow: none !important;
}}
label[data-testid="stWidgetLabel"] > div > p {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.68rem !important;
    font-style: normal !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    color: {_H_DIM} !important;
    font-weight: 500 !important;
}}

/* ── DataFrames / Tables ──────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid {_H_BORDER} !important;
}}
[data-testid="stDataFrame"] table {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.85rem !important;
}}
[data-testid="stDataFrame"] th {{
    font-size: 0.62rem !important;
    font-style: normal !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    color: {_H_DIM} !important;
    font-weight: 500 !important;
    background-color: {_H_CARD} !important;
}}

/* ── Alert / info boxes ───────────────────────────────────────────────── */
div[data-testid="stAlertContainer"] > div {{
    border-radius: 8px !important;
    border-left-width: 2px !important;
    background-color: {_H_CARD} !important;
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.82rem !important;
    border-left-color: {_H_ACCENT} !important;
}}

/* ── Radio / Checkbox ─────────────────────────────────────────────────── */
.stRadio label p, .stCheckbox label p {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.82rem !important;
    color: {_H_SUBTLE} !important;
}}

/* ── Dividers ─────────────────────────────────────────────────────────── */
hr {{ border-color: {_H_BORDER} !important; }}

/* ── Scrollbars ───────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {_H_BG}; }}
::-webkit-scrollbar-thumb {{ background: {_H_BORDER}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {_H_SUBTLE}; }}

/* ── Spinner ──────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div > div {{ border-top-color: {_H_ACCENT} !important; }}

/* ── Markdown body text ───────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.92rem !important;
    color: #cccccc !important;
    line-height: 1.7 !important;
}}
[data-testid="stMarkdownContainer"] strong {{
    color: {_H_FG} !important;
    font-weight: 600 !important;
}}

/* ── Caption text — Paper Trading: small italic, dim ─────────────────── */
[data-testid="stCaptionContainer"] p {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.72rem !important;
    font-style: italic !important;
    color: {_H_DIM} !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}}
"""
st.markdown(f"<style>{_HOME_CSS}</style>", unsafe_allow_html=True)


def _home_plotly_layout(**kwargs) -> dict:
    """Plotly layout — Carbon bg, cyan accent, Palatino labels."""
    _font = dict(family="'Palatino Linotype', 'Book Antiqua', Palatino, serif", size=11, color=_H_SUBTLE)
    defaults = dict(
        template="plotly_dark",
        paper_bgcolor=_H_CARD,
        plot_bgcolor=_H_BG,
        font=_font,
        title_font=dict(family="'Palatino Linotype', 'Book Antiqua', Palatino, serif", size=11,
                        color=_H_SUBTLE, weight=500),
        legend=dict(
            bgcolor="rgba(26,26,26,0.95)",
            bordercolor=_H_BORDER,
            borderwidth=1,
            font=dict(family="'Palatino Linotype', 'Book Antiqua', Palatino, serif", size=10, color=_H_SUBTLE),
        ),
        margin=dict(l=48, r=24, t=40, b=36),
        xaxis=dict(
            gridcolor="#1c1c1c",
            zerolinecolor=_H_BORDER,
            linecolor=_H_BORDER,
            tickfont=dict(family="'Palatino Linotype', 'Book Antiqua', Palatino, serif", size=9, color=_H_DIM),
            title_font=dict(family="'Palatino Linotype', 'Book Antiqua', Palatino, serif", size=9, color=_H_DIM),
        ),
        yaxis=dict(
            gridcolor="#1c1c1c",
            zerolinecolor=_H_BORDER,
            linecolor=_H_BORDER,
            tickfont=dict(family="'Palatino Linotype', 'Book Antiqua', Palatino, serif", size=9, color=_H_DIM),
            title_font=dict(family="'Palatino Linotype', 'Book Antiqua', Palatino, serif", size=9, color=_H_DIM),
        ),
    )
    defaults.update(kwargs)
    return defaults


st.markdown(
    f"""
    <div style="text-align:center; padding: 2rem 0 0.5rem 0;">
        <div style="font-family:'Palatino Linotype','Book Antiqua',Palatino,serif;
                    font-size:2.4rem; font-weight:400; font-style:italic;
                    color:{_H_ACCENT}; letter-spacing:0.01em; margin-bottom:0.4rem;">
            Portfolio Position Tracker
        </div>
        <div style="font-family:'Palatino Linotype','Book Antiqua',Palatino,serif;
                    font-size:0.78rem; font-weight:400; font-style:italic;
                    color:{_H_DIM}; letter-spacing:0.18em;">
            {datetime.now().strftime('%A, %B %-d, %Y')}
        </div>
    </div>
    <hr style="border:none; border-top:1px solid {_H_BORDER}; margin:1rem 0 1.5rem 0;">
    """,
    unsafe_allow_html=True,
)

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
            # Also restore options positions from the same file
            _reloaded_opts = load_options_positions()
            if _reloaded_opts:
                st.session_state['options_positions'] = _reloaded_opts
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

summary                = None   # set inside if-block when stocks are loaded
position_metrics       = None   # set inside if-block; accessible for chart slot-fill
_opts_tbl_slot         = None   # placeholder: options single-leg table
_strat_btn_slot        = None   # placeholder: strategy builder toggle + UI
_opts_mgmt_slot        = None   # placeholder: manage expanders + strategies + metrics
_combined_slot         = None   # placeholder: combined portfolio totals
_holdings_charts_slot  = None   # placeholder: holdings + options charts (filled after opts prices)

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

    # Compute Day's P&L — fetch 2-day history for all tickers in one call
    _day_pnl = 0.0
    _day_pnl_pct = 0.0
    try:
        import yfinance as _yf2
        _hist2 = _yf2.download(
            " ".join(tickers), period="2d", auto_adjust=True,
            progress=False, threads=True
        )["Close"]
        if hasattr(_hist2, 'columns'):  # multi-ticker
            _hist2 = _hist2[tickers] if set(tickers).issubset(_hist2.columns) else _hist2
        if len(_hist2) >= 2:
            _prev = _hist2.iloc[-2]
            _curr = _hist2.iloc[-1]
            for _, _row in positions_df.iterrows():
                _t = _row['ticker']
                _sh = float(_row['shares'])
                if _t in _curr.index and _t in _prev.index:
                    _day_pnl += (_curr[_t] - _prev[_t]) * _sh
            _cost_basis_total = (positions_df['shares'] * positions_df['cost_basis']).sum()
            _day_pnl_pct = (_day_pnl / _cost_basis_total * 100) if _cost_basis_total else 0.0
    except Exception:
        pass

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Portfolio Value",
            f"${summary['total_value']:,.2f}",
            f"{summary['total_pnl_pct']:+.2f}% all-time"
        )

    with col2:
        st.metric(
            "Day's P&L",
            f"${_day_pnl:,.2f}",
            f"{_day_pnl_pct:+.2f}%"
        )

    with col3:
        st.metric(
            "Positions",
            summary['num_positions'],
            f"{summary['winners']}W / {summary['losers']}L"
        )

    # Position Details Table
    st.subheader("📋 Position Details")

    _th = (
        f"text-align:{{align}}; padding:0.55rem 1rem; "
        f"font-family:'Palatino Linotype','Book Antiqua',Palatino,serif; "
        f"font-size:0.62rem; text-transform:uppercase; letter-spacing:0.13em; "
        f"color:{_H_ACCENT}; font-weight:500; border-bottom:1px solid {_H_BORDER}; "
        f"white-space:nowrap;"
    )
    _td = (
        f"padding:0.6rem 1rem; font-family:'Palatino Linotype','Book Antiqua',Palatino,serif; "
        f"font-size:0.88rem; color:{_H_FG}; border-bottom:1px solid {_H_BORDER}; "
        f"vertical-align:middle; text-align:{{align}};"
    )

    _gains_losses_header = (
        f"Total Gains "
        f"(<span style='color:{_H_LOSS};'>Losses</span>)"
    )
    _headers = [
        ("left",  "Ticker"),
        ("right", "Shares"),
        ("right", "Average Cost"),
        ("right", "Current Price"),
        ("right", "Current Value"),
        ("right", _gains_losses_header),
    ]
    _head_html = "".join(
        f'<th style="{_th.format(align=a)}">{h}</th>' for a, h in _headers
    )

    _rows_html = ""
    for _, _r in position_metrics.iterrows():
        _pnl       = float(_r['total_pnl'])
        _pnl_pct   = float(_r['total_pnl_pct'])
        _pnl_color = _H_GAIN if _pnl > 0 else (_H_LOSS if _pnl < 0 else _H_DIM)
        _pnl_sign  = "+" if _pnl >= 0 else ""
        _pct_sign  = "+" if _pnl_pct >= 0 else ""
        _rows_html += (
            f'<tr>'
            f'<td style="{_td.format(align="left")}">{_r["ticker"]}</td>'
            f'<td style="{_td.format(align="right")}">{float(_r["shares"]):,.4g}</td>'
            f'<td style="{_td.format(align="right")}">${float(_r["cost_basis"]):.2f}</td>'
            f'<td style="{_td.format(align="right")}">${float(_r["current_price"]):.2f}</td>'
            f'<td style="{_td.format(align="right")}">${float(_r["current_value"]):,.2f}</td>'
            f'<td style="{_td.format(align="right")} color:{_pnl_color};">'
            f'  {_pnl_sign}${abs(_pnl):,.2f}<br>'
            f'  <span style="font-size:0.78rem; opacity:0.80;">'
            f'    {_pct_sign}{_pnl_pct:.2f}%'
            f'  </span>'
            f'</td>'
            f'</tr>'
        )

    _table_html = f"""
    <div style="border-radius:8px; overflow:hidden; border:1px solid {_H_BORDER}; margin-bottom:1.5rem;">
      <div style="overflow-x:auto;">
        <table style="width:100%; border-collapse:collapse;">
          <thead><tr>{_head_html}</tr></thead>
          <tbody>{_rows_html}</tbody>
        </table>
      </div>
    </div>
    """
    st.markdown(_table_html, unsafe_allow_html=True)

    # ── Update Positions ──────────────────────────────────────────────────
    with st.expander("✏️ Update Positions"):
        for _up_idx, _up_row in positions_df.iterrows():
            _up_ticker  = _up_row['ticker']
            _up_shares  = float(_up_row['shares'])
            _up_cost    = float(_up_row['cost_basis'])
            st.markdown(
                f"<span style='font-family:Palatino Linotype,serif; font-size:0.88rem; "
                f"color:{_H_ACCENT}; font-style:italic;'>{_up_ticker}</span>"
                f"<span style='color:{_H_DIM}; font-size:0.78rem;'> — "
                f"{_up_shares:,.4g} shares @ ${_up_cost:.2f}</span>",
                unsafe_allow_html=True,
            )
            _uc1, _uc2, _uc3, _uc4 = st.columns([2, 2, 2, 1])
            with _uc1:
                _up_action = st.selectbox(
                    "Action", ["Buy More", "Sell Shares"],
                    key=f"_upd_act_{_up_ticker}_{_up_idx}",
                    label_visibility="collapsed",
                )
            with _uc2:
                _up_qty = st.number_input(
                    "Shares", min_value=0.0001, step=1.0, format="%.4g",
                    key=f"_upd_qty_{_up_ticker}_{_up_idx}",
                    label_visibility="collapsed",
                )
            with _uc3:
                _up_price = st.number_input(
                    "Price / Share ($)", min_value=0.0001, step=0.01,
                    value=float(_up_cost), format="%.2f",
                    key=f"_upd_price_{_up_ticker}_{_up_idx}",
                    label_visibility="collapsed",
                )
            with _uc4:
                if st.button("Apply", key=f"_upd_apply_{_up_ticker}_{_up_idx}", type="primary"):
                    _manual = st.session_state.get('manual_positions', [])
                    for _mp in _manual:
                        if _mp['ticker'] == _up_ticker:
                            if _up_action == "Buy More":
                                _new_total = _up_shares + _up_qty
                                _mp['cost_basis'] = (
                                    (_up_shares * _up_cost + _up_qty * _up_price) / _new_total
                                )
                                _mp['shares'] = _new_total
                            else:  # Sell Shares
                                _remaining = _up_shares - _up_qty
                                if _remaining <= 0:
                                    _manual = [p for p in _manual if p['ticker'] != _up_ticker]
                                else:
                                    _mp['shares'] = _remaining
                                    # cost basis unchanged on partial sell
                            break
                    st.session_state['manual_positions'] = _manual
                    _upd_df = pd.DataFrame(_manual) if _manual else pd.DataFrame()
                    if not _upd_df.empty:
                        st.session_state['positions'] = _upd_df
                        save_portfolio(_upd_df)
                    else:
                        st.session_state.pop('positions', None)
                    st.rerun()
            st.markdown(
                f"<hr style='border:none; border-top:1px solid {_H_BORDER}; margin:0.5rem 0;'>",
                unsafe_allow_html=True,
            )

    # Reserve slots filled below after options data is computed
    _opts_tbl_slot  = st.container()   # single-leg options table
    _strat_btn_slot = st.container()   # strategy builder toggle + UI
    _opts_mgmt_slot = st.container()   # manage expanders + strategies + summary metrics
    _combined_slot  = st.container()   # combined portfolio totals

    # Holdings Breakdown
    st.subheader("🥧 Holdings Breakdown")

    # Charts are filled after options prices are fetched (slot pattern)
    _holdings_charts_slot = st.container()

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
                    score_color    = _H_ACCENT if diversif_score >= 60 else _H_SUBTLE if diversif_score >= 40 else _H_LOSS
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
                            **_home_plotly_layout()
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
                line=dict(color=_H_ACCENT, width=2),
                fill='tozeroy',
                fillcolor='rgba(34,211,238,0.08)'
            ))

            # Benchmark overlay
            if benchmark_data is not None and not benchmark_data.empty:
                bench_rets = benchmark_data.pct_change().dropna()
                bench_cum  = (1 + bench_rets).cumprod()
                fig_cum.add_trace(go.Scatter(
                    x=bench_cum.index,
                    y=(bench_cum - 1) * 100,
                    name=benchmark_ticker,
                    line=dict(color='#888888', width=1.5, dash='dash'),
                ))

            fig_cum.update_layout(
                title="Portfolio vs Benchmark Cumulative Return",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                height=400,
                hovermode='x unified',
                **_home_plotly_layout()
            )
            fig_cum.add_hline(y=0, line_dash="dash", line_color=_H_BORDER, opacity=0.8)
            st.plotly_chart(fig_cum, use_container_width=True)

            # Drawdown chart
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max * 100

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                name='Drawdown',
                line=dict(color=_H_LOSS, width=2),
                fill='tozeroy',
                fillcolor='rgba(251,113,133,0.12)'
            ))
            fig_dd.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=300,
                **_home_plotly_layout()
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
        fig_hist.add_hline(y=0, line_dash="dash", line_color=_H_BORDER, opacity=0.9)
        fig_hist.update_layout(
            title="Position Unrealized P&L Over Time ($)",
            xaxis_title="Date",
            yaxis_title="Unrealized P&L ($)",
            height=450,
            hovermode='x unified',
            **_home_plotly_layout()
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

# ── OPTIONS POSITIONS ────────────────────────────────────────────────────
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

# Collect all open underlying tickers (+ strategy builder ticker if active)
_open_uls = set()
for _, _p in _singles:
    if _p['status'] == 'open':
        _open_uls.add(_p['underlying'])
for _, _p in _strategies:
    if _p['status'] == 'open':
        _open_uls.add(_p['underlying'])
_sb_ul_loaded = st.session_state.get('_sb_ul_loaded', '')
if _sb_ul_loaded:
    _open_uls.add(_sb_ul_loaded)
_ul_prices: dict = loader.fetch_current_prices(list(_open_uls)) if _open_uls else {}

_total_opt_val = 0.0
_total_opt_pnl = 0.0

# Context managers — route display into the pre-positioned slots (or inline fallback)
_mgmt_ctx = _opts_mgmt_slot or st.container()
_comb_ctx = _combined_slot  or st.container()

# ── Strategy Builder toggle + UI ─────────────────────────────────────────
_sb_ctx = _strat_btn_slot or st.container()

if '_sb_show' not in st.session_state:
    st.session_state['_sb_show'] = False

with _sb_ctx:
    _toggle_label = "✖ Close Multi-Leg Builder" if st.session_state['_sb_show'] else "🔀 Add Multi-Leg Strategy"
    _toggle_type  = "secondary" if st.session_state['_sb_show'] else "primary"
    if st.button(_toggle_label, key="_strat_toggle", type=_toggle_type, use_container_width=False):
        st.session_state['_sb_show'] = not st.session_state['_sb_show']
        if not st.session_state['_sb_show']:
            for _k in ['_sb_exps','_sb_ul_loaded','_sb_near_chain','_sb_far_chain',
                       '_sb_calls_chain','_sb_puts_chain','_sb_single_exp_loaded',
                       '_sb_near_exp_loaded','_sb_far_exp_loaded','_sb_custom_legs']:
                st.session_state.pop(_k, None)
        st.rerun()

# ── Strategy Builder UI ───────────────────────────────────────────────────
with _sb_ctx:
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
                    _iclr  = _H_GAIN if _pos2 == 'long' else _H_LOSS
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

    # ── Build styled HTML table for single-leg options ────────────────────
    _o_th = (
        f"text-align:{{align}}; padding:0.55rem 1rem; "
        f"font-family:'Palatino Linotype','Book Antiqua',Palatino,serif; "
        f"font-size:0.62rem; text-transform:uppercase; letter-spacing:0.13em; "
        f"color:{_H_ACCENT}; font-weight:500; border-bottom:1px solid {_H_BORDER}; "
        f"white-space:nowrap;"
    )
    _o_td = (
        f"padding:0.55rem 1rem; font-family:'Palatino Linotype','Book Antiqua',Palatino,serif; "
        f"font-size:0.85rem; color:{_H_FG}; border-bottom:1px solid {_H_BORDER}; "
        f"vertical-align:middle; text-align:{{align}};"
    )
    _o_gl_hdr = f"Total Gains (<span style='color:{_H_LOSS};'>Losses</span>)"
    _o_headers = [
        ("left",  "Underlying"), ("left",  "Type"),   ("right", "Strike"),
        ("right", "Expiry"),     ("right", "DTE"),     ("right", "Contracts"),
        ("right", "Cost / Share"), ("right", "Current"),
        ("right", _o_gl_hdr),   ("right", "Delta"),   ("right", "Theta"),
        ("right", "IV"),         ("left",  "Status"),
    ]
    _o_head_html = "".join(
        f'<th style="{_o_th.format(align=a)}">{h}</th>' for a, h in _o_headers
    )
    _o_rows_html = ""
    for _r2 in _sl_rows:
        _pnl_raw  = _r2['P&L $']
        _pct_raw  = _r2['P&L %']
        # Determine color from the numeric value inside the formatted string
        try:
            _pnl_num = float(_pnl_raw.replace('$','').replace(',',''))
            _o_pc = _H_GAIN if _pnl_num > 0 else (_H_LOSS if _pnl_num < 0 else _H_DIM)
        except (ValueError, AttributeError):
            _pnl_num = None
            _o_pc = _H_DIM
        _pnl_cell = (
            f'<span style="color:{_o_pc};">{_pnl_raw}<br>'
            f'<span style="font-size:0.78rem; opacity:0.80;">{_pct_raw}</span></span>'
            if _pnl_num is not None else f'<span style="color:{_H_DIM};">N/A</span>'
        )
        _o_rows_html += (
            f'<tr>'
            f'<td style="{_o_td.format(align="left")}">{_r2["Underlying"]}</td>'
            f'<td style="{_o_td.format(align="left")}">{_r2["Type"]}</td>'
            f'<td style="{_o_td.format(align="right")}">{_r2["Strike"]}</td>'
            f'<td style="{_o_td.format(align="right")}">{_r2["Expiry"]}</td>'
            f'<td style="{_o_td.format(align="right")}">{_r2["DTE"]}</td>'
            f'<td style="{_o_td.format(align="right")}">{_r2["Contracts"]}</td>'
            f'<td style="{_o_td.format(align="right")}">{_r2["Cost/sh"]}</td>'
            f'<td style="{_o_td.format(align="right")}">{_r2["Current"]}</td>'
            f'<td style="{_o_td.format(align="right")}">{_pnl_cell}</td>'
            f'<td style="{_o_td.format(align="right")}">{_r2["Delta"]}</td>'
            f'<td style="{_o_td.format(align="right")}">{_r2["Theta"]}</td>'
            f'<td style="{_o_td.format(align="right")}">{_r2["IV"]}</td>'
            f'<td style="{_o_td.format(align="left")}">{_r2["Status"]}</td>'
            f'</tr>'
        )
    _o_tbl_html = f"""
    <div style="border-radius:8px; overflow:hidden; border:1px solid {_H_BORDER}; margin-bottom:1rem;">
      <div style="overflow-x:auto;">
        <table style="width:100%; border-collapse:collapse;">
          <thead><tr>{_o_head_html}</tr></thead>
          <tbody>{_o_rows_html}</tbody>
        </table>
      </div>
    </div>
    """
    # Render inside the pre-positioned slot (between Position Details and Holdings Breakdown)
    # or inline if no stock positions were loaded
    _render_target = _opts_tbl_slot if _opts_tbl_slot is not None else st
    with _render_target:
        st.subheader("📊 Options Positions")
        st.markdown("#### Single-Leg Positions")
        st.markdown(_o_tbl_html, unsafe_allow_html=True)

    # Manage single-leg positions
    _open_count  = _sl_open_cnt
    _closed_count = _sl_cls_cnt
    with _mgmt_ctx:
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
                _clr3 = _H_GAIN if _pnl3 >= 0 else _H_LOSS
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
with _mgmt_ctx:
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

with _mgmt_ctx:
    if not _singles and not _strategies:
        st.info("No options positions yet. Use the sidebar **(single legs)** or **🔀 Add Strategy** button above.")

# ── Options summary metrics ───────────────────────────────────────────────
with _mgmt_ctx:
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
with _comb_ctx:
 if (_singles or _strategies) and positions_df is not None and summary is not None:
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

# ── Holdings charts — filled here after options prices are known ──────────
if _holdings_charts_slot is not None and position_metrics is not None:

    # ── Build options slice data from open positions ──────────────────────
    _opt_slices   = []   # [{label, value, pnl}] for options breakdown pie + P&L bar
    _opt_total_val = 0.0

    for _, _cp in _singles:
        if _cp['status'] != 'open':
            continue
        _lp_c  = _cp.get('_live_price')
        _ct_c  = _cp['cost_basis'] * _cp['contracts'] * 100
        _mv_c  = (_lp_c * _cp['contracts'] * 100) if _lp_c is not None else _ct_c
        _mv_c  = max(_mv_c, 0) or _ct_c          # guarantee positive for pie
        _pnl_c = (_mv_c - _ct_c) if _lp_c is not None else 0.0
        _lbl_c = f"{_cp['underlying']} {_cp['option_type'].upper()} ${_cp['strike']:.0f}"
        _opt_slices.append({'label': _lbl_c, 'value': _mv_c, 'pnl': _pnl_c})
        _opt_total_val += _mv_c

    for _, _sp in _strategies:
        if _sp['status'] != 'open':
            continue
        _legs_s  = _sp.get('legs', [])
        _net_ent = sum(
            l['cost_basis'] * l['contracts'] * 100 * (1 if l['position'] == 'long' else -1)
            for l in _legs_s
        )
        _net_cur_s, _all_ok_s = 0.0, True
        for _lg_s in _legs_s:
            _lp_s = _lg_s.get('_live_price')
            if _lp_s is None:
                _all_ok_s = False
                break
            _net_cur_s += _lp_s * _lg_s['contracts'] * 100 * (1 if _lg_s['position'] == 'long' else -1)
        if not _all_ok_s:
            _net_cur_s = _net_ent               # cost basis fallback
        _mv_s  = max(abs(_net_cur_s), abs(_net_ent)) or 1.0   # positive for pie
        _pnl_s = (_net_cur_s - _net_ent) if _all_ok_s else 0.0
        _opt_slices.append({'label': _sp['name'], 'value': _mv_s, 'pnl': _pnl_s})
        _opt_total_val += _mv_s

    with _holdings_charts_slot:
        _hcol1, _hcol2 = st.columns(2)

        # ── LEFT: pie toggle ──────────────────────────────────────────────
        with _hcol1:
            _pie_view = st.radio(
                "Pie view", ["Portfolio Holdings", "Options Breakdown"],
                horizontal=True, key="_pie_view", label_visibility="collapsed",
            )

            if _pie_view == "Portfolio Holdings":
                # Stock slices + one Options bucket
                _pie_labels = list(position_metrics['ticker'])
                _pie_vals   = list(position_metrics['current_value'])
                _pie_colors = [_HOLDING_COLORS[i % len(_HOLDING_COLORS)]
                               for i in range(len(_pie_labels))]
                if _opt_total_val > 0:
                    _pie_labels.append("Options")
                    _pie_vals.append(_opt_total_val)
                    _pie_colors.append('#f59e0b')   # amber bucket

                fig_pie = go.Figure(data=[go.Pie(
                    labels=_pie_labels, values=_pie_vals, hole=0.3,
                    textinfo='label+percent',
                    marker=dict(colors=_pie_colors),
                )])
                fig_pie.update_layout(
                    title="Portfolio Allocation by Value",
                    height=400, **_home_plotly_layout()
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            else:  # Options Breakdown
                if _opt_slices:
                    _op_labels  = [s['label'] for s in _opt_slices]
                    _op_vals    = [s['value'] for s in _opt_slices]
                    _op_colors  = [_OPT_COLORS[i % len(_OPT_COLORS)]
                                   for i in range(len(_op_labels))]
                    fig_opie = go.Figure(data=[go.Pie(
                        labels=_op_labels, values=_op_vals, hole=0.3,
                        textinfo='label+percent',
                        marker=dict(colors=_op_colors),
                    )])
                    fig_opie.update_layout(
                        title="Options Allocation by Market Value",
                        height=400, **_home_plotly_layout()
                    )
                    st.plotly_chart(fig_opie, use_container_width=True)
                else:
                    st.info("No open options positions to display.")

        # ── RIGHT: P&L bar toggle ─────────────────────────────────────────
        with _hcol2:
            _bar_view = st.radio(
                "P&L view", ["Stock P&L", "Options P&L"],
                horizontal=True, key="_bar_view", label_visibility="collapsed",
            )

            if _bar_view == "Stock P&L":
                _bar_colors = position_metrics['total_pnl'].apply(
                    lambda x: _H_GAIN if x > 0 else _H_LOSS if x < 0 else _H_DIM
                )
                fig_bar = go.Figure(data=[go.Bar(
                    x=position_metrics['ticker'],
                    y=position_metrics['total_pnl'],
                    marker_color=_bar_colors,
                    text=position_metrics['total_pnl'],
                    texttemplate='$%{text:,.0f}',
                    textposition='outside',
                )])
                fig_bar.update_layout(
                    title="Stock P&L by Position",
                    xaxis_title="Ticker", yaxis_title="P&L ($)",
                    height=400, **_home_plotly_layout()
                )
                fig_bar.add_hline(y=0, line_dash="dash", line_color=_H_BORDER)
                st.plotly_chart(fig_bar, use_container_width=True)

            else:  # Options P&L
                if _opt_slices:
                    _op_pnl_colors = [
                        _H_GAIN if s['pnl'] > 0 else _H_LOSS if s['pnl'] < 0 else _H_DIM
                        for s in _opt_slices
                    ]
                    fig_obar = go.Figure(data=[go.Bar(
                        x=[s['label'] for s in _opt_slices],
                        y=[s['pnl']   for s in _opt_slices],
                        marker_color=_op_pnl_colors,
                        text=[s['pnl'] for s in _opt_slices],
                        texttemplate='$%{text:,.0f}',
                        textposition='outside',
                    )])
                    fig_obar.update_layout(
                        title="Options Unrealised P&L",
                        xaxis_title="Position", yaxis_title="P&L ($)",
                        height=400, **_home_plotly_layout()
                    )
                    fig_obar.add_hline(y=0, line_dash="dash", line_color=_H_BORDER)
                    st.plotly_chart(fig_obar, use_container_width=True)
                else:
                    st.info("No open options positions to display.")

