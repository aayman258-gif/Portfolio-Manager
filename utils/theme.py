"""
Dark Professional Theme
Teal accent (#00d4aa), glowing bordered metric cards, green/red P&L.
"""

import streamlit as st

# ── Colour palette ────────────────────────────────────────────────────────────
TEAL        = "#00d4aa"
TEAL_DIM    = "rgba(0, 212, 170, 0.35)"
TEAL_GLOW   = "rgba(0, 212, 170, 0.15)"
BLUE_ACCENT = "#4a9eff"
BG_MAIN     = "#0e1117"
BG_CARD     = "#141720"
BG_SIDE     = "#0a0d14"
BG_CARD2    = "#1a1d23"
TEXT_MUTED  = "#8b9ab0"
TEXT_MAIN   = "#e8edf3"
GREEN       = "#22c55e"
RED         = "#ef4444"
AMBER       = "#f59e0b"

# ── CSS block ─────────────────────────────────────────────────────────────────
_CSS = f"""
/* ── Global ──────────────────────────────────────────────────────── */
.main .block-container {{ padding-top: 0.75rem; }}

/* ── Glowing metric cards ─────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: linear-gradient(145deg, {BG_CARD} 0%, #101318 100%);
    border: 1px solid {TEAL_DIM};
    border-radius: 10px;
    padding: 1rem 1.25rem 0.9rem;
    box-shadow: 0 0 16px {TEAL_GLOW}, 0 4px 20px rgba(0,0,0,0.5);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}}
[data-testid="stMetric"]:hover {{
    border-color: rgba(0, 212, 170, 0.65);
    box-shadow: 0 0 28px rgba(0, 212, 170, 0.28);
}}
[data-testid="stMetricLabel"] > div {{
    color: {TEXT_MUTED} !important;
    font-size: 0.70rem !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 500;
}}
[data-testid="stMetricValue"] > div {{
    color: {TEXT_MAIN} !important;
    font-weight: 700;
    font-size: 1.55rem !important;
}}
/* Delta colouring */
[data-testid="stMetricDelta"] svg {{ display: none; }}
[data-testid="stMetricDelta"] > div {{
    font-size: 0.78rem !important;
    font-weight: 500;
}}

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {BG_CARD};
    border-radius: 10px;
    padding: 4px 6px;
    gap: 4px;
    border: 1px solid rgba(0, 212, 170, 0.14);
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent !important;
    color: {TEXT_MUTED} !important;
    border-radius: 7px !important;
    border: none !important;
    font-weight: 500;
    transition: all 0.15s ease;
    padding: 0.4rem 0.9rem !important;
}}
.stTabs [aria-selected="true"] {{
    background-color: rgba(0, 212, 170, 0.13) !important;
    color: {TEAL} !important;
    font-weight: 700 !important;
    box-shadow: 0 0 8px rgba(0, 212, 170, 0.18);
}}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {{
    background-color: rgba(0, 212, 170, 0.06) !important;
    color: #a0e8da !important;
}}

/* ── Expanders ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border: 1px solid rgba(0, 212, 170, 0.18) !important;
    border-radius: 10px !important;
    background-color: {BG_CARD};
    overflow: hidden;
}}
[data-testid="stExpander"] > div:first-child {{
    background-color: {BG_CARD} !important;
    border-radius: 9px !important;
}}

/* ── Sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {BG_SIDE} !important;
    border-right: 1px solid rgba(0, 212, 170, 0.10) !important;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
    color: {TEAL} !important;
    border-left: none !important;
    padding-left: 0 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}}

/* ── Buttons ──────────────────────────────────────────────────────── */
.stButton > button {{
    border-radius: 8px !important;
    border: 1px solid rgba(0, 212, 170, 0.25) !important;
    transition: all 0.15s ease !important;
    font-weight: 500 !important;
}}
.stButton > button:hover {{
    border-color: rgba(0, 212, 170, 0.55) !important;
    box-shadow: 0 0 14px rgba(0, 212, 170, 0.22) !important;
    color: {TEAL} !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {TEAL} 0%, #00a882 100%) !important;
    border: none !important;
    color: #0a0d14 !important;
    font-weight: 700 !important;
    box-shadow: 0 0 18px rgba(0, 212, 170, 0.3) !important;
}}
.stButton > button[kind="primary"]:hover {{
    box-shadow: 0 0 28px rgba(0, 212, 170, 0.48) !important;
    transform: translateY(-1px);
}}

/* ── Input widgets ────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background-color: {BG_CARD} !important;
    border-color: rgba(0, 212, 170, 0.25) !important;
    border-radius: 8px !important;
    color: {TEXT_MAIN} !important;
    transition: border-color 0.15s ease !important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {{
    border-color: rgba(0, 212, 170, 0.6) !important;
    box-shadow: 0 0 10px rgba(0, 212, 170, 0.15) !important;
}}
.stSlider > div [data-testid="stThumbValue"] {{
    color: {TEAL} !important;
}}

/* ── DataFrames / Tables ──────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(0, 212, 170, 0.14);
}}

/* ── Alert boxes ──────────────────────────────────────────────────── */
div[data-testid="stAlertContainer"] > div {{
    border-radius: 8px !important;
    border-left-width: 4px !important;
}}

/* ── Section h3 with accent left-border ──────────────────────────── */
h3 {{
    border-left: 3px solid {TEAL};
    padding-left: 10px;
    margin-top: 1.4rem !important;
}}

/* ── Scrollbars ───────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {BG_MAIN}; }}
::-webkit-scrollbar-thumb {{ background: rgba(0, 212, 170, 0.28); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(0, 212, 170, 0.5); }}

/* ── Dividers ─────────────────────────────────────────────────────── */
hr {{ border-color: rgba(0, 212, 170, 0.15) !important; }}

/* ── Form submit button ───────────────────────────────────────────── */
[data-testid="baseButton-secondaryFormSubmit"] {{
    border-radius: 8px !important;
}}
"""


def apply_dark_theme() -> None:
    """Inject dark professional CSS.  Call once per page after st.set_page_config()."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def dark_plotly_layout(**kwargs) -> dict:
    """
    Return a dict of plotly layout kwargs for the dark professional theme.
    Pass additional keyword args to override defaults, e.g. height=600.
    """
    defaults = dict(
        template="plotly_dark",
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_MAIN,
        font=dict(color=TEXT_MAIN, size=12),
        title_font=dict(color=TEXT_MAIN, size=14),
        legend=dict(
            bgcolor="rgba(20,23,32,0.85)",
            bordercolor=TEAL_DIM,
            borderwidth=1,
        ),
        margin=dict(l=50, r=30, t=50, b=40),
    )
    defaults.update(kwargs)
    return defaults


def pnl_color(value: float) -> str:
    """Return CSS color string for P&L value (green positive, red negative)."""
    return GREEN if value >= 0 else RED


def regime_color(regime: str) -> str:
    """Return hex color for a regime name."""
    return {
        "Low Vol":       "#22c55e",   # green
        "High Vol":      "#ef4444",   # red
        "Trending":      "#4a9eff",   # blue
        "Mean Reversion": "#f59e0b",  # amber
        "Unknown":       "#6b7a8f",   # muted
    }.get(regime, "#6b7a8f")


def chart_style_toggle(sidebar_key: str = "chart_style") -> str:
    """
    Render a sidebar radio for chart style preference.
    Returns 'candlestick' or 'line'.
    """
    choice = st.sidebar.radio(
        "Chart Style",
        options=["📈 Line Chart", "🕯️ Candlestick + Volume"],
        index=0,
        key=sidebar_key,
    )
    return "candlestick" if "Candlestick" in choice else "line"
