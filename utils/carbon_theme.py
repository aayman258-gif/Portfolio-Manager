"""
Carbon Theme — shared across all pages
Dark Carbon bg (#111111) · Cyan accent (#22d3ee) · Palatino serif typography
Pink-red losses (#fb7185)
"""

import streamlit as st

# ── Colour palette ─────────────────────────────────────────────────────────────
BG      = "#111111"
CARD    = "#1a1a1a"
BORDER  = "#2a2a2a"
FG      = "#ffffff"
DIM     = "#555555"
SUBTLE  = "#888888"
ACCENT  = "#22d3ee"   # cyan
GAIN    = "#22d3ee"   # positive P&L
LOSS    = "#fb7185"   # negative P&L / pink-red
AMBER   = "#f59e0b"   # options accent

# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = f"""
/* ══ CARBON THEME ══════════════════════════════════════════════════════════════
   #111111 bg · #1a1a1a card · #2a2a2a border · #22d3ee cyan accent
   Palatino serif typography · flat / no shadows / rounded edges
   ══════════════════════════════════════════════════════════════════════════════ */

/* ── Global & App background ─────────────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stMain"],
section[data-testid="stAppViewContainer"] > div:first-child {{
    background-color: {BG} !important;
    box-shadow: none !important;
}}
.main .block-container,
[data-testid="stMainBlockContainer"] {{
    background-color: {BG} !important;
    padding-top: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
    width: 100% !important;
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 14px !important;
    color: {FG} !important;
    box-shadow: none !important;
    border: none !important;
}}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {CARD} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
    color: {SUBTLE} !important;
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.68rem !important;
    font-style: normal !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.40em !important;
    border-left: none !important;
    padding-left: 0 !important;
}}

/* ── Paper Trading: Palatino serif throughout ────────────────────────────── */
* {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
}}

/* ── Page title (h1) ─────────────────────────────────────────────────────── */
h1 {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 2.2rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    color: {ACCENT} !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}}

/* ── Subheaders (h2) ─────────────────────────────────────────────────────── */
[data-testid="stHeadingWithActionElements"] h2 {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 1.15rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    color: {ACCENT} !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    border-left: none !important;
    padding-left: 0 !important;
    margin-top: 2rem !important;
    margin-bottom: 0.8rem !important;
}}

/* ── Section headers (h3) — italic serif, cyan ───────────────────────────── */
[data-testid="stHeadingWithActionElements"] h3 {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 1.1rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    color: {ACCENT} !important;
    border-left: none !important;
    padding-left: 0 !important;
    margin-top: 1.8rem !important;
    margin-bottom: 0.8rem !important;
}}

/* ── Inline markdown h4 headers ──────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] h4 {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    color: {ACCENT} !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.5rem !important;
}}

/* ── Metric cards ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    padding: 1.2rem 1.4rem 1rem !important;
    box-shadow: none !important;
}}
[data-testid="stMetricLabel"] > div {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    color: {DIM} !important;
    font-size: 0.68rem !important;
    font-style: normal !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    font-weight: 500 !important;
}}
[data-testid="stMetricValue"] > div {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    color: {FG} !important;
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
    color: {ACCENT} !important;
}}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {CARD} !important;
    border-radius: 8px !important;
    padding: 4px 6px !important;
    gap: 4px !important;
    border: 1px solid {BORDER} !important;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent !important;
    color: {DIM} !important;
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
    color: {ACCENT} !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {{
    color: {SUBTLE} !important;
    background-color: rgba(255,255,255,0.04) !important;
}}

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border: none !important;
    border-top: 1px solid {BORDER} !important;
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
    color: {DIM} !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
}}
[data-testid="stExpander"] summary:hover span {{
    color: {ACCENT} !important;
}}

/* ── Layout block wrappers — transparent ─────────────────────────────────── */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] {{
    background-color: transparent !important;
    box-shadow: none !important;
}}
/* ── Explicit border containers only (st.container(border=True)) ─────────── */
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    background-color: {CARD} !important;
    box-shadow: none !important;
}}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {{
    border-radius: 8px !important;
    border: 1px solid {BORDER} !important;
    background-color: transparent !important;
    color: {DIM} !important;
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
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
    background-color: transparent !important;
    box-shadow: none !important;
}}
.stButton > button[kind="primary"] {{
    background: {ACCENT} !important;
    border: none !important;
    color: {BG} !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: #38e0f8 !important;
    box-shadow: none !important;
    transform: none !important;
    color: {BG} !important;
}}
.stButton > button[kind="secondary"] {{
    border: 1px solid {ACCENT} !important;
    color: {ACCENT} !important;
    background-color: transparent !important;
}}
.stButton > button[kind="secondary"]:hover {{
    background-color: rgba(34, 211, 238, 0.08) !important;
}}

/* ── Input widgets ───────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stDateInput > div > div > input {{
    background-color: {CARD} !important;
    border-color: {BORDER} !important;
    border-radius: 8px !important;
    color: {FG} !important;
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.82rem !important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: none !important;
}}
label[data-testid="stWidgetLabel"] > div > p {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.68rem !important;
    font-style: normal !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    color: {DIM} !important;
    font-weight: 500 !important;
}}

/* ── Sliders ─────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-testid="stThumbValue"] {{
    color: {ACCENT} !important;
}}
[data-testid="stSlider"] [data-testid="stSliderTrack"] > div:nth-child(2) {{
    background-color: {ACCENT} !important;
}}

/* ── DataFrames / Tables ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid {BORDER} !important;
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
    color: {DIM} !important;
    font-weight: 500 !important;
    background-color: {CARD} !important;
}}

/* ── Alert / info boxes ──────────────────────────────────────────────────── */
div[data-testid="stAlertContainer"] > div {{
    border-radius: 8px !important;
    border-left-width: 2px !important;
    background-color: {CARD} !important;
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.82rem !important;
    border-left-color: {ACCENT} !important;
}}

/* ── Radio / Checkbox ────────────────────────────────────────────────────── */
.stRadio label p, .stCheckbox label p {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.82rem !important;
    color: {SUBTLE} !important;
}}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr {{ border-color: {BORDER} !important; }}

/* ── Scrollbars ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {SUBTLE}; }}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div > div {{ border-top-color: {ACCENT} !important; }}

/* ── Markdown body text ──────────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.92rem !important;
    color: #cccccc !important;
    line-height: 1.7 !important;
}}
[data-testid="stMarkdownContainer"] strong {{
    color: {FG} !important;
    font-weight: 600 !important;
}}

/* ── Caption text ────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {{
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
    font-size: 0.72rem !important;
    font-style: italic !important;
    color: {DIM} !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}}

/* ── Chat messages (AI Assistant) ────────────────────────────────────────── */
[data-testid="stChatMessage"] {{
    background-color: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}}
[data-testid="stChatInput"] textarea {{
    background-color: {CARD} !important;
    border-color: {BORDER} !important;
    border-radius: 8px !important;
    color: {FG} !important;
    font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: none !important;
}}
"""

_PALATINO = "'Palatino Linotype', 'Book Antiqua', Palatino, serif"


def apply_carbon_theme() -> None:
    """Inject Carbon theme CSS. Call once per page after st.set_page_config()."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def carbon_plotly_layout(**kwargs) -> dict:
    """Plotly layout — Carbon bg, cyan accent, Palatino labels."""
    _font = dict(family=_PALATINO, size=11, color=SUBTLE)
    defaults = dict(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=BG,
        font=_font,
        title_font=dict(family=_PALATINO, size=11, color=SUBTLE, weight=500),
        legend=dict(
            bgcolor="rgba(26,26,26,0.95)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(family=_PALATINO, size=10, color=SUBTLE),
        ),
        margin=dict(l=48, r=24, t=40, b=36),
        xaxis=dict(
            gridcolor="#1c1c1c",
            zerolinecolor=BORDER,
            linecolor=BORDER,
            tickfont=dict(family=_PALATINO, size=9, color=DIM),
            title_font=dict(family=_PALATINO, size=9, color=DIM),
        ),
        yaxis=dict(
            gridcolor="#1c1c1c",
            zerolinecolor=BORDER,
            linecolor=BORDER,
            tickfont=dict(family=_PALATINO, size=9, color=DIM),
            title_font=dict(family=_PALATINO, size=9, color=DIM),
        ),
    )
    defaults.update(kwargs)
    return defaults


def pnl_color(value: float) -> str:
    """Return CSS color string for P&L value (cyan positive, pink-red negative)."""
    return GAIN if value >= 0 else LOSS


def regime_color(regime: str) -> str:
    """Return hex color for a regime name (Carbon palette)."""
    return {
        "Low Vol":        GAIN,    # cyan
        "High Vol":       LOSS,    # pink-red
        "Trending":       ACCENT,  # cyan (same, directional)
        "Mean Reversion": AMBER,   # amber
        "Unknown":        DIM,     # muted
    }.get(regime, DIM)


def html_table(rows_html: str, head_html: str, margin_bottom: str = "1.5rem") -> str:
    """Return a styled Carbon-theme HTML table string for use with st.markdown."""
    return (
        f'<div style="border-radius:8px; overflow:hidden; border:1px solid {BORDER}; margin-bottom:{margin_bottom};">'
        f'<div style="overflow-x:auto;">'
        f'<table style="width:100%; border-collapse:collapse;">'
        f'<thead><tr>{head_html}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div></div>'
    )
