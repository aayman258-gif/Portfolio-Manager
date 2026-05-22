"""
Carbon Theme — shared across all pages
#0d1117 base · #161b22 card · #22d3ee cyan accent
Inter body · Palatino serif headings · ambient + glow shadows
"""

import streamlit as st

# ── Colour palette ─────────────────────────────────────────────────────────────
BG      = "#0d1117"
CARD    = "#161b22"
CARD2   = "#1c2128"
BORDER  = "#30363d"
BORDER2 = "#21262d"
FG      = "#e6edf3"
DIM     = "#7d8590"
SUBTLE  = "#484f58"
ACCENT  = "#22d3ee"   # cyan
GAIN    = "#22d3ee"   # positive P&L
LOSS    = "#fb7185"   # negative P&L / pink-red
AMBER   = "#f59e0b"   # warning / options accent
GREEN   = "#4ade80"   # strong positive signal
PURPLE  = "#a78bfa"   # secondary accent

# ── Navigation definition ──────────────────────────────────────────────────────
_NAV_ITEMS = [
    ("Command Center", "/"),
    (None, None),
    ("Markets",        "/Markets"),
    ("Scoring",        "/Scoring"),
    ("Optimization",   "/Optimization"),
    (None, None),
    ("Options Suite",  "/Options_Suite"),
    (None, None),
    ("Monte Carlo",    "/Monte_Carlo"),
    ("Fundamentals",   "/Fundamentals"),
    (None, None),
    ("Risk",           "/Risk"),
    ("Attribution",    "/Attribution"),
    ("Watchlist",      "/Watchlist"),
    (None, None),
    ("AI",             "/AI"),
    (None, None),
    ("Paper Trading",  "/Paper_Trading"),
    ("Compare",        "/Portfolio_Compare"),
]

_SANS      = "'Inter', 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
_SERIF     = "'Palatino Linotype', 'Book Antiqua', Palatino, serif"
_PALATINO  = _SERIF   # backwards-compat alias

# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = f"""
/* ══ PORTFOLIO INTELLIGENCE — INSTITUTIONAL THEME ═══════════════════════════
   #0d1117 base · #161b22 card · #30363d border · #22d3ee accent
   Inter body · Palatino serif headings · 12px radius · ambient + glow shadows
   ══════════════════════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* ── Hide Streamlit chrome ─────────────────────────────────────────────────── */
header[data-testid="stHeader"],
[data-testid="stHeader"],
.stApp > header {{
    display: none !important;
}}
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"] {{
    display: none !important;
}}

/* ── App background ────────────────────────────────────────────────────────── */
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
    padding-top: 4.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
    width: 100% !important;
    box-shadow: none !important;
    border: none !important;
}}
.stMainBlockContainer, .block-container {{
    padding-left: 2rem !important;
    max-width: 100% !important;
}}

/* ── Global typography — Inter body ──────────────────────────────────────── */
* {{
    font-family: {_SANS} !important;
    color: {FG};
}}
body {{
    background-color: {BG} !important;
    font-size: 14px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}}

/* ── Headings — Palatino serif ───────────────────────────────────────────── */
h1, h2, h3, h4,
[data-testid="stHeadingWithActionElements"] h1,
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3,
[data-testid="stHeadingWithActionElements"] h4 {{
    font-family: {_SERIF} !important;
}}

h1,
[data-testid="stHeadingWithActionElements"] h1 {{
    font-size: 2.2rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    color: {ACCENT} !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}}

[data-testid="stHeadingWithActionElements"] h2 {{
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

[data-testid="stHeadingWithActionElements"] h3 {{
    font-size: 1.05rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    color: {ACCENT} !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    border-left: none !important;
    padding-left: 0 !important;
    margin-top: 1.8rem !important;
    margin-bottom: 0.8rem !important;
}}

[data-testid="stMarkdownContainer"] h4 {{
    font-family: {_SERIF} !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    font-style: italic !important;
    color: {ACCENT} !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.5rem !important;
}}

/* ── Frosted glass top nav ────────────────────────────────────────────────── */
.pim-nav {{
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 48px;
    background: rgba(13, 17, 23, 0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid {BORDER2};
    display: flex;
    align-items: center;
    padding: 0 1.5rem;
    z-index: 9999;
    box-shadow: 0 1px 0 rgba(48,54,61,0.5), 0 4px 16px rgba(0,0,0,0.3);
}}
.pim-nav-brand {{
    font-family: {_SERIF};
    font-size: 0.62rem;
    font-weight: 600;
    color: {ACCENT};
    letter-spacing: 0.28em;
    text-transform: uppercase;
    white-space: nowrap;
    text-decoration: none;
    margin-right: 2rem;
    flex-shrink: 0;
}}
.pim-nav-items {{
    display: flex;
    align-items: center;
    gap: 0.15rem;
    overflow-x: auto;
    flex: 1;
    scrollbar-width: none;
}}
.pim-nav-items::-webkit-scrollbar {{ display: none; }}
.pim-nav-items a {{
    font-family: {_SANS};
    font-size: 0.72rem;
    font-weight: 500;
    color: {DIM};
    text-decoration: none;
    padding: 0.28rem 0.75rem;
    border-radius: 20px;
    letter-spacing: 0.04em;
    white-space: nowrap;
    transition: color 0.15s ease, background-color 0.15s ease;
    flex-shrink: 0;
}}
.pim-nav-items a:hover {{
    color: {FG};
    background-color: rgba(230,237,243,0.07);
}}
.pim-nav-items a.pim-active {{
    color: {ACCENT};
    background-color: rgba(34,211,238,0.12);
    box-shadow: 0 0 0 1px rgba(34,211,238,0.2);
}}
.pim-nav-sep {{
    width: 1px;
    height: 16px;
    background: linear-gradient(to bottom, transparent, {BORDER}, transparent);
    margin: 0 0.5rem;
    flex-shrink: 0;
}}

/* ── Layout blocks — transparent, no borders ──────────────────────────────── */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] {{
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
}}

/* ── Bordered containers (st.container(border=True)) — elevated card ─────── */
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    background-color: {CARD} !important;
    border: 1px solid {BORDER2} !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4) !important;
    transition: box-shadow 0.2s ease !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] > div:hover {{
    box-shadow: 0 4px 20px rgba(0,0,0,0.5), 0 0 0 1px rgba(34,211,238,0.12) !important;
}}

/* ── Metric cards — elevated, top gradient accent bar ─────────────────────── */
[data-testid="stMetric"] {{
    background: {CARD} !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.3rem 1rem !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4) !important;
    position: relative !important;
    overflow: hidden !important;
    transition: box-shadow 0.2s ease, transform 0.15s ease !important;
}}
[data-testid="stMetric"]:hover {{
    box-shadow: 0 6px 24px rgba(0,0,0,0.5), 0 0 0 1px rgba(34,211,238,0.2), 0 0 16px rgba(34,211,238,0.08) !important;
    transform: translateY(-1px) !important;
}}
[data-testid="stMetric"]::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, {ACCENT}, {PURPLE});
    border-radius: 12px 12px 0 0;
}}
[data-testid="stMetricLabel"] > div {{
    font-family: {_SANS} !important;
    color: {DIM} !important;
    font-size: 0.68rem !important;
    font-style: normal !important;
    text-transform: uppercase !important;
    letter-spacing: 0.10em !important;
    font-weight: 500 !important;
}}
[data-testid="stMetricValue"] > div {{
    font-family: {_SERIF} !important;
    color: {FG} !important;
    font-weight: 300 !important;
    font-size: 1.75rem !important;
    letter-spacing: -0.01em !important;
}}
[data-testid="stMetricDelta"] svg {{ display: none !important; }}
[data-testid="stMetricDelta"] > div {{
    font-family: {_SANS} !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: {ACCENT} !important;
}}

/* ── Tabs — floating pill container ──────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {CARD} !important;
    border-radius: 12px !important;
    padding: 4px 6px !important;
    gap: 4px !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent !important;
    color: {DIM} !important;
    border-radius: 8px !important;
    border: none !important;
    font-family: {_SANS} !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.38rem 1rem !important;
    transition: all 0.15s ease !important;
}}
.stTabs [aria-selected="true"] {{
    background-color: rgba(34,211,238,0.12) !important;
    color: {ACCENT} !important;
    font-weight: 600 !important;
    box-shadow: 0 0 0 1px rgba(34,211,238,0.2) !important;
}}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {{
    color: {FG} !important;
    background-color: rgba(230,237,243,0.06) !important;
}}

/* ── Expanders — elevated card style ─────────────────────────────────────── */
[data-testid="stExpander"] {{
    background-color: {CARD} !important;
    border: none !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.35) !important;
    margin-bottom: 0.75rem !important;
}}
[data-testid="stExpander"] > div:first-child {{
    background-color: transparent !important;
    border-radius: 12px !important;
    padding: 0.1rem 0.25rem !important;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {{
    font-family: {_SANS} !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: {DIM} !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}}
[data-testid="stExpander"] summary:hover span {{
    color: {ACCENT} !important;
}}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {{
    border-radius: 8px !important;
    border: 1px solid {BORDER} !important;
    background-color: transparent !important;
    color: {DIM} !important;
    font-family: {_SANS} !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    transition: color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: none !important;
}}
.stButton > button:hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
    background-color: transparent !important;
    box-shadow: 0 0 0 1px rgba(34,211,238,0.2), 0 0 8px rgba(34,211,238,0.08) !important;
}}
.stButton > button[kind="primary"] {{
    background: {ACCENT} !important;
    border: none !important;
    color: {BG} !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(34,211,238,0.3) !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: #38e0f8 !important;
    box-shadow: 0 4px 16px rgba(34,211,238,0.4) !important;
    color: {BG} !important;
}}
.stButton > button[kind="secondary"] {{
    border: 1px solid {ACCENT} !important;
    color: {ACCENT} !important;
    background-color: transparent !important;
}}
.stButton > button[kind="secondary"]:hover {{
    background-color: rgba(34,211,238,0.08) !important;
    box-shadow: 0 0 0 1px rgba(34,211,238,0.3) !important;
}}

/* ── Input widgets ───────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stDateInput > div > div > input {{
    background-color: {CARD2} !important;
    border-color: {BORDER} !important;
    border-radius: 8px !important;
    color: {FG} !important;
    font-family: {_SANS} !important;
    font-size: 0.82rem !important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(34,211,238,0.1) !important;
}}
label[data-testid="stWidgetLabel"] > div > p {{
    font-family: {_SANS} !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
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
    border-radius: 12px !important;
    overflow: hidden !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35) !important;
}}
[data-testid="stDataFrame"] table {{
    font-family: {_SANS} !important;
    font-size: 0.82rem !important;
}}
[data-testid="stDataFrame"] th {{
    font-size: 0.62rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.10em !important;
    color: {DIM} !important;
    font-weight: 600 !important;
    background-color: {CARD} !important;
}}

/* ── Alert / info boxes ──────────────────────────────────────────────────── */
div[data-testid="stAlertContainer"] > div {{
    border-radius: 10px !important;
    border-left-width: 3px !important;
    background-color: {CARD} !important;
    font-family: {_SANS} !important;
    font-size: 0.82rem !important;
    border-left-color: {ACCENT} !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    border-top: none !important;
    border-right: none !important;
    border-bottom: none !important;
}}

/* ── Radio / Checkbox ────────────────────────────────────────────────────── */
.stRadio label p, .stCheckbox label p {{
    font-family: {_SANS} !important;
    font-size: 0.82rem !important;
    color: {DIM} !important;
}}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr {{
    border: none !important;
    border-top: 1px solid {BORDER2} !important;
    margin: 1.25rem 0 !important;
}}

/* ── Scrollbars ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {DIM}; }}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div > div {{ border-top-color: {ACCENT} !important; }}

/* ── Markdown body text ──────────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
    font-family: {_SANS} !important;
    font-size: 0.88rem !important;
    color: {DIM} !important;
    line-height: 1.7 !important;
}}
[data-testid="stMarkdownContainer"] strong {{
    color: {FG} !important;
    font-weight: 600 !important;
}}

/* ── Caption text ────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {{
    font-family: {_SANS} !important;
    font-size: 0.70rem !important;
    color: {DIM} !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}}

/* ── Chat messages (AI Assistant) ────────────────────────────────────────── */
[data-testid="stChatMessage"] {{
    background-color: {CARD} !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
    margin-bottom: 0.5rem !important;
}}
[data-testid="stChatInput"] textarea {{
    background-color: {CARD} !important;
    border-color: {BORDER} !important;
    border-radius: 12px !important;
    color: {FG} !important;
    font-family: {_SANS} !important;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(34,211,238,0.1) !important;
}}

/* ── Custom metric card (pim-metric-card) ────────────────────────────────── */
.pim-metric-card {{
    background: {CARD};
    border-radius: 12px;
    padding: 1.1rem 1.3rem 1rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s ease, transform 0.15s ease;
}}
.pim-metric-card:hover {{
    box-shadow: 0 6px 24px rgba(0,0,0,0.5), 0 0 0 1px rgba(34,211,238,0.2), 0 0 16px rgba(34,211,238,0.08);
    transform: translateY(-1px);
}}
.pim-metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}}
.pim-metric-card .mc-label {{
    font-family: {_SANS};
    font-size: 0.66rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: {DIM};
    margin-bottom: 0.45rem;
}}
.pim-metric-card .mc-value {{
    font-family: {_SERIF};
    font-size: 1.75rem;
    font-weight: 300;
    color: {FG};
    letter-spacing: -0.01em;
    line-height: 1.1;
}}
.pim-metric-card .mc-delta {{
    font-family: {_SANS};
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.35rem;
}}

/* ── Bento section container ─────────────────────────────────────────────── */
.pim-section {{
    background: {CARD};
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    margin-bottom: 1rem;
}}
.pim-section-sm {{
    background: {CARD};
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.35);
    margin-bottom: 0.75rem;
}}

/* ── Sidebar (kept hidden but styled if somehow enabled) ──────────────────── */
[data-testid="stSidebar"] {{
    background-color: {CARD} !important;
    border-right: 1px solid {BORDER2} !important;
}}
"""


def apply_carbon_theme() -> None:
    """Inject Carbon theme CSS. Call once per page after st.set_page_config()."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def top_nav(active: str = "") -> None:
    """
    Render the sticky frosted-glass top navigation bar.

    Parameters
    ----------
    active : str
        Label of the currently active nav item. Case-insensitive match.
    """
    active_lower = active.strip().lower()
    items_html = ""
    for label, url in _NAV_ITEMS:
        if label is None:
            items_html += '<div class="pim-nav-sep"></div>'
        else:
            is_active = label.strip().lower() == active_lower
            cls = ' class="pim-active"' if is_active else ""
            items_html += f'<a href="{url}"{cls} target="_self">{label}</a>'

    brand_html = f'<a href="/" class="pim-nav-brand">◈ &nbsp;Portfolio Intelligence</a>'
    nav_html = (
        f'<div class="pim-nav">'
        f'{brand_html}'
        f'<div class="pim-nav-items">{items_html}</div>'
        f'</div>'
    )
    st.markdown(nav_html, unsafe_allow_html=True)


def metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    accent: str = ACCENT,
    accent2: str | None = None,
) -> str:
    """
    Return HTML for an elevated metric card with a top gradient accent bar.

    Use inside st.markdown(..., unsafe_allow_html=True) or inside a column.

    Parameters
    ----------
    label   : Upper-case label text
    value   : Primary numeric / text value
    delta   : Optional delta string (shown below value)
    accent  : Left color of the top gradient bar (default cyan)
    accent2 : Right color of the gradient (default purple)
    """
    a2 = accent2 or PURPLE
    # Detect negative delta: any occurrence of "-" after a "$" or at start
    _is_neg = delta and ("-" in delta)
    delta_color = LOSS if _is_neg else GAIN
    delta_html = (
        f'<div class="mc-delta" style="color:{delta_color};">{delta}</div>'
        if delta is not None else ""
    )
    # Inline the ::before gradient directly via a wrapper div with a top border trick
    return (
        f'<div class="pim-metric-card" style="position:relative;overflow:hidden;'
        f'background:{CARD};border-radius:12px;padding:1.1rem 1.3rem 1rem;'
        f'box-shadow:0 4px 16px rgba(0,0,0,0.4);">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:linear-gradient(90deg,{accent},{a2});border-radius:12px 12px 0 0;"></div>'
        f'<div class="mc-label">{label}</div>'
        f'<div class="mc-value">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )


def carbon_plotly_layout(**kwargs) -> dict:
    """Plotly layout — Carbon bg, cyan accent, Inter labels."""
    _font = dict(family=_SANS, size=11, color=DIM)
    defaults = dict(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=BG,
        font=_font,
        title_font=dict(family=_SANS, size=11, color=DIM, weight=500),
        legend=dict(
            bgcolor=f"rgba(22,27,34,0.95)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(family=_SANS, size=10, color=DIM),
        ),
        margin=dict(l=48, r=24, t=40, b=36),
        xaxis=dict(
            gridcolor="#1c2128",
            zerolinecolor=BORDER,
            linecolor=BORDER,
            tickfont=dict(family=_SANS, size=9, color=DIM),
            title_font=dict(family=_SANS, size=9, color=DIM),
        ),
        yaxis=dict(
            gridcolor="#1c2128",
            zerolinecolor=BORDER,
            linecolor=BORDER,
            tickfont=dict(family=_SANS, size=9, color=DIM),
            title_font=dict(family=_SANS, size=9, color=DIM),
        ),
    )
    defaults.update(kwargs)
    return defaults


def pnl_color(value: float) -> str:
    """Return CSS color string for P&L value (cyan positive, pink-red negative)."""
    return GAIN if value >= 0 else LOSS


def regime_color(regime: str) -> str:
    """Return hex color for a regime name."""
    return {
        "Risk-On":         GAIN,
        "Caution":         AMBER,
        "High Volatility": LOSS,
        "Stagflation":     "#f97316",
        "Recession":       "#dc2626",
        "Mean Reversion":  ACCENT,
        "Uncertain":       PURPLE,
        "Unknown":         DIM,
        # legacy
        "Low Vol":         GAIN,
        "High Vol":        LOSS,
        "Trending":        ACCENT,
    }.get(regime, DIM)


def page_header(title: str, subtitle: str = "", centered: bool = True) -> None:
    """
    Render a page heading with optional subtitle.
    """
    align = "center" if centered else "left"
    padding = "1.2rem 0 0.4rem 0" if centered else "0.6rem 0 0.4rem 0"
    sub_html = (
        f'<div style="font-family:{_SANS};font-size:0.75rem;font-weight:400;'
        f'color:{DIM};letter-spacing:0.12em;margin-top:0.35rem;">'
        f'{subtitle}</div>'
    ) if subtitle else ""
    st.markdown(
        f'<div style="text-align:{align};padding:{padding};">'
        f'<div style="font-family:{_SERIF};font-size:2.1rem;font-weight:400;'
        f'font-style:italic;color:{ACCENT};letter-spacing:0.01em;">{title}</div>'
        f'{sub_html}'
        f'</div>'
        f'<hr>',
        unsafe_allow_html=True,
    )


def section_header(title: str, accent: bool | str = False) -> None:
    """Render an inline section divider with label.

    accent: True → cyan ACCENT, False → SUBTLE, or pass a hex color string directly.
    """
    if isinstance(accent, str) and accent.startswith("#"):
        color = accent
    else:
        color = ACCENT if accent else DIM
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.75rem;'
        f'margin:1.5rem 0 0.75rem 0;">'
        f'<span style="font-family:{_SANS};font-size:0.65rem;font-weight:600;'
        f'color:{color};letter-spacing:0.16em;text-transform:uppercase;">{title}</span>'
        f'<div style="flex:1;height:1px;background:linear-gradient(to right,'
        f'{BORDER2},transparent);"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, color: str = ACCENT) -> str:
    """Return an inline HTML badge span."""
    bg = color + "18"
    return (
        f'<span style="display:inline-block;padding:0.15rem 0.5rem;border-radius:6px;'
        f'background:{bg};border:1px solid {color}30;font-family:{_SANS};'
        f'font-size:0.62rem;font-weight:600;letter-spacing:0.08em;'
        f'text-transform:uppercase;color:{color};">{text}</span>'
    )


def signal_badge(action: str) -> str:
    """Return a colored badge for BUY / SELL / HOLD / REDUCE / WATCH signals."""
    colors = {
        "BUY":    GREEN,
        "SELL":   LOSS,
        "HOLD":   DIM,
        "REDUCE": AMBER,
        "WATCH":  PURPLE,
    }
    return badge(action, colors.get(action.upper(), DIM))


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """
    Render a compact KPI strip using st.columns.

    items : list of (label, value, delta_or_None)
    """
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label=label, value=value, delta=delta)


def html_table(rows_html: str, head_html: str, margin_bottom: str = "1.5rem") -> str:
    """Return a styled HTML table string for use with st.markdown."""
    return (
        f'<div style="border-radius:12px;overflow:hidden;'
        f'box-shadow:0 4px 16px rgba(0,0,0,0.4);margin-bottom:{margin_bottom};">'
        f'<div style="overflow-x:auto;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>{head_html}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div></div>'
    )


def info_card(title: str, body: str, accent_color: str = ACCENT) -> None:
    """Render an elevated info card with a colored top accent bar."""
    st.markdown(
        f'<div style="border-radius:12px;background:{CARD};padding:1rem 1.2rem;'
        f'margin-bottom:0.75rem;box-shadow:0 4px 16px rgba(0,0,0,0.4);'
        f'border-top:3px solid {accent_color};position:relative;overflow:hidden;">'
        f'<div style="font-family:{_SANS};font-size:0.66rem;font-weight:600;'
        f'letter-spacing:0.12em;text-transform:uppercase;color:{accent_color};'
        f'margin-bottom:0.4rem;">{title}</div>'
        f'<div style="font-family:{_SANS};font-size:0.84rem;color:{DIM};'
        f'line-height:1.65;">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Flex-div table ─────────────────────────────────────────────────────────────

def flex_table(
    df: "pd.DataFrame",
    columns: list,
    key: str = "tbl",
    row_height: int = 44,
    max_rows: int = 500,
) -> None:
    """
    Render a styled flex-div table with client-side click-to-sort.

    Parameters
    ----------
    df       : DataFrame to display.
    columns  : List of column-spec dicts with keys:
                 key          – DataFrame column name
                 label        – Header display text
                 width        – CSS width string, e.g. "20%", "120px"
                 align        – "left" | "right" | "center"  (default "left")
                 fmt          – callable(raw_val) -> str, or None
                 numeric      – bool; auto-detected from dtype if omitted
                 color_scale  – None | "rg" (red=low green=high)
                                        | "gr" (green=low red=high)
    key      : Stable unique ID string; used to deduplicate iframes on the page.
    row_height : Pixel height per data row (default 44).
    max_rows : Truncation limit (default 500).
    """
    import json
    import math
    import streamlit.components.v1 as _components
    import pandas as _pd

    HDR_H   = 42   # px — header row height
    IFRAME_PAD = 4  # px — bottom buffer to avoid scrollbar flash

    sub = df.head(max_rows)

    # ── Build serialisable rows ────────────────────────────────────────────────
    col_minmax: dict[str, tuple] = {}
    for c in columns:
        if c.get("color_scale"):
            k = c["key"]
            if k in sub.columns:
                vals = _pd.to_numeric(sub[k], errors="coerce").dropna()
                if len(vals):
                    col_minmax[k] = (float(vals.min()), float(vals.max()))

    rows_data = []
    for _, row in sub.iterrows():
        cells = []
        for c in columns:
            raw = row.get(c["key"])
            # display string
            fmt = c.get("fmt")
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                display = "—"
                sort_val = None
            elif fmt:
                display = fmt(raw)
                sort_val = raw
            else:
                display = str(raw)
                sort_val = raw

            # numeric detection for sort
            is_num = c.get("numeric")
            if is_num is None and c["key"] in sub.columns:
                is_num = _pd.api.types.is_numeric_dtype(sub[c["key"]])
            if is_num and sort_val is not None:
                try:
                    sort_val = float(sort_val)
                except (TypeError, ValueError):
                    is_num = False

            # color scale value
            cs_val = None
            if c.get("color_scale") and c["key"] in col_minmax:
                try:
                    cs_val = float(raw) if raw is not None else None
                except (TypeError, ValueError):
                    pass

            cells.append({
                "d": display,
                "s": sort_val if sort_val is not None else display,
                "n": bool(is_num),
                "cv": cs_val,
            })
        rows_data.append(cells)

    cols_meta = [
        {
            "label": c["label"],
            "width": c["width"],
            "align": c.get("align", "left"),
            "cs":    c.get("color_scale"),
            "ck":    c["key"],
        }
        for c in columns
    ]

    # JSON — use separators to keep payload tight
    rows_json   = json.dumps(rows_data,   separators=(",", ":"), default=str)
    cols_json   = json.dumps(cols_meta,   separators=(",", ":"))
    minmax_json = json.dumps(col_minmax,  separators=(",", ":"))

    n_rows  = len(sub)
    height  = HDR_H + n_rows * row_height + IFRAME_PAD

    # ── HTML ──────────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}}
.ft{{border:1px solid #30363d;border-radius:10px;overflow:hidden;background:#161b22;width:100%;}}
.ft-hdr{{display:flex;border-bottom:1px solid #30363d;background:#1c2128;}}
.ft-hc{{
  padding:0 1rem;height:42px;display:flex;align-items:center;gap:4px;
  font-size:.60rem;text-transform:uppercase;letter-spacing:.12em;
  color:#22d3ee;font-weight:600;cursor:pointer;user-select:none;
  white-space:nowrap;flex-shrink:0;transition:color .15s;
}}
.ft-hc:hover{{color:#67e8f9}}
.ft-hc.r{{justify-content:flex-end}}
.ft-hc.c{{justify-content:center}}
.si{{font-size:.6rem;color:#484f58}}
.ft-hc.act .si{{color:#22d3ee}}
.ft-row{{display:flex;border-bottom:1px solid #21262d;transition:background .1s;}}
.ft-row:last-child{{border-bottom:none}}
.ft-row:hover{{background:#1e2530}}
.ft-td{{
  padding:0 1rem;height:{row_height}px;display:flex;align-items:center;
  font-size:.86rem;color:#e6edf3;flex-shrink:0;overflow:hidden;
  white-space:nowrap;text-overflow:ellipsis;
}}
.ft-td.r{{justify-content:flex-end}}
.ft-td.c{{justify-content:center}}
.cs-cell{{
  border-radius:4px;margin:6px 8px;height:{row_height-14}px;
  padding:0 8px;display:flex;align-items:center;
}}
</style></head><body>
<div class="ft" id="ft_{key}">
  <div class="ft-hdr" id="ft_{key}_h"></div>
  <div id="ft_{key}_b"></div>
</div>
<script>
(function(){{
var ROWS={rows_json};
var COLS={cols_json};
var MM={minmax_json};
var sortCI=-1,sortAsc=true;

function hsl(t,scale){{
  // t in [0,1]; scale "rg": red(0)→green(120); "gr": green(120)→red(0)
  var h=scale==="rg"?t*120:(1-t)*120;
  return{{bg:"hsla("+h+",38%,16%,.95)",fg:"hsla("+h+",55%,72%,1)"}};
}}

function render(rows){{
  // --- header ---
  var hdr=document.getElementById("ft_{key}_h");
  hdr.innerHTML="";
  COLS.forEach(function(c,ci){{
    var el=document.createElement("div");
    var al=c.align==="right"?"r":c.align==="center"?"c":"";
    el.className="ft-hc"+(al?" "+al:"")+(sortCI===ci?" act":"");
    el.style.width=c.width;
    var ind=sortCI===ci?(sortAsc?" ↑":" ↓"):" ↕";
    el.innerHTML=c.label+'<span class="si">'+ind+'</span>';
    (function(ci2){{
      el.addEventListener("click",function(){{
        if(sortCI===ci2)sortAsc=!sortAsc;
        else{{sortCI=ci2;sortAsc=true;}}
        function _toNum(v){{
          var s=String(v).replace(/[$,%\s]/g,"");
          return /^[+\-]?\d+\.?\d*([eE][+\-]?\d+)?$/.test(s)?parseFloat(s):NaN;
        }}
        var s=rows.slice().sort(function(a,b){{
          var av=a[ci2].s,bv=b[ci2].s;
          if(a[ci2].n&&b[ci2].n)return sortAsc?av-bv:bv-av;
          var na=_toNum(av),nb=_toNum(bv);
          if(!isNaN(na)&&!isNaN(nb))return sortAsc?na-nb:nb-na;
          av=String(av).toLowerCase();bv=String(bv).toLowerCase();
          return sortAsc?av.localeCompare(bv):bv.localeCompare(av);
        }});
        render(s);
      }});
    }})(ci);
    hdr.appendChild(el);
  }});

  // --- body ---
  var body=document.getElementById("ft_{key}_b");
  body.innerHTML="";
  rows.forEach(function(row){{
    var rEl=document.createElement("div");
    rEl.className="ft-row";
    row.forEach(function(cell,ci){{
      var c=COLS[ci];
      var al=c.align==="right"?"r":c.align==="center"?"c":"";
      var cEl=document.createElement("div");
      // color scale cell
      if(c.cs&&cell.cv!==null&&MM[c.ck]){{
        var mm=MM[c.ck];
        var mn=mm[0],mx=mm[1];
        var t=(mn===mx)?0.5:(cell.cv-mn)/(mx-mn);
        var clr=hsl(t,c.cs);
        cEl.className="ft-td cs-cell"+(al?" "+al:"");
        cEl.style.cssText+="background:"+clr.bg+";color:"+clr.fg+
          ";width:calc("+c.width+" - 16px)";
      }}else{{
        cEl.className="ft-td"+(al?" "+al:"");
        cEl.style.width=c.width;
      }}
      cEl.textContent=cell.d;
      rEl.appendChild(cEl);
    }});
    body.appendChild(rEl);
  }});
}}

render(ROWS);
}})();
</script></body></html>"""

    _components.html(html, height=height, scrolling=False)


# ── Alert System ──────────────────────────────────────────────────────────────

def regime_alert_banner(current_regime: str) -> None:
    """
    Show a dismissible banner when the detected regime differs from the last
    one stored in session state.  Call after apply_carbon_theme() on any page
    that runs the regime detector.
    """
    if not current_regime:
        return

    _prev = st.session_state.get("_last_known_regime")

    if _prev is not None and _prev != current_regime:
        rc = regime_color(current_regime)
        st.markdown(
            f'<div style="background:{rc}18;border:2px solid {rc}66;border-radius:10px;'
            f'padding:14px 20px;margin-bottom:16px;">'
            f'<span style="font-size:11px;text-transform:uppercase;letter-spacing:0.12em;color:{rc};">Regime Change Detected</span>'
            f'<div style="font-size:18px;font-weight:700;color:{rc};margin-top:4px;">'
            f'{_prev} &nbsp;→&nbsp; {current_regime}</div>'
            f'<div style="font-size:0.82rem;color:#cccccc;margin-top:6px;">'
            f'The market regime has shifted. Review your allocation, risk limits, and strategy selection.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.session_state["_last_known_regime"] = current_regime


def position_alert_banners(positions_df, current_prices: dict) -> None:
    """
    Show inline alerts for positions with large drawdowns (>10%) or
    positions worth less than 50% of cost basis.
    positions_df must have columns: ticker, shares, cost_basis.
    """
    if positions_df is None or positions_df.empty:
        return

    alerts = []
    for _, row in positions_df.iterrows():
        t  = str(row.get("ticker", ""))
        cb = float(row.get("cost_basis", 0) or 0)
        px = float(current_prices.get(t, 0) or 0)
        if cb > 0 and px > 0:
            chg = (px - cb) / cb * 100
            if chg <= -20:
                alerts.append((t, chg, LOSS, "Major drawdown"))
            elif chg <= -10:
                alerts.append((t, chg, AMBER, "Significant drawdown"))

    for t, chg, color, label in alerts:
        st.markdown(
            f'<div style="background:{color}12;border-left:3px solid {color};'
            f'padding:8px 14px;margin-bottom:6px;border-radius:4px;">'
            f'<span style="color:{color};font-weight:600;">{label}</span>'
            f' — <span style="color:#e6edf3;">{t}</span>'
            f' is <span style="color:{color};">{chg:+.1f}%</span> from cost basis.'
            f'</div>',
            unsafe_allow_html=True,
        )


def macro_event_banner(events: list[dict]) -> None:
    """
    Show upcoming macro events (FOMC, CPI, NFP) as amber info banners.
    events: list of dicts with keys: name (str), days_away (int).
    """
    imminent = [e for e in events if 0 <= e.get("days_away", 999) <= 3]
    upcoming = [e for e in events if 3 < e.get("days_away", 999) <= 7]

    for e in imminent:
        label = "TODAY" if e["days_away"] == 0 else (
            "TOMORROW" if e["days_away"] == 1 else f"IN {e['days_away']} DAYS"
        )
        st.markdown(
            f'<div style="background:{AMBER}18;border:1px solid {AMBER}66;border-radius:8px;'
            f'padding:10px 16px;margin-bottom:8px;">'
            f'<span style="font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:{AMBER};">Event {label}</span>'
            f'<div style="font-size:16px;font-weight:600;color:{AMBER};margin-top:2px;">{e["name"]}</div>'
            f'<div style="font-size:0.80rem;color:#cccccc;">Consider reducing position size or hedging before this release.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    for e in upcoming:
        st.markdown(
            f'<div style="background:{SUBTLE}18;border-left:3px solid {AMBER}88;'
            f'padding:8px 14px;margin-bottom:6px;border-radius:4px;font-size:0.82rem;">'
            f'<span style="color:{AMBER};">Upcoming:</span> '
            f'<span style="color:#cccccc;">{e["name"]}</span> in {e["days_away"]} days.'
            f'</div>',
            unsafe_allow_html=True,
        )
