"""
Top navigation bar and global portfolio header.

Usage (every page, after st.set_page_config() and apply_carbon_theme()):
    from utils.navigation import render_navbar, render_global_header
    render_navbar(current_page="Market Regime Dashboard")
    render_global_header()
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, time as dtime
from pathlib import Path
import sys
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    import pytz
    _ET = pytz.timezone("America/New_York")

sys.path.append(str(Path(__file__).parent.parent))

from utils.portfolio_store import (
    load_portfolio, get_last_saved_time,
    load_options_positions, portfolio_file_exists,
)
from utils.carbon_theme import BG, CARD, BORDER, FG, DIM, SUBTLE, ACCENT, GAIN, LOSS, AMBER

# ── palette shortcuts ─────────────────────────────────────────────────────────
_P = "'Palatino Linotype', 'Book Antiqua', Palatino, serif"

# ── Navigation structure ──────────────────────────────────────────────────────
NAV_GROUPS = [
    ("PORTFOLIO", [
        ("Home",                    "/"),
        ("Position Scoring",        "/Position_Scoring"),
        ("Optimization",            "/Optimization_Rebalancing"),
        ("Monte Carlo",             "/Monte_Carlo"),
        ("Performance Attribution", "/Performance_Attribution"),
    ]),
    ("MARKET INTEL", [
        ("Market Regime",    "/Market_Regime_Dashboard"),
        ("Action Dashboard", "/Action_Dashboard"),
        ("Trade Suggestions","/Trade_Suggestions"),
        ("Risk Dashboard",   "/Risk_Dashboard"),
    ]),
    ("OPTIONS", [
        ("Options Analytics",  "/Options_Analytics"),
        ("Live Chain",         "/Live_Options_Chain"),
        ("Portfolio Hedges",   "/Portfolio_Hedges"),
        ("Vol Surface",        "/Vol_Surface"),
        ("Options Flow",       "/Options_Flow"),
    ]),
    ("RESEARCH & AI", [
        ("Stock Screener",       "/Stock_Screener"),
        ("Watchlist & Alerts",   "/Watchlist_Alerts"),
        ("AI Assistant",         "/AI_Assistant"),
    ]),
]

# ── CSS: hide sidebar + Streamlit chrome, style navbar ───────────────────────
_NAV_CSS = f"""
<style>
/* ── Hide Streamlit chrome ──────────────────────────────────────────── */
header[data-testid="stHeader"]          {{ display: none !important; }}
section[data-testid="stSidebar"]        {{ display: none !important; }}
[data-testid="collapsedControl"]        {{ display: none !important; }}
[data-testid="stSidebarNav"]            {{ display: none !important; }}
#MainMenu                               {{ display: none !important; }}
footer                                  {{ display: none !important; }}

/* Remove top-padding that Streamlit reserves for its header */
.main .block-container,
[data-testid="stMainBlockContainer"] {{
    padding-top: 0 !important;
    margin-top:  0 !important;
}}

/* ── Navbar shell ─────────────────────────────────────────────────── */
.pm-navbar {{
    position: sticky;
    top: 0;
    z-index: 9999;
    background: {BG};
    border-bottom: 1px solid {BORDER};
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1.5rem;
    height: 56px;
    width: 100%;
    box-sizing: border-box;
    font-family: {_P};
}}

/* ── Logo / brand ─────────────────────────────────────────────────── */
.pm-brand {{
    font-family: {_P};
    font-size: 1.65rem;
    font-style: italic;
    font-weight: 400;
    color: {ACCENT};
    letter-spacing: 0.01em;
    white-space: nowrap;
    text-decoration: none;
    line-height: 1;
}}

/* ── Right-hand nav groups container ─────────────────────────────── */
.pm-nav-right {{
    display: flex;
    align-items: center;
    gap: 0.15rem;
    height: 100%;
}}

/* ── Each category group ─────────────────────────────────────────── */
.pm-nav-group {{
    position: relative;
    height: 100%;
    display: flex;
    align-items: center;
}}

/* ── Category label button ───────────────────────────────────────── */
.pm-nav-cat {{
    font-family: {_P};
    font-size: 0.65rem;
    font-weight: 500;
    font-style: normal;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {SUBTLE};
    padding: 0 0.75rem;
    height: 100%;
    display: flex;
    align-items: center;
    cursor: default;
    white-space: nowrap;
    transition: color 0.15s;
    border-bottom: 2px solid transparent;
}}
.pm-nav-cat:hover {{
    color: {FG};
}}
.pm-nav-cat.pm-active {{
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}

/* ── Dropdown panel ──────────────────────────────────────────────── */
.pm-dropdown {{
    display: none;
    position: absolute;
    top: 56px;
    right: 0;
    min-width: 190px;
    background: {CARD};
    border: 1px solid {BORDER};
    border-top: none;
    z-index: 10000;
    padding: 0.4rem 0;
}}
.pm-nav-group:hover .pm-dropdown {{
    display: block;
}}

/* ── Dropdown links ──────────────────────────────────────────────── */
.pm-dropdown a {{
    display: block;
    font-family: {_P};
    font-size: 0.78rem;
    color: {SUBTLE};
    text-decoration: none;
    padding: 0.45rem 1.1rem;
    white-space: nowrap;
    transition: background 0.1s, color 0.1s;
}}
.pm-dropdown a:hover {{
    background: {BORDER};
    color: {FG};
}}
.pm-dropdown a.pm-active-link {{
    color: {ACCENT};
    background: rgba(34,211,238,0.06);
}}

/* ── Global header bar ───────────────────────────────────────────── */
.pm-header {{
    background: {CARD};
    border-bottom: 1px solid {BORDER};
    display: flex;
    align-items: center;
    padding: 0 1.5rem;
    height: 38px;
    gap: 1.8rem;
    font-family: {_P};
    font-size: 0.75rem;
}}
.pm-header-stat {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    color: {SUBTLE};
    white-space: nowrap;
}}
.pm-header-label {{
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: {DIM};
}}
.pm-header-value {{
    font-size: 0.82rem;
    color: {FG};
}}
.pm-gain  {{ color: {GAIN}  !important; }}
.pm-loss  {{ color: {LOSS}  !important; }}
.pm-amber {{ color: {AMBER} !important; }}
.pm-dim   {{ color: {DIM}   !important; }}

/* ── Regime badge ────────────────────────────────────────────────── */
.pm-regime-badge {{
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    border: 1px solid currentColor;
}}

/* ── Inline controls bar ─────────────────────────────────────────── */
.pm-controls {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 0.65rem 1rem;
    margin-bottom: 1.2rem;
}}

/* ══ DENSITY — 12 px baseline, tight spacing ═════════════════════════
   Applies globally to every page via render_navbar().
   ══════════════════════════════════════════════════════════════════ */

/* Block container — pull in padding */
.main .block-container,
[data-testid="stMainBlockContainer"] {{
    padding-top:    0.6rem !important;
    padding-left:   0.9rem !important;
    padding-right:  0.9rem !important;
    padding-bottom: 0.8rem !important;
}}

/* Base prose */
.stMarkdown p,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
    font-size: 12px !important;
    line-height: 1.5 !important;
    margin-bottom: 0.3rem !important;
}}

/* st.subheader / h3 */
[data-testid="stMarkdownContainer"] h3,
h3 {{
    font-size: 0.82rem !important;
    margin: 0.55rem 0 0.25rem !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
    padding: 0.4rem 0.6rem !important;
}}
[data-testid="stMetricLabel"] p {{
    font-size: 0.62rem !important;
    margin-bottom: 0 !important;
}}
[data-testid="stMetricValue"] > div {{
    font-size: 1.05rem !important;
    line-height: 1.3 !important;
}}
[data-testid="stMetricDelta"] > div {{
    font-size: 0.68rem !important;
}}

/* Buttons */
[data-testid="stButton"] > button,
[data-testid="stFormSubmitButton"] > button {{
    padding: 0.18rem 0.65rem !important;
    font-size: 0.73rem !important;
    min-height: 28px !important;
    line-height: 1.4 !important;
}}

/* Text + number inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {{
    font-size: 0.78rem !important;
    padding: 0.22rem 0.5rem !important;
    height: 30px !important;
    line-height: 1.4 !important;
}}

/* Select / radio / checkbox labels */
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label,
[data-testid="stMultiSelect"] label {{
    font-size: 0.72rem !important;
}}
[data-testid="stSelectbox"] > div > div {{
    font-size: 0.78rem !important;
    min-height: 30px !important;
}}

/* Column gaps */
[data-testid="stHorizontalBlock"] {{
    gap: 0.45rem !important;
    align-items: flex-start !important;
}}

/* Expander */
[data-testid="stExpander"] summary {{
    font-size: 0.78rem !important;
    padding: 0.32rem 0.65rem !important;
}}
[data-testid="stExpander"] > details > div {{
    padding: 0.5rem 0.65rem !important;
}}

/* Tabs */
[data-testid="stTab"] p {{
    font-size: 0.74rem !important;
}}
button[data-testid="stTab"] {{
    padding: 0.3rem 0.7rem !important;
}}

/* Divider */
[data-testid="stDivider"] {{
    margin: 0.4rem 0 !important;
}}

/* Captions */
[data-testid="stCaptionContainer"] p {{
    font-size: 0.65rem !important;
    line-height: 1.4 !important;
}}

/* Dataframe / table text */
[data-testid="stDataFrame"] {{
    font-size: 11px !important;
}}

/* Spinner text */
[data-testid="stSpinner"] p {{
    font-size: 0.72rem !important;
}}

/* ══ PRINT CSS — browser ⌘P export ═══════════════════════════════════ */
@media print {{
    /* Hide chrome entirely */
    .pm-navbar, .pm-header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stSidebar"],
    header, footer {{
        display: none !important;
    }}
    /* White page */
    .stApp,
    [data-testid="stAppViewContainer"],
    .main .block-container,
    [data-testid="stMainBlockContainer"] {{
        background: #ffffff !important;
        color: #111111 !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
    }}
    /* Force text dark */
    * {{ color: #111111 !important; }}
    /* Break cleanly between major sections */
    [data-testid="stExpander"],
    [data-testid="stPlotlyChart"] {{
        page-break-inside: avoid;
        break-inside: avoid;
    }}
    /* Show full tables */
    table {{ page-break-inside: auto; }}
    tr {{ page-break-inside: avoid; }}
}}

/* ══ FLATTEN THICK BORDERS (global) ══════════════════════════════════
   Remove Streamlit's default heavy colored left-border on alert boxes,
   expanders, forms, and inputs. Replace with 1 px carbon border.
   ══════════════════════════════════════════════════════════════════ */

/* Alert boxes: info / warning / error / success */
[data-testid="stAlert"],
div[data-testid="stAlert"] {{
    border: 1px solid {BORDER} !important;
    border-left: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    background-color: {CARD} !important;
    box-shadow: none !important;
    padding: 0.6rem 0.85rem !important;
}}
/* The inner wrapper that sometimes carries the thick left rule */
[data-testid="stAlert"] > div,
[data-testid="stAlert"] > div > div {{
    border-left: none !important;
    box-shadow: none !important;
}}
/* Icon colour — keep subtle, not the vivid blue/red/green */
[data-testid="stAlert"] svg {{
    color: {DIM} !important;
    fill: {DIM} !important;
}}

/* Expander */
[data-testid="stExpander"],
[data-testid="stExpander"] > details,
[data-testid="stExpander"] > details > summary {{
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    background: {CARD} !important;
}}
[data-testid="stExpander"] > details > div {{
    border-top: 1px solid {BORDER} !important;
    box-shadow: none !important;
}}

/* Form containers */
[data-testid="stForm"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0.75rem !important;
}}

/* Text / number / selectbox inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {{
    border: 1px solid {BORDER} !important;
    border-radius: 3px !important;
    box-shadow: none !important;
    background-color: {CARD} !important;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {{
    border: 1px solid {ACCENT} !important;
    box-shadow: 0 0 0 1px {ACCENT}33 !important;
    outline: none !important;
}}

/* File uploader */
[data-testid="stFileUploader"] > div {{
    border: 1px dashed {BORDER} !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    background: {CARD} !important;
}}

/* Tabs underline — keep accent, but only 2 px */
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {{
    border-bottom: 2px solid {ACCENT} !important;
    box-shadow: none !important;
}}
[data-testid="stTabs"] [data-testid="stTab"] {{
    border: none !important;
    box-shadow: none !important;
}}

/* Metric / dataframe containers */
[data-testid="stMetric"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    background: {CARD} !important;
    box-shadow: none !important;
    padding: 0.5rem 0.75rem !important;
}}

/* Generic .element-container borders sometimes appear */
div.stMarkdown [style*="border"],
div[style*="border: 2px"],
div[style*="border:2px"] {{
    border-width: 1px !important;
}}
</style>
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _active_group(current_page: str) -> str:
    """Return the category label that contains current_page."""
    for cat, pages in NAV_GROUPS:
        for label, _ in pages:
            if label == current_page:
                return cat
    return ""


def _build_navbar_html(current_page: str) -> str:
    """Build the full navbar + header HTML string."""
    active_cat = _active_group(current_page)

    groups_html = ""
    for cat, pages in NAV_GROUPS:
        cat_active = "pm-active" if cat == active_cat else ""
        links_html = ""
        for label, url in pages:
            link_active = "pm-active-link" if label == current_page else ""
            links_html += (
                f'<a href="{url}" target="_self" class="{link_active}">{label}</a>'
            )
        groups_html += (
            f'<div class="pm-nav-group">'
            f'  <span class="pm-nav-cat {cat_active}">{cat}</span>'
            f'  <div class="pm-dropdown">{links_html}</div>'
            f'</div>'
        )

    return (
        f'<div class="pm-navbar">'
        f'  <a class="pm-brand" href="/" target="_self">Portfolio Manager</a>'
        f'  <div class="pm-nav-right">{groups_html}</div>'
        f'</div>'
    )


# ── SVG favicon (injected once per page via render_navbar) ────────────────────
_FAVICON_SVG = (
    "data:image/svg+xml,"
    "%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2032%2032%27%3E"
    "%3Crect%20width%3D%2732%27%20height%3D%2732%27%20rx%3D%273%27%20fill%3D%27%23111111%27/%3E"
    "%3Crect%20x%3D%273%27%20y%3D%2721%27%20width%3D%275%27%20height%3D%278%27%20rx%3D%271%27%20fill%3D%27%2322d3ee%27%20opacity%3D%270.55%27/%3E"
    "%3Crect%20x%3D%2710%27%20y%3D%2715%27%20width%3D%275%27%20height%3D%2714%27%20rx%3D%271%27%20fill%3D%27%2322d3ee%27%20opacity%3D%270.75%27/%3E"
    "%3Crect%20x%3D%2717%27%20y%3D%279%27%20width%3D%275%27%20height%3D%2720%27%20rx%3D%271%27%20fill%3D%27%2322d3ee%27/%3E"
    "%3Crect%20x%3D%2724%27%20y%3D%2717%27%20width%3D%275%27%20height%3D%2712%27%20rx%3D%271%27%20fill%3D%27%2322d3ee%27%20opacity%3D%270.65%27/%3E"
    "%3Cpolyline%20points%3D%275.5%2C21%2012.5%2C15%2019.5%2C9%2026.5%2C17%27%20"
    "fill%3D%27none%27%20stroke%3D%27%2322d3ee%27%20stroke-width%3D%271.5%27%20"
    "stroke-linecap%3D%27round%27%20stroke-linejoin%3D%27round%27/%3E"
    "%3C/svg%3E"
)

# ── Market session detector ───────────────────────────────────────────────────

def _market_session() -> tuple[str, str]:
    """Return (label, hex_color) for the current US equity session."""
    try:
        now = datetime.now(_ET)
        wd  = now.weekday()      # 0 = Mon
        t   = now.time()
        if wd >= 5:
            return "Closed", DIM
        pre   = dtime(4,  0)
        open_ = dtime(9, 30)
        close = dtime(16, 0)
        aend  = dtime(20, 0)
        if t < pre or t >= aend:
            return "Closed",      DIM
        if t < open_:
            return "Pre-Market",  AMBER
        if t < close:
            return "Market Open", GAIN
        return "After Hours",     AMBER
    except Exception:
        return "—", DIM


# ── Price fetching (cached 5 min) ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_header_prices(tickers: tuple) -> dict:
    """Fetch current + previous close for a set of tickers."""
    result: dict = {}
    if not tickers:
        return result
    try:
        raw = yf.download(list(tickers), period="2d", progress=False, auto_adjust=True)
        if raw.empty:
            return result
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw.rename(columns={"Close": tickers[0]})[list(tickers)]
        for t in tickers:
            if t not in close.columns:
                continue
            s = close[t].dropna()
            if len(s) >= 2:
                result[t] = {"cur": float(s.iloc[-1]), "prev": float(s.iloc[-2])}
            elif len(s) == 1:
                result[t] = {"cur": float(s.iloc[0]), "prev": float(s.iloc[0])}
    except Exception:
        pass
    return result


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_market_indices() -> dict:
    """Fetch SPY, QQQ, ^VIX, ^TNX — cached 2 min."""
    result: dict = {}
    symbols = {"SPY": "SPY", "QQQ": "QQQ", "VIX": "^VIX", "10Y": "^TNX"}
    try:
        raw = yf.download(
            list(symbols.values()), period="2d",
            progress=False, auto_adjust=True,
        )
        if raw.empty:
            return result
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        for label, sym in symbols.items():
            if sym not in close.columns:
                continue
            s = close[sym].dropna()
            if len(s) >= 2:
                cur, prev = float(s.iloc[-1]), float(s.iloc[-2])
                result[label] = {
                    "cur": cur,
                    "chg_pct": (cur - prev) / prev * 100 if prev else 0.0,
                }
            elif len(s) == 1:
                result[label] = {"cur": float(s.iloc[0]), "chg_pct": 0.0}
    except Exception:
        pass
    return result


def _build_header_html() -> str:
    """Build the global portfolio stats + market indices header HTML."""
    # ── Portfolio stats ───────────────────────────────────────────────────────
    total_val_str = "—"
    pnl_str       = "—"
    pnl_class     = "pm-dim"
    regime_str    = st.session_state.get("current_regime", "—")
    regime_color  = {
        "Risk-On": GAIN, "Caution": AMBER,
        "High Volatility": LOSS, "Stagflation": "#f97316",
        "Recession": "#dc2626", "Mean Reversion": ACCENT,
        "Uncertain": PURPLE,
        # legacy
        "Low Vol": GAIN, "Trending": ACCENT,
        "High Vol": LOSS,
    }.get(regime_str, DIM)
    updated_str = "—"

    portfolio_df = load_portfolio()
    if portfolio_df is not None and not portfolio_df.empty:
        try:
            tickers = tuple(portfolio_df["ticker"].dropna().unique().tolist())
            prices  = _fetch_header_prices(tickers)
            total_val = total_prev = 0.0
            for _, row in portfolio_df.iterrows():
                t      = row["ticker"]
                sh     = float(row.get("shares", 0) or 0)
                cost   = float(row.get("cost_basis", 0) or 0)
                pi     = prices.get(t, {})
                total_val  += sh * pi.get("cur",  cost)
                total_prev += sh * pi.get("prev", cost)
            day_pnl     = total_val - total_prev
            day_pnl_pct = (day_pnl / total_prev * 100) if total_prev else 0.0
            total_val_str = f"${total_val:,.0f}"
            sign          = "+" if day_pnl >= 0 else ""
            pnl_str       = f"{sign}${day_pnl:,.0f} ({sign}{day_pnl_pct:.2f}%)"
            pnl_class     = "pm-gain" if day_pnl >= 0 else "pm-loss"
        except Exception:
            pass
        saved = get_last_saved_time()
        if saved:
            try:
                updated_str = datetime.fromisoformat(saved).strftime("%-I:%M %p")
            except Exception:
                pass

    # ── Market indices ────────────────────────────────────────────────────────
    indices     = _fetch_market_indices()
    idx_html    = ""
    sep         = f'<span style="color:{BORDER};margin:0 0.15rem;">|</span>'

    for label, fmt_fn in [
        ("SPY",  lambda d: (f"SPY {d['cur']:.2f}",   d["chg_pct"])),
        ("QQQ",  lambda d: (f"QQQ {d['cur']:.2f}",   d["chg_pct"])),
        ("VIX",  lambda d: (f"VIX {d['cur']:.1f}",   d["chg_pct"])),
        ("10Y",  lambda d: (f"10Y {d['cur']:.2f}%",  d["chg_pct"])),
    ]:
        d = indices.get(label)
        if not d:
            continue
        lbl, chg = fmt_fn(d)
        c_class   = "pm-gain" if chg > 0 else ("pm-loss" if chg < 0 else "pm-dim")
        arrow     = "▲" if chg > 0 else ("▼" if chg < 0 else "—")
        idx_html += (
            f'{sep}'
            f'<div class="pm-header-stat">'
            f'<span class="pm-header-value" style="font-size:0.72rem;">{lbl}</span>'
            f'<span class="{c_class}" style="font-size:0.65rem;">'
            f'  {arrow}{abs(chg):.2f}%</span>'
            f'</div>'
        )

    # ── Session badge ─────────────────────────────────────────────────────────
    sess_label, sess_color = _market_session()
    session_badge = (
        f'<span class="pm-regime-badge" '
        f'style="color:{sess_color};border-color:{sess_color};font-size:0.58rem;">'
        f'{sess_label}</span>'
    )

    # ── Regime badge ──────────────────────────────────────────────────────────
    regime_badge = (
        f'<span class="pm-regime-badge" style="color:{regime_color};border-color:{regime_color};">'
        f'{regime_str}</span>'
    ) if regime_str != "—" else f'<span class="pm-dim" style="font-size:0.65rem;">No Regime</span>'

    return f"""
<div class="pm-header">
  <div class="pm-header-stat">
    <span class="pm-header-label">Portfolio</span>
    <span class="pm-header-value">{total_val_str}</span>
  </div>
  <div class="pm-header-stat">
    <span class="pm-header-label">Day P&amp;L</span>
    <span class="pm-header-value {pnl_class}">{pnl_str}</span>
  </div>
  {idx_html}
  {sep}
  <div class="pm-header-stat">{session_badge}</div>
  <div class="pm-header-stat" style="margin-left:auto;">
    <span class="pm-header-label">Regime</span>
    {regime_badge}
  </div>
  <div class="pm-header-stat" style="padding-left:0.5rem;">
    <span class="pm-header-label">Updated</span>
    <span class="pm-dim" style="font-size:0.65rem;">{updated_str}</span>
  </div>
</div>
"""


# ── Session restore ───────────────────────────────────────────────────────────

def _auto_restore_session() -> None:
    """
    Reload portfolio + options from disk into session state whenever session
    state has been wiped (e.g. hard browser navigation between pages).
    This is a no-op when session state already contains data.
    """
    if ('positions' not in st.session_state or st.session_state['positions'] is None) \
            and portfolio_file_exists():
        loaded = load_portfolio()
        if loaded is not None and not loaded.empty:
            st.session_state['positions'] = loaded
            st.session_state['manual_positions'] = loaded.to_dict(orient='records')

    if 'options_positions' not in st.session_state:
        opts = load_options_positions()
        st.session_state['options_positions'] = opts if opts else []


# ── Public API ────────────────────────────────────────────────────────────────

def render_navbar(current_page: str = "") -> None:
    """
    Inject favicon, top navigation bar CSS + HTML.
    Call immediately after apply_carbon_theme() on every page.
    """
    _auto_restore_session()
    # Inject CSS separately so <style> is never mixed with <link>/<div> tags
    st.markdown(_NAV_CSS, unsafe_allow_html=True)
    # Favicon + navbar HTML in a separate call
    favicon_html = (
        f'<link rel="shortcut icon" href="{_FAVICON_SVG}">'
        f'<link rel="icon" type="image/svg+xml" href="{_FAVICON_SVG}">'
    )
    navbar_html = _build_navbar_html(current_page)
    st.markdown(favicon_html + navbar_html, unsafe_allow_html=True)


def render_global_header() -> None:
    """
    Render the global portfolio stats bar (value, P&L, regime, updated).
    Call after render_navbar().
    """
    st.markdown(_build_header_html(), unsafe_allow_html=True)


