"""
Regime-Aware Portfolio Manager
Section 14: Stock Screener & Fundamentals

Full financial statement analysis (income, balance sheet, cash flow),
7-dimension fundamental scoring, peer comparison, and screening.
Scores saved to session_state['fundamental_scores'] for Position Scoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.carbon_theme import apply_carbon_theme, carbon_plotly_layout, page_header

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Screener", page_icon="🔍", layout="wide")
apply_carbon_theme()
page_header(
    "🔍 Stock Screener & Fundamentals",
    "Financial statements · scoring · peer comparison · screening",
)

# ── Formatting helpers ────────────────────────────────────────────────────────

def _fv(v, d=1) -> str:
    """Format large number as $XB / $XM / $XK."""
    try:
        v = float(v)
        if np.isnan(v): return "—"
        if abs(v) >= 1e9: return f"${v/1e9:.{d}f}B"
        if abs(v) >= 1e6: return f"${v/1e6:.{d}f}M"
        if abs(v) >= 1e3: return f"${v/1e3:.{d}f}K"
        return f"${v:.{d}f}"
    except Exception:
        return "—"

def _fp(v, d=1) -> str:
    try:
        return f"{float(v)*100:.{d}f}%" if not np.isnan(float(v)) else "—"
    except Exception:
        return "—"

def _fx(v, d=1) -> str:
    try:
        return f"{float(v):.{d}f}×" if not np.isnan(float(v)) else "—"
    except Exception:
        return "—"

def _safe(v, default=np.nan):
    try:
        f = float(v)
        return f if not (np.isnan(f) or np.isinf(f)) else default
    except Exception:
        return default

def _fmt_stmt_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Trim to n periods and format values for display."""
    if df.empty:
        return df
    out = df.iloc[:, :n].copy()
    out.columns = [
        pd.Timestamp(c).strftime("%b %Y") if hasattr(c, "strftime") else str(c)
        for c in out.columns
    ]
    try:
        out = out.map(lambda x: _fv(x) if pd.notna(x) else "—")
    except AttributeError:
        out = out.applymap(lambda x: _fv(x) if pd.notna(x) else "—")
    return out

# ── Cached data ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def _statements(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    result = {}
    for attr in [
        "financials", "quarterly_financials",
        "balance_sheet", "quarterly_balance_sheet",
        "cashflow", "quarterly_cashflow",
    ]:
        try:
            df = getattr(t, attr)
            result[attr] = df if (df is not None and not df.empty) else pd.DataFrame()
        except Exception:
            result[attr] = pd.DataFrame()
    return result

@st.cache_data(ttl=86400, show_spinner=False)
def _prices(ticker: str, period: str = "5y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()

# ── Row lookup ────────────────────────────────────────────────────────────────

def _row(df: pd.DataFrame, *candidates) -> pd.Series:
    for c in candidates:
        if c in df.index:
            return df.loc[c]
    return pd.Series(dtype=float)

# ── Scoring (0–100 per dimension) ─────────────────────────────────────────────

def _clamp(v): return float(max(0.0, min(100.0, v)))

def _score_rev_growth(g):   return _clamp(30 + _safe(g, 0) * 300)
def _score_margin(m):       return _clamp(_safe(m, 0) * 300)
def _score_roe(r):          return _clamp(_safe(r, 0) * 300)
def _score_valuation(pe):
    pe = _safe(pe, np.nan)
    if np.isnan(pe) or pe <= 0: return 40.0
    return _clamp(100 - pe * 1.5)
def _score_leverage(de):
    de = _safe(de, np.nan)
    if np.isnan(de) or de < 0: return 50.0
    return _clamp(100 - de * 25)
def _score_fcf_margin(fm):  return _clamp(_safe(fm, 0) * 400)
def _score_eq(ocf_ni):
    v = _safe(ocf_ni, np.nan)
    if np.isnan(v) or v < 0: return 20.0
    return _clamp((v - 0.5) * 80)

def _compute_scores(nfo: dict, stmts: dict) -> dict:
    g   = _safe(nfo.get("revenueGrowth"), np.nan)
    nm  = _safe(nfo.get("profitMargins"), np.nan)
    roe = _safe(nfo.get("returnOnEquity"), np.nan)
    pe  = _safe(nfo.get("trailingPE"), np.nan)
    de  = _safe(nfo.get("debtToEquity"), np.nan)
    if not np.isnan(de) and de > 10:
        de = de / 100          # yfinance sometimes returns as percentage

    fcf = _safe(nfo.get("freeCashflow"), np.nan)
    rev = _safe(nfo.get("totalRevenue"), np.nan)
    fm  = (fcf / rev) if (not np.isnan(fcf) and not np.isnan(rev) and rev != 0) else np.nan

    fin = stmts.get("financials", pd.DataFrame())
    cf  = stmts.get("cashflow", pd.DataFrame())
    ocf_ni = np.nan
    if not fin.empty and not cf.empty:
        ni_s  = _row(fin, "Net Income", "Net Income Common Stockholders")
        ocf_s = _row(cf, "Operating Cash Flow",
                     "Cash Flow From Continuing Operating Activities")
        if not ni_s.empty and not ocf_s.empty:
            ni_v  = _safe(ni_s.iloc[0])
            ocf_v = _safe(ocf_s.iloc[0])
            if ni_v != 0 and not np.isnan(ni_v):
                ocf_ni = ocf_v / ni_v

    return {
        "Revenue Growth":   _score_rev_growth(g),
        "Profitability":    _score_margin(nm),
        "Return on Equity": _score_roe(roe),
        "Valuation":        _score_valuation(pe),
        "Leverage":         _score_leverage(de),
        "Cash Generation":  _score_fcf_margin(fm),
        "Earnings Quality": _score_eq(ocf_ni),
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

primary = st.sidebar.text_input("Primary Ticker", value="AAPL").upper().strip()

_positions = st.session_state.get("positions", pd.DataFrame())
_ptickers: list = []
if isinstance(_positions, pd.DataFrame) and not _positions.empty:
    _ptickers = [str(t) for t in _positions["ticker"].unique()]

_peer_defaults = [t for t in _ptickers if t != primary][:4]
comp_raw = st.sidebar.text_input(
    "Comparison / Peer Tickers (comma-separated)",
    value=", ".join(_peer_defaults),
    help="Portfolio tickers pre-populated",
)
comp_tickers = [t.strip().upper() for t in comp_raw.split(",") if t.strip()]

period_mode = st.sidebar.radio("Period", ["Annual", "Quarterly"], horizontal=True)
if period_mode == "Annual":
    lb_opts = {"1Y": 1, "3Y": 3, "5Y": 5, "Max": 10}
    n_periods = lb_opts[st.sidebar.selectbox("Lookback", list(lb_opts), index=2)]
else:
    lb_opts = {"4Q": 4, "8Q": 8, "12Q": 12, "Max": 20}
    n_periods = lb_opts[st.sidebar.selectbox("Lookback", list(lb_opts), index=1)]

# ── Load primary ticker data ──────────────────────────────────────────────────
if not primary:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

with st.spinner(f"Loading {primary}…"):
    nfo   = _info(primary)
    stmts = _statements(primary)
    px    = _prices(primary)

if not nfo:
    st.error(f"Could not fetch data for **{primary}**. Check the ticker symbol.")
    st.stop()

fin_key = "quarterly_financials" if period_mode == "Quarterly" else "financials"
bs_key  = "quarterly_balance_sheet" if period_mode == "Quarterly" else "balance_sheet"
cf_key  = "quarterly_cashflow" if period_mode == "Quarterly" else "cashflow"

fin = stmts.get(fin_key, pd.DataFrame())
bs  = stmts.get(bs_key,  pd.DataFrame())
cf  = stmts.get(cf_key,  pd.DataFrame())

def _dates(df):
    return [
        pd.Timestamp(c).strftime("%b %Y") if hasattr(c, "strftime") else str(c)
        for c in df.columns[:n_periods]
    ][::-1]

# ── Compute & persist scores ──────────────────────────────────────────────────
scores = _compute_scores(nfo, stmts)
_fs = st.session_state.get("fundamental_scores", {})
_fs[primary] = scores
st.session_state["fundamental_scores"] = _fs

# ── Company header ─────────────────────────────────────────────────────────────
name    = nfo.get("longName", primary)
sector  = nfo.get("sector", "—")
ind     = nfo.get("industry", "—")
mcap    = nfo.get("marketCap", 0)
cprice  = _safe(nfo.get("currentPrice") or nfo.get("regularMarketPrice"), 0)

st.markdown(
    f"""<div style="border:1px solid #2a3142;border-radius:10px;padding:16px 24px;
    margin-bottom:16px;background:#141922;">
    <h2 style="margin:0;color:#e2e8f0;">{name}
      <span style="font-size:17px;color:#6b7a8f;">({primary})</span></h2>
    <p style="margin:6px 0 0 0;color:#6b7a8f;">{sector} &nbsp;·&nbsp; {ind}
    &nbsp;·&nbsp; Market Cap: <strong style="color:#e2e8f0;">{_fv(mcap)}</strong>
    &nbsp;·&nbsp; Price: <strong style="color:#00d4aa;">${cprice:.2f}</strong></p>
    </div>""",
    unsafe_allow_html=True,
)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📋 Income Statement",
    "🏦 Balance Sheet",
    "💵 Cash Flow",
    "📈 Valuation & Peers",
    "🔎 Screener",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Key metrics grid ────────────────────────────────────────────────────
    st.subheader("Key Metrics")

    r1 = st.columns(6)
    r1[0].metric("Trailing P/E",  _fx(nfo.get("trailingPE")))
    r1[1].metric("Forward P/E",   _fx(nfo.get("forwardPE")))
    r1[2].metric("P/S (TTM)",     _fx(nfo.get("priceToSalesTrailingTwelveMonths")))
    r1[3].metric("P/B",           _fx(nfo.get("priceToBook")))
    r1[4].metric("EV/EBITDA",     _fx(nfo.get("enterpriseToEbitda")))
    r1[5].metric("EV/Revenue",    _fx(nfo.get("enterpriseToRevenue")))

    r2 = st.columns(6)
    r2[0].metric("Net Margin",    _fp(nfo.get("profitMargins")))
    r2[1].metric("Gross Margin",  _fp(nfo.get("grossMargins")))
    r2[2].metric("Op. Margin",    _fp(nfo.get("operatingMargins")))
    r2[3].metric("ROE",           _fp(nfo.get("returnOnEquity")))
    r2[4].metric("Rev Growth",    _fp(nfo.get("revenueGrowth")))
    r2[5].metric("EPS Growth",    _fp(nfo.get("earningsGrowth")))

    r3 = st.columns(6)
    de_raw  = _safe(nfo.get("debtToEquity"), np.nan)
    de_disp = f"{de_raw:.1f}" if not np.isnan(de_raw) else "—"
    cr_raw  = _safe(nfo.get("currentRatio"), np.nan)
    cr_disp = f"{cr_raw:.2f}" if not np.isnan(cr_raw) else "—"
    beta    = _safe(nfo.get("beta"), np.nan)
    w52_lo  = _safe(nfo.get("fiftyTwoWeekLow"), 0)
    w52_hi  = _safe(nfo.get("fiftyTwoWeekHigh"), 0)
    r3[0].metric("Debt/Equity",   de_disp)
    r3[1].metric("Current Ratio", cr_disp)
    r3[2].metric("Free Cash Flow",_fv(nfo.get("freeCashflow")))
    r3[3].metric("Op. Cash Flow", _fv(nfo.get("operatingCashflow")))
    r3[4].metric("Beta",          f"{beta:.2f}" if not np.isnan(beta) else "—")
    r3[5].metric("52W Range",     f"${w52_lo:.0f} – ${w52_hi:.0f}" if w52_hi else "—")

    st.divider()

    # ── Radar + score panel ─────────────────────────────────────────────────
    col_r, col_s = st.columns([3, 2])

    _PEER_COLORS = ["#fb7185", "#f59e0b", "#a78bfa", "#4a9eff", "#22c55e"]

    with col_r:
        dims = list(scores.keys())
        vals = list(scores.values())

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=dims + [dims[0]],
            fill="toself", fillcolor="rgba(0,212,170,0.12)",
            line=dict(color="#00d4aa", width=2.5),
            name=primary,
        ))

        for i, tk in enumerate(comp_tickers[:5]):
            with st.spinner(f"Scoring {tk}…"):
                p_nfo   = _info(tk)
                p_stmts = _statements(tk)
            p_sc   = _compute_scores(p_nfo, p_stmts)
            p_vals = list(p_sc.values())
            pc     = _PEER_COLORS[i % len(_PEER_COLORS)]
            fig_radar.add_trace(go.Scatterpolar(
                r=p_vals + [p_vals[0]], theta=dims + [dims[0]],
                fill="toself", fillcolor=f"{pc}18",
                line=dict(color=pc, width=1.5, dash="dot"),
                name=tk,
            ))

        fig_radar.update_layout(
            **carbon_plotly_layout(height=440, title="Fundamental Score Radar"),
            polar=dict(
                bgcolor="#141922",
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor="rgba(107,122,143,0.2)",
                    tickfont=dict(color="#6b7a8f", size=9),
                ),
                angularaxis=dict(
                    gridcolor="rgba(107,122,143,0.2)",
                    tickfont=dict(color="#e2e8f0", size=11),
                ),
            ),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_s:
        composite = sum(scores.values()) / len(scores)
        sc = "#22c55e" if composite >= 65 else "#f59e0b" if composite >= 45 else "#fb7185"
        st.markdown(
            f"""<div style="background:{sc}22;border:1px solid {sc}55;border-radius:8px;
            padding:14px;margin-bottom:16px;text-align:center;">
            <div style="font-size:32px;font-weight:700;color:{sc};">{composite:.0f}</div>
            <div style="color:#6b7a8f;font-size:12px;">Composite Score / 100</div>
            </div>""",
            unsafe_allow_html=True,
        )
        for dim, sv in scores.items():
            bc = "#22c55e" if sv >= 65 else "#f59e0b" if sv >= 45 else "#fb7185"
            st.markdown(
                f"""<div style="margin-bottom:9px;">
                <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px;">
                  <span style="color:#e2e8f0;">{dim}</span>
                  <span style="color:{bc};font-weight:600;">{sv:.0f}</span>
                </div>
                <div style="background:#1e2430;border-radius:4px;height:6px;">
                  <div style="background:{bc};width:{sv:.0f}%;height:6px;border-radius:4px;"></div>
                </div></div>""",
                unsafe_allow_html=True,
            )
        st.caption("Scores saved to session state → Position Scoring page can read them.")

    desc = nfo.get("longBusinessSummary", "")
    if desc:
        with st.expander("📄 Business Description"):
            st.write(desc)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — INCOME STATEMENT
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Income Statement ({period_mode})")

    if fin.empty:
        st.warning("No income statement data available.")
    else:
        st.dataframe(_fmt_stmt_df(fin, n_periods), use_container_width=True)
        st.divider()

        dates_i = _dates(fin)
        def _ts(s): return s.iloc[:n_periods].values[::-1]

        rev_s  = _row(fin, "Total Revenue", "Revenue")
        gp_s   = _row(fin, "Gross Profit")
        oi_s   = _row(fin, "Operating Income", "EBIT", "Operating Income Or Loss")
        ni_s   = _row(fin, "Net Income", "Net Income Common Stockholders")
        eps_s  = _row(fin, "Diluted EPS", "Basic EPS")
        ebit_s = _row(fin, "EBITDA", "Normalized EBITDA")

        # Revenue & income bars
        fig_rev = go.Figure()
        for s, label, color in [
            (rev_s, "Revenue",          "#4a9eff"),
            (gp_s,  "Gross Profit",     "#22d3ee"),
            (oi_s,  "Operating Income", "#22c55e"),
            (ni_s,  "Net Income",       "#00d4aa"),
        ]:
            if not s.empty:
                fig_rev.add_trace(go.Bar(
                    name=label, x=dates_i, y=_ts(s) / 1e9, marker_color=color,
                ))
        fig_rev.update_layout(**carbon_plotly_layout(
            height=380, title="Revenue & Profitability ($B)",
            barmode="group", yaxis_title="$B",
        ))
        st.plotly_chart(fig_rev, use_container_width=True)

        # Margins + EPS side by side
        col_m, col_e = st.columns(2)

        with col_m:
            fig_mg = go.Figure()
            rev_v = _ts(rev_s) if not rev_s.empty else None
            for s, label, color in [
                (gp_s, "Gross Margin",     "#4a9eff"),
                (oi_s, "Operating Margin", "#22d3ee"),
                (ni_s, "Net Margin",       "#00d4aa"),
            ]:
                if not s.empty and rev_v is not None:
                    mg = _ts(s) / np.where(rev_v == 0, np.nan, rev_v) * 100
                    fig_mg.add_trace(go.Scatter(
                        name=label, x=dates_i, y=mg, mode="lines+markers",
                        line=dict(color=color, width=2.5), marker=dict(size=6),
                    ))
            fig_mg.update_layout(**carbon_plotly_layout(
                height=340, title="Margin Trends", yaxis_title="%",
            ))
            fig_mg.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig_mg, use_container_width=True)

        with col_e:
            if not eps_s.empty:
                ev = _ts(eps_s)
                fig_eps = go.Figure(go.Bar(
                    name="EPS", x=dates_i, y=ev,
                    marker_color=["#22c55e" if v >= 0 else "#fb7185" for v in ev],
                ))
                fig_eps.add_hline(y=0, line_color="#ffffff", line_width=1, line_dash="dash")
                fig_eps.update_layout(**carbon_plotly_layout(
                    height=340, title="Diluted EPS", yaxis_title="$",
                ))
                st.plotly_chart(fig_eps, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — BALANCE SHEET
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Balance Sheet ({period_mode})")

    if bs.empty:
        st.warning("No balance sheet data available.")
    else:
        st.dataframe(_fmt_stmt_df(bs, n_periods), use_container_width=True)
        st.divider()

        dates_b = _dates(bs)

        assets_s  = _row(bs, "Total Assets")
        liab_s    = _row(bs, "Total Liabilities Net Minority Interest",
                         "Total Liabilities", "Total Liab")
        eq_s      = _row(bs, "Total Equity Gross Minority Interest",
                         "Stockholders Equity", "Total Stockholder Equity")
        cash_s    = _row(bs, "Cash And Cash Equivalents", "Cash",
                         "Cash And Short Term Investments")
        debt_s    = _row(bs, "Total Debt", "Long Term Debt")
        ca_s      = _row(bs, "Current Assets")
        cl_s      = _row(bs, "Current Liabilities")

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            fig_alq = go.Figure()
            for s, label, color in [
                (assets_s, "Total Assets",          "#4a9eff"),
                (liab_s,   "Total Liabilities",     "#fb7185"),
                (eq_s,     "Shareholders' Equity",  "#22c55e"),
            ]:
                if not s.empty:
                    fig_alq.add_trace(go.Bar(
                        name=label, x=dates_b, y=s.iloc[:n_periods].values[::-1] / 1e9,
                        marker_color=color,
                    ))
            fig_alq.update_layout(**carbon_plotly_layout(
                height=360, title="Assets · Liabilities · Equity ($B)",
                barmode="group", yaxis_title="$B",
            ))
            st.plotly_chart(fig_alq, use_container_width=True)

        with col_b2:
            fig_dc = go.Figure()
            for s, label, color in [
                (debt_s, "Total Debt",        "#fb7185"),
                (cash_s, "Cash & Equivalents","#22c55e"),
            ]:
                if not s.empty:
                    fig_dc.add_trace(go.Bar(
                        name=label, x=dates_b, y=s.iloc[:n_periods].values[::-1] / 1e9,
                        marker_color=color,
                    ))
            fig_dc.update_layout(**carbon_plotly_layout(
                height=360, title="Debt vs Cash ($B)",
                barmode="group", yaxis_title="$B",
            ))
            st.plotly_chart(fig_dc, use_container_width=True)

        # Current ratio trend
        if not ca_s.empty and not cl_s.empty:
            ca_v = ca_s.iloc[:n_periods].values[::-1]
            cl_v = cl_s.iloc[:n_periods].values[::-1]
            cr_v = ca_v / np.where(cl_v == 0, np.nan, cl_v)
            fig_cr = go.Figure(go.Scatter(
                x=dates_b, y=cr_v, mode="lines+markers",
                line=dict(color="#f59e0b", width=2.5), marker=dict(size=7),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.07)",
                name="Current Ratio",
            ))
            fig_cr.add_hline(y=1, line_color="#fb7185", line_dash="dash", line_width=1,
                             annotation_text="1.0", annotation_font_color="#fb7185")
            fig_cr.add_hline(y=2, line_color="#22c55e", line_dash="dot", line_width=1,
                             annotation_text="2.0 (healthy)", annotation_font_color="#22c55e")
            fig_cr.update_layout(**carbon_plotly_layout(
                height=300, title="Current Ratio", yaxis_title="Ratio",
            ))
            st.plotly_chart(fig_cr, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — CASH FLOW
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"Cash Flow Statement ({period_mode})")

    if cf.empty:
        st.warning("No cash flow data available.")
    else:
        st.dataframe(_fmt_stmt_df(cf, n_periods), use_container_width=True)
        st.divider()

        dates_c = _dates(cf)

        ocf_s   = _row(cf, "Operating Cash Flow",
                       "Cash Flow From Continuing Operating Activities")
        capex_s = _row(cf, "Capital Expenditure", "Capital Expenditures",
                       "Purchase Of PPE", "Purchases Of Property Plant And Equipment")
        icf_s   = _row(cf, "Investing Cash Flow",
                       "Cash Flow From Continuing Investing Activities")
        fcf_s   = _row(cf, "Financing Cash Flow",
                       "Cash Flow From Continuing Financing Activities")
        da_s    = _row(cf, "Depreciation And Amortization",
                       "Depreciation Depletion And Amortization")
        buyback_s = _row(cf, "Repurchase Of Capital Stock",
                         "Common Stock Repurchased", "Repurchase Of Common Stock")

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            fig_ocf = go.Figure()
            if not ocf_s.empty:
                fig_ocf.add_trace(go.Bar(
                    name="Operating CF", x=dates_c,
                    y=ocf_s.iloc[:n_periods].values[::-1] / 1e9,
                    marker_color="#22d3ee",
                ))
            if not capex_s.empty:
                fig_ocf.add_trace(go.Bar(
                    name="CapEx", x=dates_c,
                    y=capex_s.iloc[:n_periods].values[::-1] / 1e9,
                    marker_color="#fb7185",
                ))
            fig_ocf.update_layout(**carbon_plotly_layout(
                height=360, title="Operating Cash Flow & CapEx ($B)",
                barmode="group", yaxis_title="$B",
            ))
            st.plotly_chart(fig_ocf, use_container_width=True)

        with col_c2:
            if not ocf_s.empty and not capex_s.empty:
                ocf_v   = ocf_s.iloc[:n_periods].values[::-1]
                capex_v = capex_s.iloc[:n_periods].values[::-1]
                fcf_v   = ocf_v + capex_v   # capex is negative in yfinance
                fig_fcf = go.Figure(go.Bar(
                    name="Free Cash Flow", x=dates_c, y=fcf_v / 1e9,
                    marker_color=["#22c55e" if v >= 0 else "#fb7185" for v in fcf_v],
                ))
                fig_fcf.add_hline(y=0, line_color="#ffffff", line_width=1, line_dash="dash")
                fig_fcf.update_layout(**carbon_plotly_layout(
                    height=360, title="Free Cash Flow ($B)", yaxis_title="$B",
                ))
                st.plotly_chart(fig_fcf, use_container_width=True)

        # FCF margin trend
        rev_stmt = stmts.get("financials", pd.DataFrame())
        if not ocf_s.empty and not capex_s.empty and not rev_stmt.empty:
            rev_ann = _row(rev_stmt, "Total Revenue", "Revenue")
            if not rev_ann.empty:
                n_common = min(n_periods, len(ocf_s), len(capex_s), len(rev_ann))
                ocf_c   = ocf_s.iloc[:n_common].values[::-1]
                capex_c = capex_s.iloc[:n_common].values[::-1]
                rev_c   = rev_ann.iloc[:n_common].values[::-1]
                dates_fc = _dates(cf)[:n_common]
                fcf_mg  = (ocf_c + capex_c) / np.where(rev_c == 0, np.nan, rev_c) * 100
                fig_fm  = go.Figure(go.Scatter(
                    x=dates_fc, y=fcf_mg, mode="lines+markers",
                    line=dict(color="#00d4aa", width=2.5), marker=dict(size=7),
                    fill="tozeroy", fillcolor="rgba(0,212,170,0.07)",
                    name="FCF Margin",
                ))
                fig_fm.add_hline(y=0, line_color="#fb7185", line_dash="dash", line_width=1)
                fig_fm.update_layout(**carbon_plotly_layout(
                    height=300, title="Free Cash Flow Margin", yaxis_title="%",
                ))
                fig_fm.update_yaxes(ticksuffix="%")
                st.plotly_chart(fig_fm, use_container_width=True)

        if not da_s.empty:
            with st.expander("D&A and Buybacks"):
                col_da, col_bb = st.columns(2)
                with col_da:
                    fig_da = go.Figure(go.Bar(
                        name="D&A", x=dates_c,
                        y=da_s.iloc[:n_periods].values[::-1] / 1e9,
                        marker_color="#a78bfa",
                    ))
                    fig_da.update_layout(**carbon_plotly_layout(
                        height=280, title="Depreciation & Amortization ($B)", yaxis_title="$B",
                    ))
                    st.plotly_chart(fig_da, use_container_width=True)
                with col_bb:
                    if not buyback_s.empty:
                        bb_v = buyback_s.iloc[:n_periods].values[::-1]
                        fig_bb = go.Figure(go.Bar(
                            name="Buybacks", x=dates_c, y=bb_v / 1e9,
                            marker_color="#f59e0b",
                        ))
                        fig_bb.update_layout(**carbon_plotly_layout(
                            height=280, title="Share Buybacks ($B)", yaxis_title="$B",
                        ))
                        st.plotly_chart(fig_bb, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — VALUATION & PEERS
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    all_tks = [primary] + comp_tickers

    if len(all_tks) < 2:
        st.info("Add comparison tickers in the sidebar to enable peer charts.")
    else:
        st.subheader("Valuation vs Peers")
        peer_rows = []
        with st.spinner("Loading peer data…"):
            for tk in all_tks:
                pi = _info(tk)
                if not pi:
                    continue
                peer_rows.append({
                    "Ticker":       tk,
                    "Name":         pi.get("shortName", tk),
                    "Sector":       pi.get("sector", "—"),
                    "Market Cap":   pi.get("marketCap", np.nan),
                    "Price":        _safe(pi.get("currentPrice") or pi.get("regularMarketPrice")),
                    "Trailing P/E": _safe(pi.get("trailingPE")),
                    "Forward P/E":  _safe(pi.get("forwardPE")),
                    "EV/EBITDA":    _safe(pi.get("enterpriseToEbitda")),
                    "P/S":          _safe(pi.get("priceToSalesTrailingTwelveMonths")),
                    "P/B":          _safe(pi.get("priceToBook")),
                    "Net Margin":   _safe(pi.get("profitMargins")),
                    "Gross Margin": _safe(pi.get("grossMargins")),
                    "ROE":          _safe(pi.get("returnOnEquity")),
                    "ROA":          _safe(pi.get("returnOnAssets")),
                    "D/E":          _safe(pi.get("debtToEquity")),
                    "Rev Growth":   _safe(pi.get("revenueGrowth")),
                    "EPS Growth":   _safe(pi.get("earningsGrowth")),
                    "Beta":         _safe(pi.get("beta")),
                })

        peer_df = pd.DataFrame(peer_rows)

        if not peer_df.empty:
            def _bar_peers(metric, suffix, title, height=300):
                colors = [
                    "#00d4aa" if tk == primary else "#4a9eff"
                    for tk in peer_df["Ticker"]
                ]
                fig = go.Figure(go.Bar(
                    x=peer_df["Ticker"], y=peer_df[metric],
                    marker_color=colors,
                    text=[f"{v:.1f}{suffix}" if not np.isnan(v) else "—"
                          for v in peer_df[metric]],
                    textposition="outside",
                ))
                fig.update_layout(**carbon_plotly_layout(
                    height=height, title=title, showlegend=False,
                ))
                return fig

            # Valuation multiples — 2×2 grid
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.plotly_chart(_bar_peers("Trailing P/E", "×", "Trailing P/E vs Peers"),
                                use_container_width=True)
                st.plotly_chart(_bar_peers("P/S", "×", "P/S vs Peers"),
                                use_container_width=True)
            with col_v2:
                st.plotly_chart(_bar_peers("EV/EBITDA", "×", "EV/EBITDA vs Peers"),
                                use_container_width=True)
                st.plotly_chart(_bar_peers("P/B", "×", "P/B vs Peers"),
                                use_container_width=True)

            st.divider()

            # Profitability & growth grouped bar
            st.subheader("Profitability & Growth vs Peers")
            fig_prof = go.Figure()
            metrics_pct = ["Net Margin", "Gross Margin", "ROE", "ROA", "Rev Growth"]
            for _, pr in peer_df.iterrows():
                c = "#00d4aa" if pr["Ticker"] == primary else "#4a9eff"
                fig_prof.add_trace(go.Bar(
                    name=pr["Ticker"],
                    x=metrics_pct,
                    y=[pr[m] * 100 if not np.isnan(pr[m]) else 0 for m in metrics_pct],
                    marker_color=c,
                ))
            fig_prof.update_layout(**carbon_plotly_layout(
                height=380, title="Profitability & Growth Comparison (%)",
                barmode="group", yaxis_title="%",
            ))
            fig_prof.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig_prof, use_container_width=True)

            # Bubble: P/E vs Revenue Growth, sized by market cap
            st.subheader("P/E vs Revenue Growth")
            fig_bub = go.Figure()
            for _, pr in peer_df.iterrows():
                if np.isnan(pr["Trailing P/E"]) or np.isnan(pr["Rev Growth"]):
                    continue
                mcap_b = _safe(pr["Market Cap"], 1e9)
                fig_bub.add_trace(go.Scatter(
                    x=[pr["Rev Growth"] * 100],
                    y=[pr["Trailing P/E"]],
                    mode="markers+text",
                    name=pr["Ticker"],
                    text=[pr["Ticker"]],
                    textposition="top center",
                    marker=dict(
                        size=max(12, min(55, mcap_b / 1e9 / 2)),
                        color="#00d4aa" if pr["Ticker"] == primary else "#4a9eff",
                        opacity=0.8,
                        line=dict(width=1.5, color="#ffffff"),
                    ),
                ))
            fig_bub.update_layout(**carbon_plotly_layout(
                height=420,
                title="P/E vs Revenue Growth (bubble = market cap)",
                xaxis_title="Revenue Growth (%)",
                yaxis_title="Trailing P/E",
            ))
            fig_bub.update_xaxes(ticksuffix="%")
            st.plotly_chart(fig_bub, use_container_width=True)

            # Full peer table
            st.subheader("Full Peer Metrics Table")
            disp = peer_df.copy()
            for c in ["Net Margin", "Gross Margin", "ROE", "ROA", "Rev Growth", "EPS Growth"]:
                disp[c] = disp[c].apply(lambda v: f"{v*100:.1f}%" if not np.isnan(v) else "—")
            for c in ["Trailing P/E", "Forward P/E", "EV/EBITDA", "P/S", "P/B", "D/E", "Beta"]:
                disp[c] = disp[c].apply(lambda v: f"{v:.1f}" if not np.isnan(v) else "—")
            disp["Market Cap"] = disp["Market Cap"].apply(_fv)
            disp["Price"] = disp["Price"].apply(
                lambda v: f"${v:.2f}" if not np.isnan(v) else "—"
            )
            st.dataframe(disp, use_container_width=True, hide_index=True)

    # Historical P/E
    st.divider()
    st.subheader("Historical P/E")
    ttm_eps = _safe(nfo.get("trailingEps"), 0)
    if not px.empty and ttm_eps > 0:
        px_close = px["Close"] if "Close" in px.columns else px.iloc[:, 0]
        hist_pe  = px_close / ttm_eps
        pe_med   = float(hist_pe.median())
        curr_pe  = _safe(nfo.get("trailingPE"), np.nan)

        fig_hpe = go.Figure()
        fig_hpe.add_trace(go.Scatter(
            x=hist_pe.index, y=hist_pe.values, mode="lines",
            line=dict(color="#4a9eff", width=2),
            fill="tozeroy", fillcolor="rgba(74,158,255,0.07)",
            name="Trailing P/E (approx)",
        ))
        fig_hpe.add_hline(y=pe_med, line_color="#f59e0b", line_dash="dot",
                          annotation_text=f"Median {pe_med:.1f}×",
                          annotation_font_color="#f59e0b")
        if not np.isnan(curr_pe):
            fig_hpe.add_hline(y=curr_pe, line_color="#00d4aa", line_dash="dash",
                              annotation_text=f"Current {curr_pe:.1f}×",
                              annotation_font_color="#00d4aa")
        fig_hpe.update_layout(**carbon_plotly_layout(
            height=340,
            title=f"{primary} — Trailing P/E (price ÷ trailing EPS {ttm_eps:.2f})",
            xaxis_title="Date", yaxis_title="P/E Ratio",
        ))
        st.plotly_chart(fig_hpe, use_container_width=True)
    else:
        st.caption("Historical P/E unavailable — negative or zero trailing EPS.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — SCREENER
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🔎 Fundamental Screener")
    st.markdown(
        "Screen any list of tickers against fundamental thresholds. "
        "Portfolio tickers are pre-loaded; add more below."
    )

    scr_defaults = list(dict.fromkeys(
        _ptickers + [t for t in comp_tickers if t not in _ptickers]
    ))
    scr_input = st.text_input(
        "Tickers to screen (comma-separated)",
        value=", ".join(scr_defaults),
        key="scr_tickers",
    )
    scr_tickers = [t.strip().upper() for t in scr_input.split(",") if t.strip()]

    if not scr_tickers:
        st.info("Enter at least one ticker to screen.")
    else:
        st.markdown("**Filters** — slide to extremes to disable:")
        sf1, sf2, sf3 = st.columns(3)
        with sf1:
            f_rev_g  = st.slider("Min Revenue Growth (%)", -50, 100, -50, 1)
            f_pe     = st.slider("Max Trailing P/E", 0, 150, 150, 1)
            f_nm     = st.slider("Min Net Margin (%)", -50, 60, -50, 1)
        with sf2:
            f_roe    = st.slider("Min ROE (%)", -50, 100, -50, 1)
            f_de     = st.slider("Max Debt/Equity", 0, 20, 20, 1)
            f_gm     = st.slider("Min Gross Margin (%)", 0, 100, 0, 1)
        with sf3:
            f_cr     = st.slider("Min Current Ratio", 0.0, 5.0, 0.0, 0.1)
            f_fcf    = st.slider("Min FCF ($M)", -5000, 50000, -5000, 100)
            f_score  = st.slider("Min Composite Score", 0, 100, 0, 5)

        if st.button("▶ Run Screener", type="primary"):
            results = []
            prog    = st.progress(0)
            status  = st.empty()

            for i, tk in enumerate(scr_tickers):
                status.text(f"Fetching {tk}… ({i+1}/{len(scr_tickers)})")
                prog.progress((i + 1) / len(scr_tickers))

                pi  = _info(tk)
                ps  = _statements(tk)
                sc  = _compute_scores(pi, ps)
                csc = sum(sc.values()) / len(sc)

                if not pi:
                    continue

                rev_g  = _safe(pi.get("revenueGrowth"), np.nan) * 100
                pe_v   = _safe(pi.get("trailingPE"), 9999)
                nm_v   = _safe(pi.get("profitMargins"), np.nan) * 100
                roe_v  = _safe(pi.get("returnOnEquity"), np.nan) * 100
                de_v   = _safe(pi.get("debtToEquity"), 0)
                gm_v   = _safe(pi.get("grossMargins"), np.nan) * 100
                cr_v   = _safe(pi.get("currentRatio"), 0)
                fcf_v  = _safe(pi.get("freeCashflow"), np.nan) / 1e6

                def _ok(v, threshold, op="ge"):
                    if np.isnan(v): return True   # missing data passes filter
                    return v >= threshold if op == "ge" else v <= threshold

                if not _ok(rev_g, f_rev_g):  continue
                if not _ok(pe_v, f_pe, "le"): continue
                if not _ok(nm_v, f_nm):       continue
                if not _ok(roe_v, f_roe):     continue
                if not _ok(de_v, f_de, "le"): continue
                if not _ok(gm_v, f_gm):       continue
                if not _ok(cr_v, f_cr):       continue
                if not _ok(fcf_v, f_fcf):     continue
                if csc < f_score:              continue

                results.append({
                    "Ticker":       tk,
                    "Name":         pi.get("shortName", tk),
                    "Sector":       pi.get("sector", "—"),
                    "Market Cap":   pi.get("marketCap", np.nan),
                    "Price":        _safe(pi.get("currentPrice") or pi.get("regularMarketPrice")),
                    "P/E":          pe_v if pe_v < 9999 else np.nan,
                    "EV/EBITDA":    _safe(pi.get("enterpriseToEbitda")),
                    "P/S":          _safe(pi.get("priceToSalesTrailingTwelveMonths")),
                    "Net Margin %": nm_v,
                    "Gross Margin %": gm_v,
                    "ROE %":        roe_v,
                    "Rev Growth %": rev_g,
                    "D/E":          de_v,
                    "Current Ratio":cr_v,
                    "FCF ($M)":     fcf_v,
                    "Score":        csc,
                })

            prog.empty()
            status.empty()

            if not results:
                st.warning("No tickers passed the current filters.")
            else:
                res_df = pd.DataFrame(results).sort_values("Score", ascending=False)

                st.success(f"**{len(res_df)}** of {len(scr_tickers)} tickers passed.")

                res_disp = res_df.copy()
                res_disp["Market Cap"] = res_disp["Market Cap"].apply(_fv)
                st.dataframe(
                    res_disp.style
                    .format({
                        "Price":          "${:.2f}",
                        "P/E":            "{:.1f}",
                        "EV/EBITDA":      "{:.1f}",
                        "P/S":            "{:.1f}",
                        "Net Margin %":   "{:.1f}%",
                        "Gross Margin %": "{:.1f}%",
                        "ROE %":          "{:.1f}%",
                        "Rev Growth %":   "{:.1f}%",
                        "D/E":            "{:.1f}",
                        "Current Ratio":  "{:.2f}",
                        "FCF ($M)":       "${:,.0f}",
                        "Score":          "{:.0f}",
                    }, na_rep="—")
                    .background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100),
                    use_container_width=True, hide_index=True,
                )

                st.download_button(
                    "⬇ Download Results CSV",
                    res_df.to_csv(index=False),
                    f"screener_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                )

                # Score comparison chart
                fig_sc = go.Figure(go.Bar(
                    x=res_df["Ticker"], y=res_df["Score"],
                    marker_color=[
                        "#22c55e" if v >= 65 else "#f59e0b" if v >= 45 else "#fb7185"
                        for v in res_df["Score"]
                    ],
                    text=[f"{v:.0f}" for v in res_df["Score"]],
                    textposition="outside",
                ))
                fig_sc.update_layout(**carbon_plotly_layout(
                    height=320, title="Composite Score — Passing Tickers",
                    yaxis_title="Score", showlegend=False,
                ))
                fig_sc.update_yaxes(range=[0, 105])
                st.plotly_chart(fig_sc, use_container_width=True)
