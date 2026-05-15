"""
Regime-Aware Portfolio Manager
Section 11: Enhanced Watchlist & Alerts
Track tickers with price alerts, target prices, thesis notes, regime scoring.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
import data.alpaca_client as alpaca
from calculations.regime_detector import RegimeDetector
from calculations.scoring_engine import ScoringEngine
from data.market_data import MarketDataLoader
from utils.carbon_theme import (
    ACCENT, AMBER, BG, BORDER, CARD, CARD2, DIM, FG, GAIN, LOSS, SUBTLE,
    apply_carbon_theme, carbon_plotly_layout, flex_table, page_header,
    section_header, top_nav, regime_color, macro_event_banner,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Watchlist & Alerts", page_icon="◈", layout="wide")
apply_carbon_theme()
top_nav("Watchlist")
page_header("Watchlist & Alerts", "Monitor tickers with custom alerts, target prices, and regime scoring")

# ── Persistence ───────────────────────────────────────────────────────────────
_WL_PATH = Path.home() / ".portfolio_manager" / "watchlist.json"

def _load_watchlist() -> list[dict]:
    try:
        if _WL_PATH.exists():
            with open(_WL_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return []

def _save_watchlist(wl: list[dict]) -> None:
    _WL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_WL_PATH, "w") as f:
        json.dump(wl, f, indent=2, default=str)

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = _load_watchlist()

watchlist: list[dict] = st.session_state["watchlist"]

# ── Market context (regime) ───────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _get_regime() -> str:
    try:
        ml = MarketDataLoader()
        det = RegimeDetector()
        spy = ml.load_index_data("SPY", "2y")
        vix = ml.load_vix_data("2y")
        spx, vxp = ml.align_data(spy, vix)
        reg, _ = det.classify_regime(spx, vxp)
        return str(reg.iloc[-1])
    except Exception:
        return "Unknown"

current_regime = _get_regime()

# Macro event banners
try:
    from data.calendar_data import get_upcoming_events as _get_evts
    _raw = _get_evts(days_ahead=7)
    _evts = [{"name": e["label"], "days_away": e["days"]} for e in _raw]
    if _evts:
        macro_event_banner(_evts)
except Exception:
    pass

# ── Add ticker form ───────────────────────────────────────────────────────────
section_header("Add to Watchlist")
with st.form("add_wl_form", clear_on_submit=True):
    _r1c1, _r1c2, _r1c3 = st.columns([1, 2, 2])
    new_ticker  = _r1c1.text_input("Ticker", placeholder="AAPL").upper().strip()
    thesis      = _r1c2.text_input("Investment Thesis", placeholder="Bullish on AI capex cycle")
    notes       = _r1c3.text_input("Notes", placeholder="Optional")

    _r2c1, _r2c2, _r2c3, _r2c4, _r2c5 = st.columns(5)
    target_price      = _r2c1.number_input("Target Price ($)", min_value=0.0, value=0.0, step=1.0,
                                            help="Your price target — shown as % upside/downside")
    stop_loss         = _r2c2.number_input("Stop Loss ($)",    min_value=0.0, value=0.0, step=1.0,
                                            help="0 = disabled")
    alert_price_above = _r2c3.number_input("Alert Above ($)",  min_value=0.0, value=0.0, step=1.0)
    alert_price_below = _r2c4.number_input("Alert Below ($)",  min_value=0.0, value=0.0, step=1.0)
    alert_pct_move    = _r2c5.number_input("Alert Day Move >(%)", min_value=0.0, value=0.0, step=0.5,
                                            help="0 = disabled")
    submitted = st.form_submit_button("Add to Watchlist", type="primary")

if submitted and new_ticker:
    existing = [w["ticker"] for w in watchlist]
    if new_ticker in existing:
        st.warning(f"{new_ticker} is already on your watchlist.")
    else:
        watchlist.append({
            "ticker":            new_ticker,
            "thesis":            thesis,
            "notes":             notes,
            "added_at":          datetime.now().isoformat(),
            "target_price":      target_price  if target_price  > 0 else None,
            "stop_loss":         stop_loss      if stop_loss     > 0 else None,
            "alert_price_above": alert_price_above if alert_price_above > 0 else None,
            "alert_price_below": alert_price_below if alert_price_below > 0 else None,
            "alert_pct_move":    alert_pct_move    if alert_pct_move    > 0 else None,
        })
        _save_watchlist(watchlist)
        st.success(f"Added {new_ticker} to watchlist.")
        st.rerun()

# ── Load live data for watchlist ──────────────────────────────────────────────
if not watchlist:
    st.info("Your watchlist is empty. Add tickers above.")
    st.stop()

wl_tickers = [w["ticker"] for w in watchlist]

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_watchlist_data(tickers: tuple) -> pd.DataFrame:
    rows = []
    try:
        close = alpaca.get_close_prices(list(tickers), period="10d")
        for t in tickers:
            try:
                s = close[t.upper()].dropna() if t.upper() in close.columns else pd.Series(dtype=float)
                cur  = float(s.iloc[-1]) if len(s) >= 1 else None
                prev = float(s.iloc[-2]) if len(s) >= 2 else cur
                chg  = ((cur - prev) / prev * 100) if (cur and prev and prev != 0) else None
                vol_20 = None
                if len(s) >= 21:
                    log_rets = np.log(s / s.shift(1)).dropna()
                    vol_20 = float(log_rets.iloc[-20:].std() * np.sqrt(252) * 100)
                rows.append({"ticker": t, "price": cur, "prev_close": prev,
                             "day_chg_pct": chg, "vol_20d_ann": vol_20})
            except Exception:
                rows.append({"ticker": t, "price": None, "prev_close": None,
                             "day_chg_pct": None, "vol_20d_ann": None})
    except Exception:
        for t in tickers:
            rows.append({"ticker": t, "price": None, "prev_close": None,
                         "day_chg_pct": None, "vol_20d_ann": None})
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800, show_spinner=False)
def _regime_score_ticker(ticker: str, regime: str) -> float | None:
    try:
        scorer = ScoringEngine()
        ml2    = MarketDataLoader()
        det2   = RegimeDetector()
        hist   = ml2.load_single_stock(ticker, period="1y")
        if hist.empty:
            return None
        spy2   = ml2.load_index_data("SPY", "1y")
        vix2   = ml2.load_vix_data("1y")
        spx2, vxp2 = ml2.align_data(spy2, vix2)
        reg2, sigs2 = det2.classify_regime(spx2, vxp2)
        scores = scorer.score_portfolio({ticker: hist}, reg2, sigs2)
        if ticker in scores:
            return round(float(scores[ticker].get("composite", 50)), 1)
        return None
    except Exception:
        return None

with st.spinner("Refreshing watchlist data…"):
    live = _fetch_watchlist_data(tuple(wl_tickers))

wl_df = pd.DataFrame(watchlist)
if not live.empty:
    wl_df = wl_df.merge(live, on="ticker", how="left")
else:
    for col in ["price", "prev_close", "day_chg_pct", "vol_20d_ann"]:
        wl_df[col] = None

# ── Triggered alerts ──────────────────────────────────────────────────────────
triggered: list[str] = []
stop_triggered: list[str] = []
for _, row in wl_df.iterrows():
    t   = row["ticker"]
    p   = row.get("price")
    chg = row.get("day_chg_pct")
    if p is not None:
        if row.get("alert_price_above") and p > row["alert_price_above"]:
            triggered.append(f"{t} crossed above ${row['alert_price_above']:.2f} (now ${p:.2f})")
        if row.get("alert_price_below") and p < row["alert_price_below"]:
            triggered.append(f"{t} crossed below ${row['alert_price_below']:.2f} (now ${p:.2f})")
        if row.get("stop_loss") and p < row["stop_loss"]:
            stop_triggered.append(f"{t} hit stop loss ${row['stop_loss']:.2f} (now ${p:.2f})")
        if row.get("target_price") and p >= row["target_price"]:
            triggered.append(f"{t} reached target price ${row['target_price']:.2f} (now ${p:.2f})")
    if chg is not None and row.get("alert_pct_move") and abs(chg) >= row["alert_pct_move"]:
        triggered.append(f"{t} moved {chg:+.2f}% today (threshold ±{row['alert_pct_move']:.1f}%)")

for msg in stop_triggered:
    st.markdown(
        f'<div style="background:rgba(251,113,133,0.12);border:2px solid {LOSS};'
        f'padding:10px 16px;margin-bottom:8px;border-radius:8px;font-size:0.88rem;color:{LOSS};font-weight:600;">'
        f'STOP LOSS: {msg}</div>',
        unsafe_allow_html=True,
    )
for msg in triggered:
    st.markdown(
        f'<div style="background:rgba(245,158,11,0.10);border-left:3px solid {AMBER};'
        f'padding:8px 14px;margin-bottom:6px;border-radius:4px;font-size:0.82rem;color:{AMBER};">'
        f'Alert: {msg}</div>',
        unsafe_allow_html=True,
    )

# ── Main watchlist table ──────────────────────────────────────────────────────
section_header("Watchlist")

# Build display rows
display_rows = []
for _, row in wl_df.iterrows():
    t      = row["ticker"]
    p      = row.get("price")
    chg    = row.get("day_chg_pct")
    vol    = row.get("vol_20d_ann")
    tgt    = row.get("target_price")
    sl     = row.get("stop_loss")
    added  = str(row.get("added_at", ""))[:10]
    thesis = row.get("thesis", "") or ""

    upside = None
    if p and tgt:
        upside = (tgt / p - 1) * 100

    display_rows.append({
        "Ticker":     t,
        "Price":      f"${p:.2f}"   if p   is not None else "—",
        "Day %":      f"{chg:+.2f}%" if chg is not None else "—",
        "20d Vol":    f"{vol:.1f}%" if vol is not None else "—",
        "Target":     f"${tgt:.2f}" if tgt is not None else "—",
        "Upside":     f"{upside:+.1f}%" if upside is not None else "—",
        "Stop Loss":  f"${sl:.2f}"  if sl  is not None else "—",
        "Regime":     current_regime,
        "Added":      added,
        "Thesis":     thesis,
        "_chg":       chg,
        "_upside":    upside,
    })

if display_rows:
    flex_table(
        pd.DataFrame(display_rows),
        columns=[
            {"key": "Ticker",    "label": "Ticker",    "width": "7%",  "align": "left"},
            {"key": "Price",     "label": "Price",     "width": "8%",  "align": "right",  "numeric": True},
            {"key": "Day %",     "label": "Day %",     "width": "8%",  "align": "right",  "numeric": True,
             "color_scale": "rg"},
            {"key": "20d Vol",   "label": "20d Vol",   "width": "7%",  "align": "right",  "numeric": True},
            {"key": "Target",    "label": "Target",    "width": "8%",  "align": "right",  "numeric": True},
            {"key": "Upside",    "label": "Upside",    "width": "8%",  "align": "right",  "numeric": True,
             "color_scale": "rg"},
            {"key": "Stop Loss", "label": "Stop Loss", "width": "9%",  "align": "right",  "numeric": True},
            {"key": "Regime",    "label": "Regime",    "width": "10%", "align": "left"},
            {"key": "Added",     "label": "Added",     "width": "9%",  "align": "left"},
            {"key": "Thesis",    "label": "Thesis",    "width": "26%", "align": "left"},
        ],
        key="wl_main_tbl",
    )

# ── Regime scoring ────────────────────────────────────────────────────────────
section_header("Regime Scores")
st.caption(f"Composite regime scores for watchlist tickers under current regime: **{current_regime}**")

_score_rows = []
for t in wl_tickers:
    with st.spinner(f"Scoring {t}…"):
        score = _regime_score_ticker(t, current_regime)
    _score_rows.append({"Ticker": t, "Score": score if score is not None else "—",
                         "_score_num": score or 0})

_score_df = pd.DataFrame(_score_rows).sort_values("_score_num", ascending=False)
flex_table(
    _score_df[["Ticker", "Score"]],
    columns=[
        {"key": "Ticker", "label": "Ticker", "width": "30%", "align": "left"},
        {"key": "Score",  "label": f"Score ({current_regime})",
         "width": "70%", "align": "right", "numeric": True, "color_scale": "rg",
         "fmt": lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else str(v)},
    ],
    key="wl_scores_tbl",
)

# ── Remove tickers ────────────────────────────────────────────────────────────
section_header("Manage Watchlist")
with st.expander("Remove a Ticker"):
    to_remove = st.selectbox("Select ticker to remove", ["—"] + wl_tickers, key="_wl_rm")
    if st.button("Remove", type="secondary", key="_wl_rm_btn") and to_remove != "—":
        watchlist = [w for w in watchlist if w["ticker"] != to_remove]
        st.session_state["watchlist"] = watchlist
        _save_watchlist(watchlist)
        st.success(f"Removed {to_remove}.")
        st.rerun()

with st.expander("Edit Alerts / Targets"):
    _edit_t = st.selectbox("Ticker", wl_tickers, key="_wl_edit_sel")
    _edit_entry = next((w for w in watchlist if w["ticker"] == _edit_t), None)
    if _edit_entry:
        with st.form("edit_wl_form"):
            _ec1, _ec2, _ec3 = st.columns(3)
            _new_thesis = _ec1.text_input("Thesis", value=_edit_entry.get("thesis", "") or "")
            _new_tgt    = _ec2.number_input("Target Price ($)", min_value=0.0,
                                             value=float(_edit_entry.get("target_price") or 0))
            _new_sl     = _ec3.number_input("Stop Loss ($)",    min_value=0.0,
                                             value=float(_edit_entry.get("stop_loss") or 0))
            _ea1, _ea2, _ea3 = st.columns(3)
            _new_abv    = _ea1.number_input("Alert Above ($)",  min_value=0.0,
                                             value=float(_edit_entry.get("alert_price_above") or 0))
            _new_blw    = _ea2.number_input("Alert Below ($)",  min_value=0.0,
                                             value=float(_edit_entry.get("alert_price_below") or 0))
            _new_pct    = _ea3.number_input("Alert Day Move (%)", min_value=0.0,
                                             value=float(_edit_entry.get("alert_pct_move") or 0))
            _save_edit = st.form_submit_button("Save Changes", type="primary")

        if _save_edit:
            for w in watchlist:
                if w["ticker"] == _edit_t:
                    w["thesis"]            = _new_thesis
                    w["target_price"]      = _new_tgt   if _new_tgt  > 0 else None
                    w["stop_loss"]         = _new_sl    if _new_sl   > 0 else None
                    w["alert_price_above"] = _new_abv   if _new_abv  > 0 else None
                    w["alert_price_below"] = _new_blw   if _new_blw  > 0 else None
                    w["alert_pct_move"]    = _new_pct   if _new_pct  > 0 else None
            st.session_state["watchlist"] = watchlist
            _save_watchlist(watchlist)
            st.success(f"Updated {_edit_t}.")
            st.rerun()

# ── Mini price charts ─────────────────────────────────────────────────────────
section_header("7-Day Price Sparklines")

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_sparklines(tickers: tuple) -> dict:
    result = {}
    try:
        close = alpaca.get_close_prices(list(tickers), period="10d")
        for t in tickers:
            col = t.upper()
            if col in close.columns:
                result[t] = close[col].dropna().tail(7)
    except Exception:
        pass
    return result

spark_data = _fetch_sparklines(tuple(wl_tickers))

cols_per_row = 4
for i in range(0, len(wl_tickers), cols_per_row):
    row_tickers = wl_tickers[i:i + cols_per_row]
    spark_cols  = st.columns(cols_per_row)
    for col, t in zip(spark_cols, row_tickers):
        with col:
            s = spark_data.get(t)
            entry = next((w for w in watchlist if w["ticker"] == t), {})
            tgt_p = entry.get("target_price")
            sl_p  = entry.get("stop_loss")

            if s is not None and len(s) > 1:
                color = GAIN if s.iloc[-1] >= s.iloc[0] else LOSS
                rgb   = ",".join(str(int(color.lstrip("#")[i:i+2], 16)) for i in (0, 2, 4))
                fig_s = go.Figure(go.Scatter(
                    x=s.index, y=s.values,
                    line=dict(color=color, width=1.5),
                    fill="tozeroy",
                    fillcolor=f"rgba({rgb},0.12)",
                ))
                if tgt_p:
                    fig_s.add_hline(y=tgt_p, line_color=GAIN, line_dash="dot", line_width=1,
                                    annotation_text="Target", annotation_font_color=GAIN)
                if sl_p:
                    fig_s.add_hline(y=sl_p,  line_color=LOSS, line_dash="dot", line_width=1,
                                    annotation_text="Stop",   annotation_font_color=LOSS)
                fig_s.update_layout(**carbon_plotly_layout(height=140, margin=dict(l=8, r=8, t=28, b=8)))
                fig_s.update_xaxes(showticklabels=False)
                fig_s.update_layout(title=dict(text=t, font=dict(size=11, color=ACCENT), x=0.5))
                st.plotly_chart(fig_s, use_container_width=True)
            else:
                col.markdown(
                    f'<div style="text-align:center;color:{DIM};padding:2rem;">{t}<br>No data</div>',
                    unsafe_allow_html=True,
                )
