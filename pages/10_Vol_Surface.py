"""
Regime-Aware Portfolio Manager
Section 10: Volatility Surface — IV Surface, Term Structure & Skew
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from data.options_data import OptionsDataLoader
from utils.theme import apply_dark_theme, dark_plotly_layout

st.set_page_config(page_title="Vol Surface", page_icon="🌋", layout="wide")
apply_dark_theme()

st.title("🌋 Volatility Surface")
st.markdown("**Section 10:** Visualize implied volatility across all strikes and expirations")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Surface Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
max_exp = st.sidebar.slider("Max Expirations to Scan", 4, 12, 8,
                             help="More = richer surface but slower to load")
min_dte = st.sidebar.number_input("Min DTE", min_value=1, max_value=30, value=7)
max_dte = st.sidebar.number_input("Max DTE", min_value=30, max_value=730, value=180)

if st.sidebar.button("🔄 Build Surface", type="primary", use_container_width=True):
    st.session_state['vs_ticker']  = ticker
    st.session_state['vs_params']  = (int(max_exp), int(min_dte), int(max_dte))
    st.session_state['vs_loaded']  = True

if not st.session_state.get('vs_loaded', False):
    st.info("👈 Configure the settings in the sidebar and click **Build Surface** to begin.")
    st.markdown("""
    ### What You'll See

    **Tab 1 — IV Surface**
    Interactive 3D surface (or 2D heatmap) showing implied volatility across
    every strike percentage and expiration in the chain.

    **Tab 2 — Term Structure**
    ATM implied volatility plotted against days-to-expiry. Labels whether the
    curve is in *Contango* (far IV > near IV) or *Backwardation* (near IV > far IV).

    **Tab 3 — Skew Analysis**
    Put-call skew per expiration: the difference between 5%-OTM put IV and 5%-OTM
    call IV. Positive values signal bearish hedging demand (put premium).
    """)
    st.stop()

ticker = st.session_state['vs_ticker']
max_exp_p, min_dte_p, max_dte_p = st.session_state['vs_params']

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
# Standard strike percentages (put side ≤ 1.0, call side ≥ 1.0)
_STRIKE_PCTS = np.array([
    0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975,
    1.00,
    1.025, 1.05, 1.075, 1.10, 1.125, 1.15, 1.20
])
_STRIKE_LABELS = [f"{p*100:.1f}%" for p in _STRIKE_PCTS]


@st.cache_data(ttl=300, show_spinner="Building volatility surface…")
def _build_iv_matrix(ticker: str, max_exp: int, min_dte: int, max_dte: int):
    """Fetch IV data for each expiration and interpolate to standard strikes."""
    from data.options_data import OptionsDataLoader
    loader = OptionsDataLoader()

    expirations = loader.get_options_expirations(ticker)
    if not expirations:
        return None, None, None, None, []

    now = datetime.now()
    dtes_list, iv_rows, exp_labels = [], [], []
    spot = None
    scanned = 0

    for exp in expirations:
        if scanned >= max_exp:
            break
        try:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
        except ValueError:
            continue
        dte = (exp_date - now).days
        if not (min_dte <= dte <= max_dte):
            scanned += 1
            continue

        calls, puts, underlying = loader.get_options_chain(ticker, exp)
        if calls.empty or puts.empty:
            scanned += 1
            continue

        if spot is None:
            spot = float(underlying)

        iv_row = []
        for pct in _STRIKE_PCTS:
            target = underlying * pct
            df = puts if pct <= 1.0 else calls
            if df.empty:
                iv_row.append(np.nan)
                continue
            diff = (df['strike'] - target).abs()
            nearest = df.loc[diff.idxmin()]
            iv_val = float(nearest.get('impliedVolatility', np.nan) or np.nan)
            iv_row.append(iv_val if iv_val and iv_val > 0.005 else np.nan)

        valid = [v for v in iv_row if not np.isnan(v)]
        if not valid:
            scanned += 1
            continue
        chain_avg = float(np.mean(valid))
        iv_row = [v if not np.isnan(v) else chain_avg for v in iv_row]

        dtes_list.append(dte)
        iv_rows.append(iv_row)
        exp_labels.append(f"{exp} ({dte}d)")
        scanned += 1

    if len(dtes_list) < 2:
        return None, None, None, None, []

    sort_idx = np.argsort(dtes_list)
    dtes_arr    = np.array(dtes_list)[sort_idx]
    iv_matrix   = np.array(iv_rows)[sort_idx]
    labels_sorted = [exp_labels[i] for i in sort_idx]

    return dtes_arr, _STRIKE_PCTS, iv_matrix, spot, labels_sorted


# ─────────────────────────────────────────────────────────────────────────────
# Build surface
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching options data for {ticker}…"):
    dtes, strike_pcts, iv_matrix, spot, exp_labels = _build_iv_matrix(
        ticker, max_exp_p, min_dte_p, max_dte_p
    )

if dtes is None or len(dtes) < 2:
    st.error(
        "Not enough data to build a surface. "
        "Try widening the DTE range or choosing a more liquid ticker (e.g. SPY, QQQ, AAPL)."
    )
    st.stop()

iv_pct = iv_matrix * 100  # display as percentages

# ─────────────────────────────────────────────────────────────────────────────
# Summary row
# ─────────────────────────────────────────────────────────────────────────────
atm_idx = np.argmin(np.abs(strike_pcts - 1.0))
atm_ivs = iv_pct[:, atm_idx]
front_atm = float(atm_ivs[0])
back_atm  = float(atm_ivs[-1])
overall_min = float(np.nanmin(iv_pct))
overall_max = float(np.nanmax(iv_pct))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spot Price", f"${spot:.2f}")
c2.metric("Front ATM IV", f"{front_atm:.1f}%")
c3.metric("Back ATM IV", f"{back_atm:.1f}%")
c4.metric("Surface Min IV", f"{overall_min:.1f}%")
c5.metric("Surface Max IV", f"{overall_max:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🌋 IV Surface", "📈 Term Structure", "⚖️ Skew Analysis"])

# ── Tab 1: IV Surface ─────────────────────────────────────────────────────────
with tab1:
    view_mode = st.radio("View Mode", ["3D Surface", "2D Heatmap"], horizontal=True)

    if view_mode == "3D Surface":
        fig = go.Figure(go.Surface(
            x=strike_pcts * 100,
            y=dtes,
            z=iv_pct,
            colorscale='Viridis',
            colorbar=dict(title="IV %", ticksuffix="%", tickfont=dict(color='#e8edf3')),
            hovertemplate=(
                "Strike: %{x:.1f}%<br>"
                "DTE: %{y}d<br>"
                "IV: %{z:.1f}%<extra></extra>"
            )
        ))
        fig.update_layout(
            **dark_plotly_layout(
                height=620,
                title=f"{ticker} Implied Volatility Surface",
                scene=dict(
                    xaxis=dict(title="Strike %", ticksuffix="%",
                               gridcolor='rgba(107,122,143,0.2)', color='#e8edf3'),
                    yaxis=dict(title="Days to Expiry",
                               gridcolor='rgba(107,122,143,0.2)', color='#e8edf3'),
                    zaxis=dict(title="IV %", ticksuffix="%",
                               gridcolor='rgba(107,122,143,0.2)', color='#e8edf3'),
                    bgcolor='#0e1117',
                ),
                margin=dict(l=0, r=0, t=50, b=0),
            )
        )
    else:
        fig = go.Figure(go.Heatmap(
            x=_STRIKE_LABELS,
            y=[f"{d}d" for d in dtes],
            z=iv_pct,
            colorscale='Viridis',
            colorbar=dict(title="IV %", ticksuffix="%", tickfont=dict(color='#e8edf3')),
            hovertemplate="Strike: %{x}<br>DTE: %{y}<br>IV: %{z:.1f}%<extra></extra>",
            text=np.round(iv_pct, 1),
            texttemplate="%{text:.1f}%",
            textfont=dict(size=10)
        ))
        fig.update_layout(
            **dark_plotly_layout(
                height=max(300, len(dtes) * 45 + 120),
                title=f"{ticker} IV Heatmap (Strike % vs DTE)",
                xaxis_title="Strike (% of Spot)",
                yaxis_title="Days to Expiry",
            )
        )
        # Vertical line at ATM
        fig.add_vline(x=_STRIKE_LABELS[atm_idx],
                      line_color='#00d4aa', line_dash='dash', line_width=1.5,
                      annotation_text="ATM", annotation_font_color='#00d4aa')

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 IV Matrix Data"):
        df_display = pd.DataFrame(
            np.round(iv_pct, 2),
            index=exp_labels,
            columns=_STRIKE_LABELS
        )
        st.dataframe(
            df_display.style.background_gradient(cmap='YlOrRd', axis=None)
                      .format("{:.2f}%"),
            use_container_width=True
        )

# ── Tab 2: Term Structure ─────────────────────────────────────────────────────
with tab2:
    # ATM IV per expiration
    ts_fig = go.Figure()

    ts_fig.add_trace(go.Scatter(
        x=dtes, y=atm_ivs,
        mode='lines+markers',
        name='ATM IV',
        line=dict(color='#00d4aa', width=2.5),
        marker=dict(size=8, color='#00d4aa'),
        hovertemplate="DTE: %{x}d<br>ATM IV: %{y:.1f}%<extra></extra>"
    ))

    # Annotations
    for i, (dte, iv, label) in enumerate(zip(dtes, atm_ivs, exp_labels)):
        if i in (0, len(dtes) - 1) or i == len(dtes) // 2:
            ts_fig.add_annotation(
                x=dte, y=iv,
                text=f"{iv:.1f}%",
                showarrow=True, arrowhead=2, arrowcolor='#00d4aa',
                font=dict(color='#e8edf3', size=10),
                bgcolor='rgba(20,23,32,0.85)',
                bordercolor='rgba(0,212,170,0.3)',
                ay=-30
            )

    ts_fig.update_layout(
        **dark_plotly_layout(
            height=420,
            title=f"{ticker} ATM IV Term Structure",
            xaxis_title="Days to Expiry",
            yaxis_title="ATM Implied Volatility (%)",
        )
    )
    ts_fig.update_xaxes(gridcolor='rgba(107,122,143,0.15)')
    ts_fig.update_yaxes(ticksuffix="%", gridcolor='rgba(107,122,143,0.15)')
    st.plotly_chart(ts_fig, use_container_width=True)

    # Structure metrics
    slope = (back_atm - front_atm) / max(dtes[-1] - dtes[0], 1) * 30  # % per 30 DTE
    structure_label = "Contango (far IV > near IV)" if back_atm > front_atm else "Backwardation (near IV > far IV)"
    structure_color = "#4a9eff" if back_atm > front_atm else "#f59e0b"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Front-Month ATM IV", f"{front_atm:.1f}%")
    m2.metric("Back-Month ATM IV", f"{back_atm:.1f}%")
    m3.metric("IV Slope", f"{slope:+.2f}% / 30d")
    m4.metric("Structure", "↗ Contango" if back_atm > front_atm else "↘ Backwardation")

    st.markdown(
        f'<div style="background:rgba(74,158,255,0.1);border:1px solid {structure_color}55;'
        f'border-radius:8px;padding:10px 18px;font-size:14px;">'
        f'<strong>Term Structure:</strong> <span style="color:{structure_color}">{structure_label}</span> — '
        f'ATM IV slope = {slope:+.2f}% per 30 DTE</div>',
        unsafe_allow_html=True
    )

    # Term structure table
    with st.expander("📋 Term Structure Details"):
        ts_df = pd.DataFrame({
            'Expiration': exp_labels,
            'DTE': dtes,
            'ATM IV (%)': np.round(atm_ivs, 2),
        })
        st.dataframe(ts_df.style.format({'ATM IV (%)': '{:.2f}%'}),
                     use_container_width=True, hide_index=True)

# ── Tab 3: Skew Analysis ──────────────────────────────────────────────────────
with tab3:
    # Skew = put 95% IV - call 105% IV
    put_95_idx  = np.argmin(np.abs(strike_pcts - 0.95))
    call_105_idx = np.argmin(np.abs(strike_pcts - 1.05))

    skews = iv_pct[:, put_95_idx] - iv_pct[:, call_105_idx]

    bar_colors = ['#22c55e' if s >= 0 else '#f59e0b' for s in skews]

    skew_fig = go.Figure(go.Bar(
        x=[f"{d}d" for d in dtes],
        y=skews,
        marker_color=bar_colors,
        text=[f"{s:+.1f}%" for s in skews],
        textposition='outside',
        hovertemplate="DTE: %{x}<br>Skew: %{y:+.2f}%<extra></extra>"
    ))

    skew_fig.add_hline(y=0, line_color='#6b7a8f', line_dash='dash', line_width=1)

    skew_fig.update_layout(
        **dark_plotly_layout(
            height=400,
            title=f"{ticker} Put-Call Skew by Expiration  (95% Put IV − 105% Call IV)",
            xaxis_title="Expiration",
            yaxis_title="Skew (IV %)",
        )
    )
    skew_fig.update_yaxes(ticksuffix="%", gridcolor='rgba(107,122,143,0.15)')
    st.plotly_chart(skew_fig, use_container_width=True)

    # Skew metrics
    avg_skew = float(np.mean(skews))
    max_skew_idx = int(np.argmax(skews))
    min_skew_idx = int(np.argmin(skews))

    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("Average Skew", f"{avg_skew:+.2f}%",
               "Put premium" if avg_skew > 0 else "Call premium")
    sm2.metric("Highest Put Skew",
               f"{skews[max_skew_idx]:+.2f}%",
               f"At {exp_labels[max_skew_idx]}")
    sm3.metric("Lowest Skew",
               f"{skews[min_skew_idx]:+.2f}%",
               f"At {exp_labels[min_skew_idx]}")

    skew_interp = (
        "**Put-skewed surface** — Options market is paying up for downside protection. "
        "Consistent with elevated crash/tail risk concern."
        if avg_skew > 1.5 else
        "**Call-skewed surface** — Upside calls trading at a premium. "
        "May reflect demand for upside leverage or covered-call selling on puts."
        if avg_skew < -1.5 else
        "**Balanced skew** — Puts and calls trading at roughly similar volatilities. "
        "Market is not pricing in a strong directional move."
    )
    st.info(skew_interp)

    # Per-expiration skew table
    with st.expander("📋 Skew by Expiration"):
        skew_df = pd.DataFrame({
            'Expiration': exp_labels,
            'DTE': dtes,
            f'95% Put IV (%)': np.round(iv_pct[:, put_95_idx], 2),
            f'105% Call IV (%)': np.round(iv_pct[:, call_105_idx], 2),
            'Skew (%)': np.round(skews, 2),
        })
        st.dataframe(
            skew_df.style.format({
                f'95% Put IV (%)': '{:.2f}%',
                f'105% Call IV (%)': '{:.2f}%',
                'Skew (%)': '{:+.2f}%',
            }),
            use_container_width=True, hide_index=True
        )
