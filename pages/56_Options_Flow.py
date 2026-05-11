"""
Regime-Aware Portfolio Manager
Section 11: Options Flow — Unusual Activity & Informed Trade Detection
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
from utils.carbon_theme import apply_carbon_theme, carbon_plotly_layout, page_header

st.set_page_config(page_title="Options Flow", page_icon="🌊", layout="wide")
apply_carbon_theme()

page_header("🌊 Options Flow", "Surface unusual options activity and potentially informed trades")

options_loader = OptionsDataLoader()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Flow Scanner Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()

n_exps = st.sidebar.selectbox(
    "Expirations to Scan",
    options=[3, 5, 8, "All"],
    index=0,
    help="Number of nearest expirations to scan"
)

vol_oi_thresh = st.sidebar.slider(
    "Volume/OI Threshold",
    min_value=1.0, max_value=10.0, value=2.0, step=0.5,
    help="Flag options where volume is this many times the open interest"
)

min_volume = st.sidebar.number_input(
    "Min Volume Filter",
    min_value=10, max_value=1000, value=100, step=10,
    help="Only show options with at least this much volume"
)

if st.sidebar.button("🔍 Scan Flow", type="primary", use_container_width=True):
    st.session_state['flow_ticker']  = ticker
    st.session_state['flow_params']  = (n_exps, vol_oi_thresh, int(min_volume))
    st.session_state['flow_loaded']  = True

if not st.session_state.get('flow_loaded', False):
    st.info("👈 Set your scanner parameters and click **Scan Flow** to detect unusual activity.")
    st.markdown("""
    ### What Unusual Flow Means

    **Volume/OI Ratio > threshold** suggests a large number of new contracts being
    opened today relative to total existing open interest — potentially informed traders
    taking a position ahead of an anticipated move.

    **Bought vs Sold classification** uses the last price relative to the bid-ask midpoint:
    - *Bought* → last price ≥ mid price (paid the ask, aggressive buyer)
    - *Sold* → last price < mid price (sold at bid, aggressive seller)

    **Dollar Premium** = Volume × Last Price × 100 (total dollars transacted)
    """)
    st.stop()

ticker      = st.session_state['flow_ticker']
n_exps_p, vol_oi_p, min_vol_p = st.session_state['flow_params']


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=180, show_spinner="Scanning options flow…")
def _get_flow(ticker: str, n_exps, vol_oi_thresh: float, min_volume: int):
    """Fetch options data and detect unusual activity."""
    from data.options_data import OptionsDataLoader
    loader = OptionsDataLoader()

    expirations = loader.get_options_expirations(ticker)
    if not expirations:
        return pd.DataFrame(), {}

    if n_exps == "All":
        exp_list = expirations
    else:
        exp_list = expirations[:int(n_exps)]

    now = datetime.now()
    all_rows = []

    for exp in exp_list:
        try:
            dte = (datetime.strptime(exp, '%Y-%m-%d') - now).days
        except ValueError:
            continue

        calls, puts, underlying = loader.get_options_chain(ticker, exp)

        for df_opt, otype in [(calls, 'Call'), (puts, 'Put')]:
            if df_opt.empty:
                continue

            for _, row in df_opt.iterrows():
                def _safe_float(val, default=0.0):
                    try:
                        v = float(val)
                        return v if np.isfinite(v) else default
                    except (TypeError, ValueError):
                        return default

                vol    = _safe_float(row.get('volume'), 0.0)
                oi     = _safe_float(row.get('openInterest'), 0.0)
                last   = _safe_float(row.get('lastPrice'), 0.0)
                bid    = _safe_float(row.get('bid'), 0.0)
                ask    = _safe_float(row.get('ask'), 0.0)
                iv     = _safe_float(row.get('impliedVolatility'), 0.0)
                strike = _safe_float(row.get('strike'), 0.0)

                if vol < min_volume:
                    continue

                # Volume/OI ratio
                if oi > 0:
                    vol_oi_ratio = vol / oi
                    unusual = vol_oi_ratio >= vol_oi_thresh
                else:
                    vol_oi_ratio = np.nan
                    unusual = True  # New position (no prior OI)

                # Side classification
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                side = 'Bought' if last >= mid - 0.01 else 'Sold'

                dollar_prem = vol * last * 100

                all_rows.append({
                    'Type':          otype,
                    'Strike':        strike,
                    'Expiry':        exp,
                    'DTE':           dte,
                    'Last':          last,
                    'Bid':           bid,
                    'Ask':           ask,
                    'IV (%)':        round(iv * 100, 1),
                    'Volume':        int(vol) if np.isfinite(vol) else 0,
                    'OI':            int(oi)  if np.isfinite(oi)  else 0,
                    'Vol/OI':        round(vol_oi_ratio, 2) if not np.isnan(vol_oi_ratio) else None,
                    'Side':          side,
                    'Dollar Premium': dollar_prem,
                    'Unusual':       unusual,
                    'Underlying':    underlying,
                })

    if not all_rows:
        return pd.DataFrame(), {}

    df = pd.DataFrame(all_rows)

    # Summary
    call_df  = df[df['Type'] == 'Call']
    put_df   = df[df['Type'] == 'Put']
    summary  = {
        'total_call_prem': call_df['Dollar Premium'].sum(),
        'total_put_prem':  put_df['Dollar Premium'].sum(),
        'total_rows':      len(df),
        'unusual_rows':    int(df['Unusual'].sum()),
        'underlying':      float(df['Underlying'].iloc[0]) if not df.empty else 0,
    }
    cp_total = summary['total_call_prem'] + summary['total_put_prem']
    summary['flow_ratio'] = (summary['total_call_prem'] / cp_total) if cp_total > 0 else 0.5

    # Return only unusual rows sorted by dollar premium
    unusual_df = df[df['Unusual']].sort_values('Dollar Premium', ascending=False)
    return unusual_df.reset_index(drop=True), summary


# ─────────────────────────────────────────────────────────────────────────────
# Execute scan
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"Scanning {ticker} options flow…"):
    flow_df, summary = _get_flow(ticker, n_exps_p, vol_oi_p, min_vol_p)

if flow_df.empty:
    st.warning(
        f"No unusual flow detected for **{ticker}** with current settings. "
        "Try lowering the Vol/OI threshold or Min Volume, or scan more expirations."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Summary metrics row
# ─────────────────────────────────────────────────────────────────────────────
st.subheader(f"📊 {ticker} Flow Summary — ${summary['underlying']:.2f}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Call Premium", f"${summary['total_call_prem']:,.0f}")
m2.metric("Total Put Premium",  f"${summary['total_put_prem']:,.0f}")

ratio = summary['flow_ratio']
ratio_label = "📈 Call-Heavy" if ratio > 0.6 else "📉 Put-Heavy" if ratio < 0.4 else "⚖️ Balanced"
m3.metric("Call/Total Flow", f"{ratio:.1%}", ratio_label)
m4.metric("Unusual Trades",  f"{summary['unusual_rows']}")
m5.metric("Scanned Options", f"{summary['total_rows']}")

# Flow bar
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=[summary['total_call_prem']], y=['Flow'],
    orientation='h', name='Call Premium',
    marker_color='#22d3ee',
    text=f"${summary['total_call_prem']/1e6:.1f}M calls",
    textposition='inside', insidetextanchor='start'
))
fig_bar.add_trace(go.Bar(
    x=[summary['total_put_prem']], y=['Flow'],
    orientation='h', name='Put Premium',
    marker_color='#fb7185',
    text=f"${summary['total_put_prem']/1e6:.1f}M puts",
    textposition='inside', insidetextanchor='end'
))
fig_bar.update_layout(
    **carbon_plotly_layout(
        height=120, barmode='stack',
        xaxis_title="Dollar Premium ($)",
        showlegend=True,
        legend=dict(orientation='h', y=1.4,
                    bgcolor='rgba(20,23,32,0.85)',
                    bordercolor='rgba(0,212,170,0.35)', borderwidth=1),
        margin=dict(l=50, r=30, t=30, b=30),
    )
)
fig_bar.update_xaxes(tickprefix="$", tickformat=",.0f",
                     gridcolor='rgba(107,122,143,0.15)')
st.plotly_chart(fig_bar, use_container_width=True)

# Interpretation
if ratio > 0.65:
    interp = "**Call-dominated flow** — Majority of dollar premium is in calls. Bullish bias."
    interp_color = "#22d3ee"
elif ratio < 0.35:
    interp = "**Put-dominated flow** — Majority of dollar premium is in puts. Bearish/protective bias."
    interp_color = "#fb7185"
else:
    interp = "**Balanced flow** — Roughly equal call and put premium. No strong directional bias."
    interp_color = "#f59e0b"

st.markdown(
    f'<div style="background:rgba(255,255,255,0.03);border-left:3px solid {interp_color};'
    f'border-radius:4px;padding:8px 16px;margin-bottom:16px;">{interp}</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# Unusual Activity Table
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🚨 Unusual Activity Table")
st.markdown(f"Showing **{len(flow_df)}** trades above threshold "
            f"(Vol/OI ≥ {vol_oi_p}×, Volume ≥ {min_vol_p})")

# Format for display
display_df = flow_df[[
    'Type', 'Strike', 'Expiry', 'DTE', 'Last', 'Bid', 'Ask',
    'IV (%)', 'Volume', 'OI', 'Vol/OI', 'Side', 'Dollar Premium'
]].copy()

display_df['Dollar Premium'] = display_df['Dollar Premium'].apply(
    lambda x: f"${x/1_000:.1f}K" if x < 1_000_000 else f"${x/1_000_000:.2f}M"
)

def _row_style(row):
    base_color = '#0a1c22' if row['Type'] == 'Call' else '#1c0a12'
    return [f'background-color: {base_color}'] * len(row)

styled_table = (
    display_df.style
    .apply(_row_style, axis=1)
    .format({
        'Strike': '${:.0f}',
        'Last':   '${:.2f}',
        'Bid':    '${:.2f}',
        'Ask':    '${:.2f}',
        'IV (%)': '{:.1f}%',
        'Volume': '{:,}',
        'OI':     '{:,}',
        'Vol/OI': lambda x: f'{x:.1f}×' if x is not None and not (isinstance(x, float) and np.isnan(x)) else 'New',
    })
)

st.dataframe(styled_table, use_container_width=True, hide_index=True, height=500)

# ─────────────────────────────────────────────────────────────────────────────
# Top 10 Leaderboard
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🏆 Top 10 Largest Trades by Dollar Premium")

top10 = flow_df.head(10).copy()

for rank, (_, row) in enumerate(top10.iterrows(), 1):
    type_color  = '#22d3ee' if row['Type'] == 'Call' else '#fb7185'
    side_color  = '#22d3ee' if row['Side'] == 'Bought' else '#f59e0b'
    dollar_str  = (f"${row['Dollar Premium']/1_000:.1f}K"
                   if row['Dollar Premium'] < 1_000_000
                   else f"${row['Dollar Premium']/1_000_000:.2f}M")
    vol_oi_str = (f"{row['Vol/OI']:.1f}×" if row['Vol/OI'] is not None
                  and not (isinstance(row['Vol/OI'], float) and np.isnan(row['Vol/OI']))
                  else "New OI")

    st.markdown(
        f"""<div style="background:#1a1a1a;border:1px solid rgba(34,211,238,0.15);
        border-radius:10px;padding:12px 18px;margin-bottom:8px;display:flex;
        align-items:center;gap:16px;">
        <span style="font-size:22px;font-weight:700;color:#888888;min-width:32px">#{rank}</span>
        <span style="background:{type_color}22;color:{type_color};padding:2px 10px;
        border-radius:12px;font-weight:700;font-size:13px;">{row['Type'].upper()}</span>
        <span style="background:{side_color}22;color:{side_color};padding:2px 10px;
        border-radius:12px;font-weight:600;font-size:13px;">{row['Side']}</span>
        <span style="font-weight:700;font-size:15px;color:#ffffff;">
            ${row['Strike']:.0f} strike — {row['Expiry']} ({row['DTE']}d)
        </span>
        <span style="margin-left:auto;text-align:right;">
            <span style="font-size:18px;font-weight:700;color:#ffffff;">{dollar_str}</span>
            <span style="font-size:12px;color:#888888;margin-left:8px;">
                Vol {row['Volume']:,} / OI {row['OI']:,} ({vol_oi_str})
                &nbsp;|&nbsp; IV {row['IV (%)']:.1f}%
            </span>
        </span></div>""",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# Volume Heatmap
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Volume Heatmap — Strike × Expiration")

# Bin strikes into % buckets
if not flow_df.empty and 'Underlying' in flow_df.columns:
    spot = float(flow_df['Underlying'].iloc[0])
    bins = [0.80, 0.85, 0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10, 1.15, 1.20]
    bin_labels = [f"{int(b*100)}%" for b in bins[:-1]]

    flow_df['strike_pct'] = flow_df['Strike'] / spot
    flow_df['strike_bin'] = pd.cut(
        flow_df['strike_pct'], bins=bins, labels=bin_labels, include_lowest=True
    )

    pivot = (
        flow_df.groupby(['Expiry', 'strike_bin'], observed=False)['Dollar Premium']
        .sum()
        .unstack(fill_value=0)
    )

    if not pivot.empty:
        pivot_vals = pivot.values / 1_000  # in $K

        heat_fig = go.Figure(go.Heatmap(
            x=list(pivot.columns),
            y=[f"{idx} ({(datetime.strptime(idx, '%Y-%m-%d') - datetime.now()).days}d)"
               for idx in pivot.index],
            z=pivot_vals,
            colorscale='Plasma',
            colorbar=dict(title="$K Premium", tickprefix="$",
                          ticksuffix="K", tickfont=dict(color='#ffffff')),
            hovertemplate="Strike: %{x}<br>Expiry: %{y}<br>Premium: $%{z:,.0f}K<extra></extra>",
        ))
        heat_fig.update_layout(
            **carbon_plotly_layout(
                height=max(300, len(pivot) * 50 + 120),
                title=f"{ticker} Dollar Premium Heatmap (Strike % of Spot × Expiration)",
                xaxis_title="Strike (% of Spot)",
                yaxis_title="Expiration",
            )
        )
        st.plotly_chart(heat_fig, use_container_width=True)
    else:
        st.info("Not enough data to build heatmap with current filters.")
