"""
Institutional-grade number formatting utilities.
Used across all pages for consistent compact display.
"""
from __future__ import annotations

# ── Palette ──────────────────────────────────────────────────────────────────
GAIN = "#4ade80"
LOSS = "#fb7185"
DIM  = "#4b5563"


# ── Money ────────────────────────────────────────────────────────────────────

def fmt_money(val: float, compact: bool = True) -> str:
    """$1.24M · $34.5K · $1,234.56"""
    if val is None:
        return "—"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "—"
    if compact:
        if abs(val) >= 1_000_000:
            return f"${val / 1_000_000:.2f}M"
        if abs(val) >= 10_000:
            return f"${val / 1_000:.1f}K"
    return f"${val:,.2f}"


def fmt_money_full(val: float) -> str:
    """Always full precision: $1,234,567.89"""
    if val is None:
        return "—"
    try:
        return f"${float(val):,.2f}"
    except (TypeError, ValueError):
        return "—"


# ── Percentages ──────────────────────────────────────────────────────────────

def fmt_pct(val: float, decimals: int = 2, arrow: bool = True) -> str:
    """▲ 2.34%  ·  ▼ 1.23%  ·  — 0.00%"""
    if val is None:
        return "—"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "—"
    if arrow:
        sym = "▲" if val > 0 else ("▼" if val < 0 else "—")
        return f"{sym} {abs(val):.{decimals}f}%"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.{decimals}f}%"


def fmt_bps(val: float) -> str:
    """Format a decimal rate change as basis points: 0.0025 → +25 bps"""
    if val is None:
        return "—"
    try:
        bps = float(val) * 10_000
        sign = "+" if bps > 0 else ""
        return f"{sign}{bps:.0f} bps"
    except (TypeError, ValueError):
        return "—"


# ── Plain numbers ─────────────────────────────────────────────────────────────

def fmt_number(val: float, decimals: int = 2) -> str:
    """1,234.56"""
    if val is None:
        return "—"
    try:
        return f"{float(val):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


# ── Color helpers ─────────────────────────────────────────────────────────────

def pnl_color(val: float) -> str:
    """Return green/red/dim based on sign of val."""
    if val is None:
        return DIM
    try:
        v = float(val)
    except (TypeError, ValueError):
        return DIM
    if v > 0:
        return GAIN
    if v < 0:
        return LOSS
    return DIM


def pnl_bg(pct: float, max_pct: float = 25.0, alpha: float = 0.18) -> str:
    """
    RGBA background for a P&L heatmap table cell.
    Green for gains, red for losses, intensity proportional to |pct|.
    """
    try:
        pct = float(pct)
    except (TypeError, ValueError):
        return "transparent"
    if pct == 0:
        return "transparent"
    intensity = min(abs(pct) / max_pct, 1.0) * alpha
    if pct > 0:
        return f"rgba(74,222,128,{intensity:.3f})"
    return f"rgba(251,113,133,{intensity:.3f})"


# ── Sparkline SVG ─────────────────────────────────────────────────────────────

def sparkline_svg(
    prices: list[float],
    profit: bool,
    width: int = 64,
    height: int = 22,
) -> str:
    """
    Inline SVG sparkline from a list of closing prices.
    profit=True → cyan/green line; False → red line.
    Returns empty string when there aren't enough points.
    """
    pts = [float(p) for p in prices if p is not None]
    if len(pts) < 2:
        return ""
    mn, mx = min(pts), max(pts)
    rng = mx - mn if mx != mn else 1e-9
    n = len(pts)
    coords = " ".join(
        f"{i / (n - 1) * width:.1f},"
        f"{height - 2 - (p - mn) / rng * (height - 4):.1f}"
        for i, p in enumerate(pts)
    )
    color = "#22d3ee" if profit else "#fb7185"
    # Fill under the line with a subtle gradient
    first_x = "0"
    last_x  = str(width)
    fill_pts = f"{first_x},{height} {coords} {last_x},{height}"
    return (
        f'<svg width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'style="display:inline-block;vertical-align:middle;overflow:visible;">'
        f'<polygon points="{fill_pts}" '
        f'fill="{color}" fill-opacity="0.08"/>'
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )
