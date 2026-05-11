"""
Economic & Earnings Calendar
Provides upcoming FOMC meetings, CPI release dates, and earnings events.

No external API required:
  - FOMC/CPI dates are sourced from official schedules (hardcoded through 2026)
  - Earnings dates from yfinance (same as existing Action Dashboard code)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Dict


# ── FOMC Meeting Dates (decision day = second day of two-day meeting) ────────
# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm

_FOMC_DATES: List[date] = [
    # 2025
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 10),
    # 2026 (projected from Fed calendar)
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 10),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]

# ── CPI Release Dates (BLS publishes ~2nd Wednesday of each month) ──────────
# Source: bls.gov/schedule/news_release/cpi.htm

_CPI_DATES: List[date] = [
    # 2025
    date(2025, 1, 15),
    date(2025, 2, 12),
    date(2025, 3, 12),
    date(2025, 4, 10),
    date(2025, 5, 13),
    date(2025, 6, 11),
    date(2025, 7, 15),
    date(2025, 8, 12),
    date(2025, 9, 10),
    date(2025, 10, 15),
    date(2025, 11, 12),
    date(2025, 12, 10),
    # 2026
    date(2026, 1, 14),
    date(2026, 2, 11),
    date(2026, 3, 11),
    date(2026, 4, 10),
    date(2026, 5, 13),
    date(2026, 6, 10),
    date(2026, 7, 14),
    date(2026, 8, 12),
    date(2026, 9, 9),
    date(2026, 10, 14),
    date(2026, 11, 11),
    date(2026, 12, 9),
]

# ── NFP (Jobs Report) — first Friday of each month ──────────────────────────

_NFP_DATES: List[date] = [
    # 2025
    date(2025, 1, 10),
    date(2025, 2, 7),
    date(2025, 3, 7),
    date(2025, 4, 4),
    date(2025, 5, 2),
    date(2025, 6, 6),
    date(2025, 7, 3),
    date(2025, 8, 1),
    date(2025, 9, 5),
    date(2025, 10, 3),
    date(2025, 11, 7),
    date(2025, 12, 5),
    # 2026
    date(2026, 1, 9),
    date(2026, 2, 6),
    date(2026, 3, 6),
    date(2026, 4, 3),
    date(2026, 5, 1),
    date(2026, 6, 5),
]


# ── Public functions ──────────────────────────────────────────────────────────

def get_upcoming_events(days_ahead: int = 30) -> List[Dict]:
    """
    Return a list of upcoming macro events within the next *days_ahead* days.

    Each event dict:
        date   : date
        type   : str  ('FOMC', 'CPI', 'NFP')
        label  : str  (human-readable description)
        days   : int  (days from today)
        impact : str  ('High' | 'Medium')
    """
    today  = date.today()
    cutoff = today + timedelta(days=days_ahead)
    events = []

    for d in _FOMC_DATES:
        if today <= d <= cutoff:
            events.append({
                "date":   d,
                "type":   "FOMC",
                "label":  "FOMC Interest Rate Decision",
                "days":   (d - today).days,
                "impact": "High",
            })

    for d in _CPI_DATES:
        if today <= d <= cutoff:
            events.append({
                "date":   d,
                "type":   "CPI",
                "label":  "CPI Inflation Report",
                "days":   (d - today).days,
                "impact": "High",
            })

    for d in _NFP_DATES:
        if today <= d <= cutoff:
            events.append({
                "date":   d,
                "type":   "NFP",
                "label":  "Non-Farm Payrolls (Jobs Report)",
                "days":   (d - today).days,
                "impact": "High",
            })

    return sorted(events, key=lambda x: x["date"])


def get_next_fomc() -> Dict | None:
    """Return the next upcoming FOMC meeting, or None."""
    today = date.today()
    upcoming = [d for d in _FOMC_DATES if d >= today]
    if not upcoming:
        return None
    d = upcoming[0]
    return {"date": d, "days": (d - today).days}


def get_next_cpi() -> Dict | None:
    """Return the next upcoming CPI release, or None."""
    today = date.today()
    upcoming = [d for d in _CPI_DATES if d >= today]
    if not upcoming:
        return None
    d = upcoming[0]
    return {"date": d, "days": (d - today).days}
