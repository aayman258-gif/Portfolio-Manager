"""
Regime Detection Engine — v2 (6-Signal Macro-Enhanced Model)
=============================================================

Replaces the 3-signal price-only model with a 6-signal model that combines
price dynamics with FRED macro indicators for higher-confidence classification.

Signals
───────
  Price-based (always available):
    1. vol_signal    — realized-volatility percentile (20-day, annualised)
    2. trend_signal  — 50-day MA slope normalised to [0, 1]

  Macro-based (from FRED; fall back gracefully when unavailable):
    3. curve_signal  — yield-curve spread (10Y-2Y) rolling percentile
    4. credit_signal — HY OAS rolling percentile (0=tight, 1=wide)
    5. infl_signal   — CPI YoY rolling percentile (0=deflation, 1=high)
    6. rrate_signal  — real-rate (Fed Funds − CPI) rolling percentile

Regimes (6 named + Uncertain)
──────────────────────────────
  Risk-On        low vol · uptrend · steep curve · tight credit
  Caution        moderate vol · mixed trend · flattening curve
  High Volatility elevated vol · wide credit spreads (acute stress)
  Stagflation    high inflation · negative real rates · weak/flat trend
  Recession      inverted curve · wide credit · downtrend
  Mean Reversion range-bound · no trend · moderate vol
  Uncertain      no regime clears the confidence threshold (0.40)

Methodology
───────────
Each signal is normalised to [0, 1] via a rolling 252-day (or 36-month)
percentile so the scoring is self-calibrating relative to recent history.

Each regime is scored by a weighted linear combination of signals.
The weights represent how much each signal contributes to that regime.
Scores are re-scaled so that a "perfect" regime match → 1.0.

When FRED data is unavailable only price signals are used; each regime's
score is re-normalised to its maximum achievable price-only value.

Backward compatibility
──────────────────────
Public interface unchanged:
    detector.classify_regime(prices, vix=None)  →  (regime_series, signals_df)
    detector.get_current_confidence(signals_df) →  dict
    detector.get_regime_description(regime_str) →  dict
    detector.get_regime_stats(regime_series)    →  DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _stats
from typing import Dict, Optional, Tuple


# ── Constants ─────────────────────────────────────────────────────────────────

UNCERTAIN_THRESHOLD = 0.40   # minimum score to assign a named regime

_NAMED_REGIMES = [
    "Risk-On", "Caution", "High Volatility",
    "Stagflation", "Recession", "Mean Reversion",
]

# ── Regime scoring weights ────────────────────────────────────────────────────
# Each tuple: (vol_w, trend_w, curve_w, credit_w, infl_w, rrate_w)
# Weights within a regime sum to 1.0.
# Price-only columns (vol, trend) are always populated.
# Macro columns may be 0.5 (neutral) when FRED is unavailable.

_WEIGHTS: Dict[str, tuple] = {
    #                vol    trend  curve  credit  infl  rrate
    "Risk-On":      (0.25,  0.25,  0.22,  0.18,   0.05, 0.05),
    "Caution":      (0.25,  0.20,  0.20,  0.20,   0.10, 0.05),
    "High Volatility":(0.40, 0.15, 0.05,  0.30,   0.05, 0.05),
    "Stagflation":  (0.10,  0.15,  0.20,  0.10,   0.30, 0.15),
    "Recession":    (0.15,  0.20,  0.30,  0.25,   0.05, 0.05),
    "Mean Reversion":(0.25,  0.25, 0.20,  0.20,   0.05, 0.05),
}

# "Ideal" normalised signal values for each regime.
# A signal == ideal_value contributes its full weight; distance is penalised.
#                vol   trend  curve  credit  infl  rrate
_IDEAL: Dict[str, tuple] = {
    "Risk-On":       (0.10, 0.85,  0.80,  0.10,  0.30, 0.60),
    "Caution":       (0.55, 0.40,  0.35,  0.55,  0.50, 0.50),
    "High Volatility":(0.90, 0.30, 0.30,  0.85,  0.50, 0.40),
    "Stagflation":   (0.55, 0.30,  0.25,  0.55,  0.85, 0.15),
    "Recession":     (0.75, 0.15,  0.10,  0.85,  0.35, 0.65),
    "Mean Reversion":(0.45, 0.50,  0.50,  0.35,  0.40, 0.45),
}


# ── Rolling percentile helper ─────────────────────────────────────────────────

def _roll_pct(s: pd.Series, window: int) -> pd.Series:
    """Rolling percentile of the current value within its own *window* history."""
    def _pct(x):
        v = x.iloc[-1]
        arr = x.dropna().values
        if len(arr) < 10:
            return np.nan
        return float(_stats.percentileofscore(arr, v, kind="rank") / 100)
    return s.rolling(window, min_periods=max(20, window // 4)).apply(_pct, raw=False)


# ── Core scoring ──────────────────────────────────────────────────────────────

def _score_regimes(
    vol: float,
    trend: float,
    curve: float = 0.5,
    credit: float = 0.5,
    infl: float = 0.5,
    rrate: float = 0.5,
    macro_available: bool = False,
) -> Dict[str, float]:
    """
    Return a score in [0, 1] for each named regime.

    Scoring: for each regime the score is a weighted sum where each weight is
    multiplied by  (1 − |signal − ideal|)  — i.e. proximity to the ideal value.
    When macro is unavailable the macro weights are redistributed to price signals.
    """
    signals = np.array([vol, trend, curve, credit, infl, rrate])
    scores  = {}

    for name in _NAMED_REGIMES:
        ideal  = np.array(_IDEAL[name])
        w_full = np.array(_WEIGHTS[name])

        if not macro_available:
            # Collapse macro weights back onto price signals proportionally
            price_w = w_full[:2].sum()
            macro_w = w_full[2:].sum()
            w = w_full.copy()
            w[0] += macro_w * (w_full[0] / price_w) if price_w else macro_w / 2
            w[1] += macro_w * (w_full[1] / price_w) if price_w else macro_w / 2
            w[2:] = 0.0
        else:
            w = w_full

        proximity = 1.0 - np.abs(signals - ideal)   # [0, 1] per signal
        scores[name] = float(np.dot(w, proximity))

    return scores


# ── RegimeDetector ────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Market regime classifier — macro-enhanced 6-signal model.

    Parameters
    ----------
    lookback_vol   : rolling window (days) for realized-vol calculation
    lookback_trend : rolling window (days) for MA trend detection
    """

    def __init__(self, lookback_vol: int = 20, lookback_trend: int = 50):
        self.lookback_vol   = lookback_vol
        self.lookback_trend = lookback_trend

    # ── Signal calculators ────────────────────────────────────────────────────

    def calculate_realized_volatility(self, prices: pd.Series) -> pd.Series:
        returns = prices.pct_change()
        return returns.rolling(self.lookback_vol).std() * np.sqrt(252)

    def detect_trend(self, prices: pd.Series) -> pd.Series:
        """MA slope normalised: -1 = strong down, 0 = flat, +1 = strong up."""
        ma = prices.rolling(self.lookback_trend).mean()
        slope = ma.diff(10) / ma
        return slope.clip(-0.05, 0.05) / 0.05   # normalise to [-1, 1]

    def calculate_entropy(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Shannon entropy of return distribution (kept for backward compat)."""
        from scipy.stats import entropy as _ent
        def _window_ent(x):
            if len(x) < 5:
                return np.nan
            hist, _ = np.histogram(x, bins=10, density=True)
            hist = hist[hist > 0]
            return float(_ent(hist))
        return returns.rolling(window).apply(_window_ent, raw=True)

    # ── Main classifier ───────────────────────────────────────────────────────

    def classify_regime(
        self,
        prices: pd.Series,
        vix: Optional[pd.Series] = None,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Classify each bar into a regime.

        Parameters
        ----------
        prices   : daily close prices (DatetimeIndex)
        vix      : aligned VIX series (optional, for display)
        macro_df : aligned FRED macro DataFrame from fred_client.align_macro_to_index()
                   If None, automatically fetched (cached); falls back to price-only.

        Returns
        -------
        regime  : pd.Series of regime labels
        signals : pd.DataFrame with all signals + confidence columns
        """
        # ── Try to get macro data ─────────────────────────────────────────────
        if macro_df is None:
            try:
                from data.fred_client import get_all_macro, align_macro_to_index
                _macro_raw = get_all_macro()
                macro_df   = align_macro_to_index(_macro_raw, prices.index)
            except Exception:
                macro_df = pd.DataFrame(index=prices.index)

        has_macro = not macro_df.empty and "T10Y2Y" in macro_df.columns

        # ── Price-based signals ───────────────────────────────────────────────
        realized_vol  = self.calculate_realized_volatility(prices)
        returns       = prices.pct_change()
        entropy       = self.calculate_entropy(returns)
        trend_raw     = self.detect_trend(prices)    # [-1, 1]

        vol_pct   = _roll_pct(realized_vol.dropna(), 252).reindex(prices.index)
        trend_nrm = ((trend_raw + 1) / 2).clip(0, 1)   # [-1,1] → [0,1]

        # ── Macro signals (rolling percentile over available history) ─────────
        def _macro_pct(col: str, window: int = 756) -> pd.Series:
            if not has_macro or col not in macro_df.columns:
                return pd.Series(0.5, index=prices.index)
            s = macro_df[col].dropna()
            if s.empty:
                return pd.Series(0.5, index=prices.index)
            return _roll_pct(s, min(window, max(60, len(s) // 2))).reindex(prices.index, method="ffill").fillna(0.5)

        curve_pct  = _macro_pct("T10Y2Y")       # 0=inverted, 1=steep
        credit_pct = _macro_pct("BAMLH0A0HYM2") # 0=tight, 1=wide
        infl_pct   = _macro_pct("CPI_YOY")      # 0=low/deflation, 1=high
        rrate_pct  = _macro_pct("REAL_RATE")     # 0=accommodative, 1=restrictive

        # ── Bar-by-bar classification ─────────────────────────────────────────
        n = len(prices)
        regime      = pd.Series("Uncertain", index=prices.index)
        confidence  = pd.Series(0.0,         index=prices.index)
        regime_conf = {r: pd.Series(np.nan,  index=prices.index) for r in _NAMED_REGIMES}

        for i in range(n):
            v  = vol_pct.iloc[i]
            tr = trend_nrm.iloc[i]
            if pd.isna(v) or pd.isna(tr):
                continue

            cu = float(curve_pct.iloc[i])  if not pd.isna(curve_pct.iloc[i])  else 0.5
            cr = float(credit_pct.iloc[i]) if not pd.isna(credit_pct.iloc[i]) else 0.5
            inf= float(infl_pct.iloc[i])   if not pd.isna(infl_pct.iloc[i])   else 0.5
            rr = float(rrate_pct.iloc[i])  if not pd.isna(rrate_pct.iloc[i])  else 0.5

            scores = _score_regimes(
                vol=float(v), trend=float(tr),
                curve=cu, credit=cr, infl=inf, rrate=rr,
                macro_available=has_macro,
            )

            for r in _NAMED_REGIMES:
                regime_conf[r].iloc[i] = scores[r]

            best_r    = max(scores, key=scores.get)
            best_conf = scores[best_r]

            if best_conf >= UNCERTAIN_THRESHOLD:
                regime.iloc[i]     = best_r
                confidence.iloc[i] = best_conf
            else:
                regime.iloc[i]     = "Uncertain"
                confidence.iloc[i] = best_conf

        # ── Build signals DataFrame ───────────────────────────────────────────
        signals = pd.DataFrame({
            "price":           prices,
            "realized_vol":    realized_vol,
            "vol_percentile":  vol_pct,
            "entropy":         entropy,
            "trend":           trend_raw,           # raw [-1,1] for compat
            "trend_signal":    trend_nrm,
            "regime":          regime,
            "confidence":      confidence,
            "macro_available": has_macro,
        })

        # Macro signals (for display)
        if has_macro:
            for col in ["T10Y2Y", "DGS10", "DGS2", "BAMLH0A0HYM2",
                        "FEDFUNDS", "CPI_YOY", "REAL_RATE", "UNRATE"]:
                if col in macro_df.columns:
                    signals[col] = macro_df[col]
            signals["curve_pct"]  = curve_pct
            signals["credit_pct"] = credit_pct
            signals["infl_pct"]   = infl_pct
            signals["rrate_pct"]  = rrate_pct

        # Per-regime confidence columns
        for r in _NAMED_REGIMES:
            col_key = f"conf_{r.lower().replace(' ', '_').replace('-', '_')}"
            signals[col_key] = regime_conf[r]

        # Backward-compat aliases
        signals["conf_low_vol"]  = signals.get("conf_mean_reversion", pd.Series(np.nan, index=prices.index))
        signals["conf_high_vol"] = signals.get("conf_high_volatility", pd.Series(np.nan, index=prices.index))
        signals["conf_trending"] = signals.get("conf_risk_on",         pd.Series(np.nan, index=prices.index))
        signals["conf_mean_rev"] = signals.get("conf_mean_reversion",  pd.Series(np.nan, index=prices.index))

        if vix is not None:
            signals["vix"] = vix

        return regime, signals

    # ── Confidence summary ────────────────────────────────────────────────────

    def get_current_confidence(self, signals: pd.DataFrame) -> Dict[str, float]:
        """Return latest per-regime confidence scores as a dict."""
        last = signals.iloc[-1]
        result = {}
        for r in _NAMED_REGIMES:
            col = f"conf_{r.lower().replace(' ', '_').replace('-', '_')}"
            result[r] = float(last.get(col, 0) or 0)
        result["best_confidence"] = float(last.get("confidence", 0) or 0)
        result["macro_available"] = bool(last.get("macro_available", False))
        return result

    def get_regime_stats(self, regime: pd.Series) -> pd.DataFrame:
        stats_df = regime.value_counts().to_frame("count")
        stats_df["percentage"] = (stats_df["count"] / len(regime) * 100).round(2)
        return stats_df

    # ── Regime descriptions ───────────────────────────────────────────────────

    def get_regime_description(self, regime: str) -> dict:
        return _REGIME_DESCRIPTIONS.get(regime, _REGIME_DESCRIPTIONS["Unknown"])


# ── Regime playbook ───────────────────────────────────────────────────────────

_REGIME_DESCRIPTIONS: Dict[str, dict] = {
    "Risk-On": {
        "description":           "Favorable conditions — low volatility, uptrend, healthy macro",
        "characteristics":       "Low realized vol · upward momentum · steep yield curve · tight credit spreads",
        "macro_context":         "Yield curve positive (10Y > 2Y), HY spreads compressed, real rates moderate",
        "portfolio_implications":"Maximum equity exposure; growth and momentum tilt",
        "strategy":              "Buy breakouts, add cyclicals and growth; keep cash light",
        "risk_level":            "Low",
        "recommended_exposure":  "85–100%",
        "options_bias":          "Sell puts / buy calls; implied vol likely elevated vs realized",
    },
    "Caution": {
        "description":           "Deteriorating conditions — maintain core but reduce risk",
        "characteristics":       "Rising vol, flattening yield curve, credit spreads beginning to widen",
        "macro_context":         "Curve flattening (<50 bps), HY spreads at moderate levels, Fed tightening",
        "portfolio_implications":"Reduce equity weight; rotate toward quality, low-beta, dividends",
        "strategy":              "Trim momentum names, add defensives (XLP, XLU, XLV), raise cash buffer",
        "risk_level":            "Medium",
        "recommended_exposure":  "65–80%",
        "options_bias":          "Protective puts on concentrated positions; covered calls for income",
    },
    "High Volatility": {
        "description":           "Acute stress — volatility spike, credit markets seizing",
        "characteristics":       "Realized vol in top quartile, HY spreads sharply wider, flight to quality",
        "macro_context":         "Credit spreads >500 bps, VIX >25, forced de-risking across assets",
        "portfolio_implications":"Sharply reduce equity exposure; prioritize capital preservation",
        "strategy":              "Raise cash, buy Treasuries; wait for vol to peak before re-entry",
        "risk_level":            "High",
        "recommended_exposure":  "30–55%",
        "options_bias":          "Buy puts for protection; avoid selling premium into expanding vol",
    },
    "Stagflation": {
        "description":           "Inflation above trend, growth slowing — historically worst for equities",
        "characteristics":       "CPI above 4% YoY, real rates negative, curve flat or inverted, weak trend",
        "macro_context":         "Fed behind the curve or unable to hike; commodity prices elevated",
        "portfolio_implications":"Real assets, TIPS, commodities; short duration bonds",
        "strategy":              "Overweight energy, materials, gold; avoid long-duration and growth",
        "risk_level":            "High",
        "recommended_exposure":  "40–60%",
        "options_bias":          "Inflation-linked trades; consider collar strategies on equity book",
    },
    "Recession": {
        "description":           "Macro contraction — inverted curve, wide spreads, downtrend",
        "characteristics":       "Inverted 10Y-2Y, HY OAS >600 bps, unemployment rising, equities in downtrend",
        "macro_context":         "Yield curve deeply inverted >6 months, credit markets stressed, GDP contracting",
        "portfolio_implications":"Maximum defensive: Treasuries, gold, healthcare, staples; minimal equities",
        "strategy":              "Preserve capital; look for cycle-bottom signals (curve re-steepening) to re-enter",
        "risk_level":            "Very High",
        "recommended_exposure":  "20–40%",
        "options_bias":          "Long puts, protective structures; avoid naked equity risk",
    },
    "Mean Reversion": {
        "description":           "Range-bound — no clear trend, moderate volatility",
        "characteristics":       "Flat MA slope, vol near historical median, neutral credit spreads",
        "macro_context":         "Macro signals mixed; economy in transition or pause",
        "portfolio_implications":"Balanced exposure; contrarian opportunities in oversold names",
        "strategy":              "Range trading; sell strength / buy weakness; patience",
        "risk_level":            "Medium",
        "recommended_exposure":  "60–75%",
        "options_bias":          "Iron condors, short straddles — sell vol into range",
    },
    "Uncertain": {
        "description":           "Ambiguous conditions — no regime clears confidence threshold",
        "characteristics":       "Mixed signals; market in transition between regimes",
        "macro_context":         "Signals conflicting; high model uncertainty",
        "portfolio_implications":"Hold diversified core; avoid large directional bets",
        "strategy":              "Reduce position sizes; wait for signals to converge",
        "risk_level":            "Elevated (unclear)",
        "recommended_exposure":  "50–70%",
        "options_bias":          "Reduce delta; favour defined-risk structures",
    },
    "Unknown": {
        "description":           "Insufficient data",
        "characteristics":       "Not enough history to classify",
        "macro_context":         "N/A",
        "portfolio_implications":"Maintain current positioning",
        "strategy":              "Wait for more data",
        "risk_level":            "Unknown",
        "recommended_exposure":  "Maintain current",
        "options_bias":          "N/A",
    },
}
