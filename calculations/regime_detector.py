"""
Regime Detection Engine
Classifies market state using VIX, realized volatility, and entropy metrics.

Regimes: Low Vol · High Vol · Trending · Mean Reversion · Uncertain
Each bar carries per-regime confidence scores; 'Uncertain' fires when no
regime clears the UNCERTAIN_THRESHOLD, indicating ambiguous / transitional
market conditions.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict


# Minimum confidence required to assign a named regime.
# Below this threshold the bar is labelled "Uncertain".
UNCERTAIN_THRESHOLD = 0.35

_NAMED_REGIMES = ['Low Vol', 'High Vol', 'Trending', 'Mean Reversion']


class RegimeDetector:
    """Detects market regimes using multiple signals"""

    def __init__(self, lookback_vol: int = 20, lookback_trend: int = 50):
        self.lookback_vol = lookback_vol
        self.lookback_trend = lookback_trend

    # ── Raw signal calculators ────────────────────────────────────────────────

    def calculate_realized_volatility(self, prices: pd.Series) -> pd.Series:
        """Calculate rolling realized volatility (annualized)"""
        returns = prices.pct_change()
        realized_vol = returns.rolling(window=self.lookback_vol).std() * np.sqrt(252)
        return realized_vol

    def calculate_entropy(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Shannon entropy of return distribution.
        Higher entropy = more uncertainty / randomness in the market.
        """
        def window_entropy(x):
            if len(x) < 5:
                return np.nan
            hist, _ = np.histogram(x, bins=10, density=True)
            hist = hist[hist > 0]
            return stats.entropy(hist)

        return returns.rolling(window=window).apply(window_entropy, raw=True)

    def detect_trend(self, prices: pd.Series) -> pd.Series:
        """
        Detect trending regime using moving average slope.
        Returns: 1 (uptrend), -1 (downtrend), 0 (sideways)
        """
        ma = prices.rolling(window=self.lookback_trend).mean()
        ma_slope = ma.diff(10) / ma

        trend = pd.Series(0, index=prices.index)
        trend[ma_slope > 0.02] = 1
        trend[ma_slope < -0.02] = -1
        return trend

    # ── Confidence scoring ────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        vol_pct: float,
        ent: float,
        ent_median: float,
        tr: float,
    ) -> Dict[str, float]:
        """
        Return a confidence score in [0, 1] for each named regime.

        Scores are derived from how closely the current signal values
        match each regime's defining conditions:

          Low Vol       — low vol percentile + no trend
          High Vol      — high vol percentile + above-median entropy
          Trending      — strong trend signal + moderate vol
          Mean Reversion— mid-high vol + no trend
        """
        # ── Low Vol ──────────────────────────────────────────────────────────
        lv_vol   = max(0.0, (0.30 - vol_pct) / 0.30)   # 1 → 0 as vol_pct: 0 → 0.30
        lv_trend = 1.0 if tr == 0 else 0.20             # penalise trending
        lv = lv_vol * lv_trend

        # ── High Vol ─────────────────────────────────────────────────────────
        hv_vol = max(0.0, (vol_pct - 0.70) / 0.30)     # 0 → 1 as vol_pct: 0.70 → 1.0
        ent_bonus = max(0.0, (ent - ent_median) / (ent_median + 1e-9))
        hv = min(1.0, hv_vol * (1.0 + 0.5 * ent_bonus))

        # ── Trending ─────────────────────────────────────────────────────────
        tr_strength = abs(tr)                           # 0 or 1
        tr_vol_ok   = max(0.0, (0.60 - vol_pct) / 0.60)  # prefer moderate vol
        trending    = tr_strength * tr_vol_ok

        # ── Mean Reversion ───────────────────────────────────────────────────
        mr_vol   = max(0.0, (vol_pct - 0.50) / 0.50)   # 0 → 1 as vol_pct: 0.50 → 1.0
        mr_trend = 1.0 if tr == 0 else 0.15             # penalise trending
        mr = mr_vol * mr_trend

        return {
            'Low Vol':        round(lv,       4),
            'High Vol':       round(hv,       4),
            'Trending':       round(trending, 4),
            'Mean Reversion': round(mr,       4),
        }

    # ── Main classifier ───────────────────────────────────────────────────────

    def classify_regime(
        self,
        prices: pd.Series,
        vix: pd.Series = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Classify each bar into one of five states:
            Low Vol | High Vol | Trending | Mean Reversion | Uncertain

        'Uncertain' is assigned whenever no named regime achieves a
        confidence score ≥ UNCERTAIN_THRESHOLD (0.35).  This makes
        transitional / ambiguous market conditions an explicit, first-
        class regime rather than silently inheriting the previous label.

        Returns
        -------
        regime   : pd.Series  — regime label per bar
        signals  : pd.DataFrame — all signals + confidence columns
        """
        realized_vol  = self.calculate_realized_volatility(prices)
        returns       = prices.pct_change()
        entropy       = self.calculate_entropy(returns)
        trend         = self.detect_trend(prices)

        # Volatility percentile (rolling 1-year window)
        vol_percentile = realized_vol.rolling(window=252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
            if len(x) > 0 else np.nan,
            raw=False,
        )

        ent_median = float(entropy.median())   # precompute once

        regime     = pd.Series('Uncertain', index=prices.index)
        confidence = pd.Series(0.0,         index=prices.index)

        # Per-regime confidence columns
        conf_lv = pd.Series(np.nan, index=prices.index)
        conf_hv = pd.Series(np.nan, index=prices.index)
        conf_tr = pd.Series(np.nan, index=prices.index)
        conf_mr = pd.Series(np.nan, index=prices.index)

        for i in range(len(prices)):
            if pd.isna(vol_percentile.iloc[i]) or pd.isna(entropy.iloc[i]):
                # Not enough history — stay 'Uncertain'
                continue

            scores = self._compute_confidence(
                vol_pct    = float(vol_percentile.iloc[i]),
                ent        = float(entropy.iloc[i]),
                ent_median = ent_median,
                tr         = float(trend.iloc[i]),
            )

            conf_lv.iloc[i] = scores['Low Vol']
            conf_hv.iloc[i] = scores['High Vol']
            conf_tr.iloc[i] = scores['Trending']
            conf_mr.iloc[i] = scores['Mean Reversion']

            best_regime = max(scores, key=scores.get)
            best_conf   = scores[best_regime]

            if best_conf >= UNCERTAIN_THRESHOLD:
                regime.iloc[i]     = best_regime
                confidence.iloc[i] = best_conf
            else:
                regime.iloc[i]     = 'Uncertain'
                confidence.iloc[i] = best_conf   # record even low confidence

        signals = pd.DataFrame({
            'price':         prices,
            'realized_vol':  realized_vol,
            'vol_percentile': vol_percentile,
            'entropy':       entropy,
            'trend':         trend,
            'regime':        regime,
            'confidence':    confidence,
            'conf_low_vol':  conf_lv,
            'conf_high_vol': conf_hv,
            'conf_trending': conf_tr,
            'conf_mean_rev': conf_mr,
        })

        if vix is not None:
            signals['vix'] = vix

        return regime, signals

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_current_confidence(self, signals: pd.DataFrame) -> Dict[str, float]:
        """
        Return the latest per-regime confidence scores as a dict.
        Useful for displaying a confidence breakdown in the UI.
        """
        last = signals.iloc[-1]
        return {
            'Low Vol':        float(last.get('conf_low_vol',  0) or 0),
            'High Vol':       float(last.get('conf_high_vol', 0) or 0),
            'Trending':       float(last.get('conf_trending', 0) or 0),
            'Mean Reversion': float(last.get('conf_mean_rev', 0) or 0),
            'best_confidence': float(last.get('confidence',  0) or 0),
        }

    def get_regime_stats(self, regime: pd.Series) -> pd.DataFrame:
        """Get statistics about regime distribution"""
        stats_df = regime.value_counts().to_frame('count')
        stats_df['percentage'] = (stats_df['count'] / len(regime) * 100).round(2)
        return stats_df

    def get_regime_description(self, regime: str) -> dict:
        """Get description and implications for each regime"""
        descriptions = {
            'Low Vol': {
                'description': 'Low volatility, range-bound market',
                'characteristics': 'Calm markets, low uncertainty, stable prices',
                'portfolio_implications': 'Favor quality value stocks, dividend growth, defensive sectors',
                'strategy': 'Income generation, buy quality on dips',
                'risk_level': 'Low',
                'recommended_exposure': 'High (80–100%)',
            },
            'High Vol': {
                'description': 'High volatility expansion environment',
                'characteristics': 'Elevated uncertainty, big price swings, fear in market',
                'portfolio_implications': 'Reduce equity exposure, increase cash/bonds, focus on quality',
                'strategy': 'Defensive positioning, preserve capital, wait for opportunities',
                'risk_level': 'High',
                'recommended_exposure': 'Low (40–60%)',
            },
            'Trending': {
                'description': 'Strong directional market momentum',
                'characteristics': 'Clear trend (up or down), momentum prevails',
                'portfolio_implications': 'Ride momentum, sector rotation, growth tilt if uptrend',
                'strategy': 'Follow the trend, momentum stocks, avoid fighting direction',
                'risk_level': 'Medium',
                'recommended_exposure': 'Medium-High (70–90%)',
            },
            'Mean Reversion': {
                'description': 'Range-bound with elevated volatility',
                'characteristics': 'No clear trend, volatility elevated, choppy action',
                'portfolio_implications': 'Contrarian plays, buy oversold quality names',
                'strategy': 'Range trading, sell strength / buy weakness, patience',
                'risk_level': 'Medium',
                'recommended_exposure': 'Medium (60–80%)',
            },
            'Uncertain': {
                'description': 'Ambiguous / transitional market conditions',
                'characteristics': (
                    'No single regime dominates the signal mix. '
                    'Conditions are inconsistent or in flux.'
                ),
                'portfolio_implications': (
                    'Avoid large directional bets. '
                    'Reduce position sizes until regime clarity returns.'
                ),
                'strategy': (
                    'Hold diversified core; defer new risk until signals converge. '
                    'Watch vol percentile and trend for resolution.'
                ),
                'risk_level': 'Elevated (unclear)',
                'recommended_exposure': 'Conservative (50–70%)',
            },
            'Unknown': {
                'description': 'Insufficient data',
                'characteristics': 'Not enough history to classify regime',
                'portfolio_implications': 'Maintain current positioning',
                'strategy': 'Wait for more data',
                'risk_level': 'Unknown',
                'recommended_exposure': 'Maintain current',
            },
        }

        return descriptions.get(regime, descriptions['Unknown'])
