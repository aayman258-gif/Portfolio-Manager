"""
Regime Detection Engine
Adapted from regime-trading-system for portfolio management
Classifies market state using VIX, realized volatility, and entropy metrics
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple


class RegimeDetector:
    """Detects market regimes using multiple signals"""

    def __init__(self, lookback_vol: int = 20, lookback_trend: int = 50):
        self.lookback_vol = lookback_vol
        self.lookback_trend = lookback_trend

    def calculate_realized_volatility(self, prices: pd.Series) -> pd.Series:
        """Calculate rolling realized volatility (annualized)"""
        returns = prices.pct_change()
        realized_vol = returns.rolling(window=self.lookback_vol).std() * np.sqrt(252)
        return realized_vol

    def calculate_entropy(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Shannon entropy of return distribution
        Higher entropy = more uncertainty/randomness in the market
        """
        def window_entropy(x):
            if len(x) < 5:
                return np.nan
            hist, _ = np.histogram(x, bins=10, density=True)
            hist = hist[hist > 0]
            return stats.entropy(hist)

        entropy = returns.rolling(window=window).apply(window_entropy, raw=True)
        return entropy

    def detect_trend(self, prices: pd.Series) -> pd.Series:
        """
        Detect trending regime using moving average slope
        Returns: 1 (uptrend), -1 (downtrend), 0 (sideways)
        """
        ma = prices.rolling(window=self.lookback_trend).mean()
        ma_slope = ma.diff(10) / ma

        trend = pd.Series(0, index=prices.index)
        trend[ma_slope > 0.02] = 1
        trend[ma_slope < -0.02] = -1

        return trend

    def classify_regime(
        self,
        prices: pd.Series,
        vix: pd.Series = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Classify market regime into:
        - Low Vol: Low volatility, range-bound
        - High Vol: High volatility, expansion
        - Trending: Strong directional movement
        - Mean Reversion: High vol but ranging

        Returns:
            regime: Series with regime labels
            signals: DataFrame with all signals for analysis
        """
        realized_vol = self.calculate_realized_volatility(prices)
        returns = prices.pct_change()
        entropy = self.calculate_entropy(returns)
        trend = self.detect_trend(prices)

        # Volatility percentile
        vol_percentile = realized_vol.rolling(window=252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 0 else np.nan,
            raw=False
        )

        regime = pd.Series('Unknown', index=prices.index)

        for i in range(len(prices)):
            if pd.isna(vol_percentile.iloc[i]) or pd.isna(entropy.iloc[i]):
                continue

            vol_pct = vol_percentile.iloc[i]
            ent = entropy.iloc[i]
            tr = trend.iloc[i]

            # Classification logic
            if vol_pct < 0.3 and abs(tr) == 0:
                regime.iloc[i] = 'Low Vol'
            elif vol_pct > 0.7 and ent > entropy.median():
                regime.iloc[i] = 'High Vol'
            elif abs(tr) == 1 and vol_pct < 0.6:
                regime.iloc[i] = 'Trending'
            elif vol_pct > 0.5 and abs(tr) == 0:
                regime.iloc[i] = 'Mean Reversion'
            else:
                if i > 0 and regime.iloc[i-1] != 'Unknown':
                    regime.iloc[i] = regime.iloc[i-1]

        signals = pd.DataFrame({
            'price': prices,
            'realized_vol': realized_vol,
            'vol_percentile': vol_percentile,
            'entropy': entropy,
            'trend': trend,
            'regime': regime
        })

        if vix is not None:
            signals['vix'] = vix

        return regime, signals

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
                'recommended_exposure': 'High (80-100%)'
            },
            'High Vol': {
                'description': 'High volatility expansion environment',
                'characteristics': 'Elevated uncertainty, big price swings, fear in market',
                'portfolio_implications': 'Reduce equity exposure, increase cash/bonds, focus on quality',
                'strategy': 'Defensive positioning, preserve capital, wait for opportunities',
                'risk_level': 'High',
                'recommended_exposure': 'Low (40-60%)'
            },
            'Trending': {
                'description': 'Strong directional market momentum',
                'characteristics': 'Clear trend (up or down), momentum prevails',
                'portfolio_implications': 'Ride momentum, sector rotation, growth tilt if uptrend',
                'strategy': 'Follow the trend, momentum stocks, avoid fighting direction',
                'risk_level': 'Medium',
                'recommended_exposure': 'Medium-High (70-90%)'
            },
            'Mean Reversion': {
                'description': 'Range-bound with elevated volatility',
                'characteristics': 'No clear trend, volatility elevated, choppy action',
                'portfolio_implications': 'Contrarian plays, buy oversold quality names',
                'strategy': 'Range trading, sell strength/buy weakness, patience',
                'risk_level': 'Medium',
                'recommended_exposure': 'Medium (60-80%)'
            },
            'Unknown': {
                'description': 'Insufficient data or transitional period',
                'characteristics': 'Not enough data to classify regime',
                'portfolio_implications': 'Maintain current positioning',
                'strategy': 'Wait for regime clarity',
                'risk_level': 'Unknown',
                'recommended_exposure': 'Maintain current'
            }
        }

        return descriptions.get(regime, descriptions['Unknown'])
