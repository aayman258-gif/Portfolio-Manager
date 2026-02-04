"""
Unified Scoring Engine
Combines quantitative signals and fundamental metrics into single score
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Tuple
from datetime import datetime, timedelta


class ScoringEngine:
    """Score positions on quant + fundamental metrics"""

    def __init__(self):
        self.regime_weights = {
            'Low Vol': {'quant': 0.3, 'fundamental': 0.7},      # Value regime
            'High Vol': {'quant': 0.4, 'fundamental': 0.6},     # Quality focus
            'Trending': {'quant': 0.7, 'fundamental': 0.3},     # Momentum regime
            'Mean Reversion': {'quant': 0.5, 'fundamental': 0.5}  # Balanced
        }

    def calculate_momentum_score(self, prices: pd.Series, lookback: int = 60) -> float:
        """
        Calculate momentum score based on price trends
        Returns score 0-100
        """
        if len(prices) < lookback:
            return 50.0  # Neutral if insufficient data

        # Multiple timeframe momentum
        ret_1m = (prices.iloc[-1] / prices.iloc[-22] - 1) * 100 if len(prices) >= 22 else 0
        ret_3m = (prices.iloc[-1] / prices.iloc[-66] - 1) * 100 if len(prices) >= 66 else 0
        ret_6m = (prices.iloc[-1] / prices.iloc[-126] - 1) * 100 if len(prices) >= 126 else 0

        # Normalize to 0-100 scale
        # Positive momentum = high score
        momentum_raw = (ret_1m * 0.5 + ret_3m * 0.3 + ret_6m * 0.2)

        # Convert to 0-100 scale (assume -50% to +50% range)
        score = np.clip((momentum_raw + 50) / 100 * 100, 0, 100)

        return score

    def calculate_volatility_score(self, prices: pd.Series, lookback: int = 60) -> float:
        """
        Calculate volatility score (lower vol = higher score)
        Returns score 0-100
        """
        if len(prices) < lookback:
            return 50.0

        returns = prices.pct_change().dropna()
        volatility = returns.tail(lookback).std() * np.sqrt(252) * 100  # Annualized %

        # Lower volatility = higher score
        # Assume 0-80% volatility range
        score = np.clip(100 - (volatility / 80 * 100), 0, 100)

        return score

    def calculate_regime_fit_score(
        self,
        prices: pd.Series,
        current_regime: str
    ) -> float:
        """
        Calculate how well stock fits current regime
        Returns score 0-100
        """
        if len(prices) < 60:
            return 50.0

        returns = prices.pct_change().dropna()
        volatility = returns.tail(60).std() * np.sqrt(252) * 100

        # Calculate trend strength
        ma_20 = prices.rolling(20).mean()
        ma_50 = prices.rolling(50).mean()
        trend_strength = ((prices.iloc[-1] - ma_50.iloc[-1]) / ma_50.iloc[-1] * 100) if not pd.isna(ma_50.iloc[-1]) else 0

        score = 50.0  # Default neutral

        if current_regime == 'Low Vol':
            # Favor low volatility, stable stocks
            score = 100 - (volatility / 40 * 100)  # Lower vol = higher score
            score = np.clip(score, 0, 100)

        elif current_regime == 'High Vol':
            # Favor defensive, stable stocks in high vol
            score = 100 - (volatility / 40 * 100)
            score = np.clip(score, 0, 100)

        elif current_regime == 'Trending':
            # Favor momentum and trend alignment
            if trend_strength > 0:
                score = 50 + np.clip(trend_strength * 2, 0, 50)
            else:
                score = 50 + np.clip(trend_strength * 2, -50, 0)

        elif current_regime == 'Mean Reversion':
            # Favor oversold quality (negative momentum but low vol)
            if trend_strength < 0:
                score = 60 - (volatility / 60 * 40)  # Oversold + low vol = high score
            else:
                score = 40

        return np.clip(score, 0, 100)

    def fetch_fundamental_data(self, ticker: str) -> Dict:
        """
        Fetch fundamental metrics using yfinance
        Returns dictionary with key metrics
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            fundamentals = {
                # Valuation
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),

                # Profitability
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),

                # Growth
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),

                # Financial Health
                'debt_to_equity': info.get('debtToEquity', None),
                'current_ratio': info.get('currentRatio', None),
                'quick_ratio': info.get('quickRatio', None),

                # Dividend
                'dividend_yield': info.get('dividendYield', None),
                'payout_ratio': info.get('payoutRatio', None),

                # Other
                'beta': info.get('beta', None),
                'market_cap': info.get('marketCap', None)
            }

            return fundamentals

        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")
            return {}

    def calculate_growth_score(self, fundamentals: Dict) -> float:
        """
        Calculate growth score from fundamentals
        Returns score 0-100
        """
        rev_growth = fundamentals.get('revenue_growth', None)
        earnings_growth = fundamentals.get('earnings_growth', None)

        if rev_growth is None and earnings_growth is None:
            return 50.0  # Neutral if no data

        # Convert to percentages
        rev_growth_pct = (rev_growth * 100) if rev_growth is not None else 0
        earn_growth_pct = (earnings_growth * 100) if earnings_growth is not None else 0

        # Average growth
        avg_growth = (rev_growth_pct + earn_growth_pct) / 2 if rev_growth and earnings_growth else (rev_growth_pct or earn_growth_pct)

        # Normalize to 0-100 (assume -20% to +40% range)
        score = np.clip((avg_growth + 20) / 60 * 100, 0, 100)

        return score

    def calculate_quality_score(self, fundamentals: Dict) -> float:
        """
        Calculate quality score (margins, ROE, financial health)
        Returns score 0-100
        """
        scores = []

        # Profit margins
        profit_margin = fundamentals.get('profit_margin', None)
        if profit_margin is not None:
            margin_score = np.clip(profit_margin * 100 / 30 * 100, 0, 100)  # 30% = perfect
            scores.append(margin_score)

        # ROE
        roe = fundamentals.get('roe', None)
        if roe is not None:
            roe_score = np.clip(roe * 100 / 25 * 100, 0, 100)  # 25% ROE = perfect
            scores.append(roe_score)

        # Financial health (debt to equity)
        debt_to_eq = fundamentals.get('debt_to_equity', None)
        if debt_to_eq is not None:
            # Lower debt = higher score
            debt_score = np.clip(100 - (debt_to_eq / 100 * 100), 0, 100)
            scores.append(debt_score)

        # Current ratio
        current_ratio = fundamentals.get('current_ratio', None)
        if current_ratio is not None:
            # Ideal around 2.0
            ratio_score = np.clip((min(current_ratio, 3) / 3) * 100, 0, 100)
            scores.append(ratio_score)

        if not scores:
            return 50.0

        return np.mean(scores)

    def calculate_valuation_score(self, fundamentals: Dict) -> float:
        """
        Calculate valuation score (lower P/E, P/B = higher score)
        Returns score 0-100
        """
        scores = []

        # P/E ratio
        pe = fundamentals.get('pe_ratio', None)
        if pe is not None and pe > 0:
            # Lower P/E = higher score (assume 0-40 range)
            pe_score = np.clip(100 - (pe / 40 * 100), 0, 100)
            scores.append(pe_score)

        # PEG ratio
        peg = fundamentals.get('peg_ratio', None)
        if peg is not None and peg > 0:
            # PEG around 1.0 is ideal, <1 is undervalued
            peg_score = np.clip(100 - (abs(peg - 1.0) / 2 * 100), 0, 100)
            scores.append(peg_score)

        # Price to Book
        pb = fundamentals.get('price_to_book', None)
        if pb is not None and pb > 0:
            # Lower P/B = higher score (assume 0-5 range)
            pb_score = np.clip(100 - (pb / 5 * 100), 0, 100)
            scores.append(pb_score)

        if not scores:
            return 50.0

        return np.mean(scores)

    def calculate_unified_score(
        self,
        ticker: str,
        prices: pd.Series,
        current_regime: str
    ) -> Dict:
        """
        Calculate unified score combining quant + fundamental
        Weighted by current regime

        Returns:
            Dictionary with all scores and final unified score
        """
        # Quantitative scores
        momentum_score = self.calculate_momentum_score(prices)
        volatility_score = self.calculate_volatility_score(prices)
        regime_fit_score = self.calculate_regime_fit_score(prices, current_regime)

        quant_score = (momentum_score * 0.4 + volatility_score * 0.3 + regime_fit_score * 0.3)

        # Fundamental scores
        fundamentals = self.fetch_fundamental_data(ticker)
        growth_score = self.calculate_growth_score(fundamentals)
        quality_score = self.calculate_quality_score(fundamentals)
        valuation_score = self.calculate_valuation_score(fundamentals)

        fundamental_score = (growth_score * 0.35 + quality_score * 0.35 + valuation_score * 0.30)

        # Combine based on regime
        weights = self.regime_weights.get(current_regime, {'quant': 0.5, 'fundamental': 0.5})
        unified_score = (quant_score * weights['quant'] + fundamental_score * weights['fundamental'])

        result = {
            # Quant scores
            'momentum_score': momentum_score,
            'volatility_score': volatility_score,
            'regime_fit_score': regime_fit_score,
            'quant_score': quant_score,

            # Fundamental scores
            'growth_score': growth_score,
            'quality_score': quality_score,
            'valuation_score': valuation_score,
            'fundamental_score': fundamental_score,

            # Combined
            'unified_score': unified_score,

            # Weights used
            'regime': current_regime,
            'quant_weight': weights['quant'],
            'fundamental_weight': weights['fundamental'],

            # Raw fundamentals for display
            'fundamentals': fundamentals
        }

        return result

    def get_score_interpretation(self, score: float) -> Tuple[str, str]:
        """
        Interpret score and provide rating + color

        Returns:
            (rating, color)
        """
        if score >= 80:
            return "Strong Buy", "green"
        elif score >= 65:
            return "Buy", "lightgreen"
        elif score >= 50:
            return "Hold", "yellow"
        elif score >= 35:
            return "Sell", "orange"
        else:
            return "Strong Sell", "red"
