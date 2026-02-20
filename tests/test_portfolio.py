"""Unit tests for portfolio manager calculations"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from calculations.options_analytics import OptionsAnalytics
from calculations.regime_detector import RegimeDetector
from calculations.optimizer import RegimeAwareOptimizer


class TestBlackScholesPricing(unittest.TestCase):
    """Tests for Black-Scholes / Merton option pricing"""

    def setUp(self):
        self.calc = OptionsAnalytics()

    def test_atm_call_price(self):
        """ATM call ≈ 10.45 (S=K=100, T=1, σ=0.2, r=0.05, q=0)"""
        price = self.calc.black_scholes(100, 100, 1, 0.2, 'call', 0.05)
        self.assertAlmostEqual(price, 10.45, delta=0.02)

    def test_atm_put_price(self):
        """ATM put ≈ 5.57 (S=K=100, T=1, σ=0.2, r=0.05, q=0)"""
        price = self.calc.black_scholes(100, 100, 1, 0.2, 'put', 0.05)
        self.assertAlmostEqual(price, 5.57, delta=0.02)

    def test_deep_itm_call_lower_bound(self):
        """Deep ITM call >= intrinsic lower bound S - K*e^{-rT}"""
        S, K, T, r = 150, 100, 1, 0.05
        price = self.calc.black_scholes(S, K, T, 0.2, 'call', r)
        lower_bound = S - K * np.exp(-r * T)
        self.assertGreaterEqual(price, lower_bound - 1e-10)

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call < 0.01"""
        price = self.calc.black_scholes(50, 200, 0.1, 0.2, 'call', 0.05)
        self.assertLess(price, 0.01)

    def test_at_expiry_call_intrinsic(self):
        """At T=0: call returns exact intrinsic value"""
        self.assertEqual(self.calc.black_scholes(110, 100, 0, 0.2, 'call', 0.05), 10.0)
        self.assertEqual(self.calc.black_scholes(90, 100, 0, 0.2, 'call', 0.05), 0.0)

    def test_at_expiry_put_intrinsic(self):
        """At T=0: put returns exact intrinsic value"""
        self.assertEqual(self.calc.black_scholes(90, 100, 0, 0.2, 'put', 0.05), 10.0)
        self.assertEqual(self.calc.black_scholes(110, 100, 0, 0.2, 'put', 0.05), 0.0)

    def test_put_call_parity(self):
        """C - P = S*e^{-qT} - K*e^{-rT} (put-call parity, generalized Merton)"""
        param_combos = [
            (100, 100, 1.0, 0.2, 0.05, 0.0),
            (110, 100, 0.5, 0.3, 0.04, 0.0),
            (90,  100, 2.0, 0.15, 0.03, 0.0),
            (150, 120, 0.25, 0.4, 0.06, 0.0),
        ]
        for S, K, T, sigma, r, q in param_combos:
            call = self.calc.black_scholes(S, K, T, sigma, 'call', r, q)
            put = self.calc.black_scholes(S, K, T, sigma, 'put', r, q)
            parity = S * np.exp(-q * T) - K * np.exp(-r * T)
            self.assertAlmostEqual(
                call - put, parity, places=8,
                msg=f"Put-call parity failed for S={S}, K={K}, T={T}"
            )

    def test_call_monotone_in_spot(self):
        """Call price monotonically increasing in S"""
        prices = [
            self.calc.black_scholes(S, 100, 1, 0.2, 'call', 0.05)
            for S in range(80, 130, 5)
        ]
        for i in range(len(prices) - 1):
            self.assertLess(prices[i], prices[i + 1])

    def test_put_monotone_decreasing_in_spot(self):
        """Put price monotonically decreasing in S"""
        prices = [
            self.calc.black_scholes(S, 100, 1, 0.2, 'put', 0.05)
            for S in range(80, 130, 5)
        ]
        for i in range(len(prices) - 1):
            self.assertGreater(prices[i], prices[i + 1])

    def test_all_prices_non_negative(self):
        """All option prices >= 0 across S/K/T grid"""
        for S in [50, 100, 150]:
            for K in [50, 100, 150]:
                for T in [0.1, 0.5, 1.0]:
                    call = self.calc.black_scholes(S, K, T, 0.2, 'call', 0.05)
                    put = self.calc.black_scholes(S, K, T, 0.2, 'put', 0.05)
                    self.assertGreaterEqual(call, 0.0)
                    self.assertGreaterEqual(put, 0.0)


class TestGreeks(unittest.TestCase):
    """Tests for Black-Scholes / Merton Greeks"""

    def setUp(self):
        self.calc = OptionsAnalytics()

    def test_call_delta_range(self):
        """Call delta in (0, 1)"""
        g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'call', 0.05)
        self.assertGreater(g['delta'], 0)
        self.assertLess(g['delta'], 1)

    def test_put_delta_range(self):
        """Put delta in (-1, 0)"""
        g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'put', 0.05)
        self.assertLess(g['delta'], 0)
        self.assertGreater(g['delta'], -1)

    def test_deep_itm_call_delta(self):
        """Deep ITM call delta > 0.95"""
        g = self.calc.calculate_greeks(200, 100, 1, 0.2, 'call', 0.05)
        self.assertGreater(g['delta'], 0.95)

    def test_deep_otm_call_delta(self):
        """Deep OTM call delta < 0.05"""
        g = self.calc.calculate_greeks(50, 200, 1, 0.2, 'call', 0.05)
        self.assertLess(g['delta'], 0.05)

    def test_put_call_delta_parity(self):
        """call_delta - put_delta = 1 (q=0 case)"""
        call_g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'call', 0.05)
        put_g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'put', 0.05)
        self.assertAlmostEqual(call_g['delta'] - put_g['delta'], 1.0, places=8)

    def test_gamma_positive(self):
        """Gamma > 0 for both call and put"""
        call_g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'call', 0.05)
        put_g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'put', 0.05)
        self.assertGreater(call_g['gamma'], 0)
        self.assertGreater(put_g['gamma'], 0)

    def test_call_put_gamma_equal(self):
        """Call gamma = put gamma"""
        call_g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'call', 0.05)
        put_g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'put', 0.05)
        self.assertAlmostEqual(call_g['gamma'], put_g['gamma'], places=8)

    def test_gamma_peaks_atm(self):
        """Gamma peaks near ATM vs deep ITM and OTM"""
        atm_g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'call', 0.05)
        itm_g = self.calc.calculate_greeks(130, 100, 1, 0.2, 'call', 0.05)
        otm_g = self.calc.calculate_greeks(70,  100, 1, 0.2, 'call', 0.05)
        self.assertGreater(atm_g['gamma'], itm_g['gamma'])
        self.assertGreater(atm_g['gamma'], otm_g['gamma'])

    def test_theta_negative_atm(self):
        """ATM call and put theta < 0"""
        call_g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'call', 0.05)
        put_g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'put', 0.05)
        self.assertLess(call_g['theta'], 0)
        self.assertLess(put_g['theta'], 0)

    def test_vega_positive(self):
        """Vega > 0 for both call and put"""
        call_g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'call', 0.05)
        put_g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'put', 0.05)
        self.assertGreater(call_g['vega'], 0)
        self.assertGreater(put_g['vega'], 0)

    def test_call_put_vega_equal(self):
        """Call vega = put vega"""
        call_g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'call', 0.05)
        put_g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'put', 0.05)
        self.assertAlmostEqual(call_g['vega'], put_g['vega'], places=8)

    def test_call_rho_positive(self):
        """Call rho > 0"""
        g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'call', 0.05)
        self.assertGreater(g['rho'], 0)

    def test_put_rho_negative(self):
        """Put rho < 0"""
        g = self.calc.calculate_greeks(100, 100, 1, 0.2, 'put', 0.05)
        self.assertLess(g['rho'], 0)

    def test_at_expiry_greeks_zero(self):
        """At T=0: gamma, theta, vega, rho all equal 0"""
        g = self.calc.calculate_greeks(100, 100, 0, 0.2, 'call', 0.05)
        self.assertEqual(g['gamma'], 0.0)
        self.assertEqual(g['theta'], 0.0)
        self.assertEqual(g['vega'], 0.0)
        self.assertEqual(g['rho'], 0.0)


class TestRegimeClassification(unittest.TestCase):
    """Tests for regime detection and classification"""

    def setUp(self):
        self.detector = RegimeDetector(lookback_vol=20, lookback_trend=50)
        np.random.seed(42)

    def _make_two_phase_series(self, n=300, vol1=0.30, drift1=0.0, vol2=0.01, drift2=0.0):
        """Build a two-phase price series: first half vol1, second half vol2."""
        n1 = n // 2
        n2 = n - n1
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        r1 = np.random.normal(drift1 / 252, vol1 / np.sqrt(252), n1)
        r2 = np.random.normal(drift2 / 252, vol2 / np.sqrt(252), n2)
        prices = 100 * np.exp(np.cumsum(np.concatenate([r1, r2])))
        return pd.Series(prices, index=dates)

    def _make_trending_series(self, n=400):
        """Generate a strongly trending price series.

        80% annual drift gives MA slope ≈ 0.032, well above the 0.02 threshold
        used by detect_trend (which checks 10-day change in 50-day MA / MA).
        """
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        returns = np.random.normal(0.80 / 252, 0.08 / np.sqrt(252), n)
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates)

    def test_low_vol_regime_detected(self):
        """Two-phase series (high → low vol) produces Low Vol as most common final regime"""
        # First half: 30% vol, second half: 1% vol
        prices = self._make_two_phase_series(vol1=0.30, vol2=0.01)
        regime, _ = self.detector.classify_regime(prices)
        final = regime[regime != 'Unknown'].iloc[-30:]
        if len(final) > 0:
            self.assertEqual(final.value_counts().index[0], 'Low Vol')

    def test_trending_regime_detected(self):
        """Strongly trending series produces Trending as most common final regime"""
        prices = self._make_trending_series()
        regime, _ = self.detector.classify_regime(prices)
        final = regime[regime != 'Unknown'].iloc[-30:]
        if len(final) > 0:
            self.assertEqual(final.value_counts().index[0], 'Trending')

    def test_high_vol_regime_detected(self):
        """Two-phase series (low → high vol) produces High Vol as most common final regime"""
        # First half: 10% vol, second half: 80% vol
        prices = self._make_two_phase_series(vol1=0.10, vol2=0.80)
        regime, _ = self.detector.classify_regime(prices)
        final = regime[regime != 'Unknown'].iloc[-30:]
        if len(final) > 0:
            self.assertEqual(final.value_counts().index[0], 'High Vol')

    def test_regime_series_length_matches_prices(self):
        """Regime series length equals price series length"""
        prices = self._make_trending_series()
        regime, _ = self.detector.classify_regime(prices)
        self.assertEqual(len(regime), len(prices))

    def test_signals_required_columns(self):
        """Signals DataFrame contains all required columns"""
        prices = self._make_trending_series()
        _, signals = self.detector.classify_regime(prices)
        required = {'price', 'realized_vol', 'vol_percentile', 'entropy', 'trend', 'regime'}
        self.assertTrue(required.issubset(set(signals.columns)))

    def test_vix_column_when_provided(self):
        """vix column present in signals when VIX series is passed"""
        prices = self._make_trending_series()
        vix = pd.Series(np.random.uniform(15, 25, len(prices)), index=prices.index)
        _, signals = self.detector.classify_regime(prices, vix)
        self.assertIn('vix', signals.columns)

    def test_all_regime_labels_valid(self):
        """All regime labels are members of the valid set"""
        valid = {'Low Vol', 'High Vol', 'Trending', 'Mean Reversion', 'Unknown'}
        prices = self._make_two_phase_series(vol1=0.10, vol2=0.80)
        regime, _ = self.detector.classify_regime(prices)
        self.assertTrue(set(regime.unique()).issubset(valid))

    def test_get_regime_description_required_keys(self):
        """get_regime_description returns all 6 required keys for every regime name"""
        required_keys = {
            'description', 'characteristics', 'portfolio_implications',
            'strategy', 'risk_level', 'recommended_exposure'
        }
        for name in ['Low Vol', 'High Vol', 'Trending', 'Mean Reversion', 'Unknown']:
            desc = self.detector.get_regime_description(name)
            self.assertTrue(
                required_keys.issubset(set(desc.keys())),
                msg=f"Missing keys in get_regime_description('{name}')"
            )


class TestPortfolioWeightConstraints(unittest.TestCase):
    """Tests for regime-based portfolio weight constraints"""

    def setUp(self):
        self.optimizer = RegimeAwareOptimizer()
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    def _make_synthetic_prices(self, n=500):
        """Create a synthetic price DataFrame to avoid live network calls."""
        np.random.seed(123)
        dates = pd.date_range('2022-01-01', periods=n, freq='B')
        data = {}
        for ticker in self.tickers:
            returns = np.random.normal(0.0005, 0.015, n)
            data[ticker] = 100 * np.exp(np.cumsum(returns))
        return pd.DataFrame(data, index=dates)

    def _run_optimize(self, regime):
        """Run optimize_portfolio with mocked price data (no live network)."""
        synthetic = self._make_synthetic_prices()
        with patch.object(self.optimizer, 'fetch_price_data', return_value=synthetic):
            return self.optimizer.optimize_portfolio(self.tickers, regime)

    def test_weights_sum_to_one_all_regimes(self):
        """Weights sum to 1.0 (tolerance 1e-6) for all 4 regimes"""
        for regime in ['Low Vol', 'High Vol', 'Trending', 'Mean Reversion']:
            result = self._run_optimize(regime)
            total = sum(result['weights'].values())
            self.assertAlmostEqual(
                total, 1.0, places=6,
                msg=f"Weights sum {total:.8f} != 1.0 for {regime}"
            )

    def test_no_equity_weight_above_30pct(self):
        """No individual equity weight > 0.30 for all 4 regimes"""
        for regime in ['Low Vol', 'High Vol', 'Trending', 'Mean Reversion']:
            result = self._run_optimize(regime)
            for ticker, w in result['weights'].items():
                if ticker != 'CASH':
                    self.assertLessEqual(
                        w, 0.30 + 1e-6,
                        msg=f"{ticker} weight {w:.4f} > 0.30 in {regime}"
                    )

    def test_all_weights_non_negative(self):
        """All weights >= 0 for all 4 regimes"""
        for regime in ['Low Vol', 'High Vol', 'Trending', 'Mean Reversion']:
            result = self._run_optimize(regime)
            for ticker, w in result['weights'].items():
                self.assertGreaterEqual(
                    w, 0.0,
                    msg=f"Negative weight {w:.6f} for {ticker} in {regime}"
                )

    def test_high_vol_cash_gte_40pct(self):
        """High Vol: CASH weight >= 0.40"""
        result = self._run_optimize('High Vol')
        cash = result['weights'].get('CASH', 0.0)
        self.assertGreaterEqual(cash, 0.40 - 1e-6)

    def test_trending_equity_total_lte_90pct(self):
        """Trending: total equity weight <= 0.90"""
        result = self._run_optimize('Trending')
        equity = sum(v for k, v in result['weights'].items() if k != 'CASH')
        self.assertLessEqual(equity, 0.90 + 1e-6)

    def test_mean_reversion_equity_total_lte_80pct(self):
        """Mean Reversion: total equity weight <= 0.80"""
        result = self._run_optimize('Mean Reversion')
        equity = sum(v for k, v in result['weights'].items() if k != 'CASH')
        self.assertLessEqual(equity, 0.80 + 1e-6)

    def test_low_vol_max_equity_allowed_100(self):
        """Low Vol: max_equity_allowed == 100.0"""
        result = self._run_optimize('Low Vol')
        self.assertAlmostEqual(result['max_equity_allowed'], 100.0, places=6)

    def test_result_dict_required_keys(self):
        """Result dict contains all 7 required keys"""
        required = {
            'weights', 'expected_return', 'volatility',
            'sharpe_ratio', 'regime', 'max_equity_allowed', 'cash_allocation'
        }
        result = self._run_optimize('Low Vol')
        self.assertTrue(required.issubset(set(result.keys())))

    def test_cash_present_constrained_regimes(self):
        """CASH key present in weights for High Vol, Trending, Mean Reversion"""
        for regime in ['High Vol', 'Trending', 'Mean Reversion']:
            result = self._run_optimize(regime)
            self.assertIn(
                'CASH', result['weights'],
                msg=f"CASH not in weights for {regime}"
            )


if __name__ == '__main__':
    unittest.main()
