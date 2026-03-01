"""
Portfolio Optimizer
Regime-aware optimization using PyPortfolioOpt.

Expected-return methods (selectable in UI):
  mean_historical  — simple annualised historical mean  (baseline)
  ema              — exponentially-weighted historical mean
  capm             — CAPM: rf + β × (E[Rm] - rf), betas vs SPY
  black_litterman  — BL with CAPM equilibrium prior; no explicit views
                     (posterior ≈ shrinkage-adjusted CAPM)
"""

import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta


# ── Constants ──────────────────────────────────────────────────────────────────
_RISK_FREE   = 0.04    # annual risk-free rate used throughout
_MKT_PREMIUM = 0.055   # expected equity risk premium (Damodaran estimate)
_SPY_PERIOD  = "2y"    # lookback for market proxy


class RegimeAwareOptimizer:
    """Optimize portfolio allocation conditioned on market regime."""

    def __init__(self):
        self.regime_constraints = {
            'Low Vol': {
                'max_equity_exposure': 1.0,
                'risk_free_min': 0.0,
                'description': 'Aggressive allocation in calm markets',
            },
            'High Vol': {
                'max_equity_exposure': 0.6,
                'risk_free_min': 0.4,
                'description': 'Defensive allocation in volatile markets',
            },
            'Trending': {
                'max_equity_exposure': 0.9,
                'risk_free_min': 0.1,
                'description': 'High exposure to capture trend',
            },
            'Mean Reversion': {
                'max_equity_exposure': 0.8,
                'risk_free_min': 0.2,
                'description': 'Moderate allocation in choppy markets',
            },
            'Uncertain': {
                'max_equity_exposure': 0.65,
                'risk_free_min': 0.35,
                'description': 'Conservative allocation while regime is ambiguous',
            },
        }

    # ── Data fetching ─────────────────────────────────────────────────────────

    def fetch_price_data(self, tickers: List[str], period: str = '2y') -> pd.DataFrame:
        """Fetch historical price data for all tickers."""
        import yfinance as yf

        prices = pd.DataFrame()
        for ticker in tickers:
            try:
                data = yf.download(ticker, period=period, progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                if not data.empty and 'Close' in data.columns:
                    prices[ticker] = data['Close']
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")

        prices = prices.ffill()
        return prices

    def _fetch_market_proxy(self) -> Optional[pd.Series]:
        """Fetch SPY close prices as the market proxy."""
        import yfinance as yf
        try:
            raw = yf.download("SPY", period=_SPY_PERIOD, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            return raw["Close"].dropna()
        except Exception:
            return None

    # ── Expected return methods ───────────────────────────────────────────────

    def calculate_expected_returns(
        self,
        prices: pd.DataFrame,
        method: str = 'capm',
    ) -> pd.Series:
        """
        Calculate annualised expected returns.

        Parameters
        ----------
        method : str
            'mean_historical' | 'ema' | 'capm' | 'black_litterman'
        """
        if method == 'mean_historical':
            return expected_returns.mean_historical_return(prices, frequency=252)

        if method == 'ema':
            return expected_returns.ema_historical_return(prices, frequency=252)

        if method in ('capm', 'black_litterman'):
            return self._capm_returns(prices)

        # fallback
        return expected_returns.mean_historical_return(prices, frequency=252)

    def _capm_returns(self, prices: pd.DataFrame) -> pd.Series:
        """
        CAPM expected returns: E(Ri) = rf + β_i × ERP

        Betas are estimated via OLS against SPY daily returns over the
        shared history.  If SPY data is unavailable, falls back to
        PyPortfolioOpt's built-in capm_return().
        """
        spy = self._fetch_market_proxy()
        if spy is None or len(spy) < 60:
            return expected_returns.capm_return(prices, frequency=252)

        mkt_ret = spy.pct_change().dropna()
        mu = {}
        for col in prices.columns:
            asset_ret = prices[col].pct_change().dropna()
            common    = asset_ret.index.intersection(mkt_ret.index)
            if len(common) < 30:
                mu[col] = _RISK_FREE + _MKT_PREMIUM   # fallback: beta = 1
                continue
            a = asset_ret.loc[common].values
            m = mkt_ret.loc[common].values
            cov_mat = np.cov(a, m, ddof=1)
            beta    = cov_mat[0, 1] / (cov_mat[1, 1] + 1e-12)
            mu[col] = _RISK_FREE + beta * _MKT_PREMIUM

        return pd.Series(mu)

    def _black_litterman_returns(
        self,
        prices: pd.DataFrame,
        S: pd.DataFrame,
    ) -> pd.Series:
        """
        Black-Litterman posterior expected returns.

        Prior  : CAPM equilibrium returns (computed above).
        Views  : None supplied — posterior collapses to the prior with
                 Ledoit-Wolf shrinkage applied through the covariance.
        Result : A shrinkage-adjusted CAPM estimate that is more
                 stable than simple historical means out-of-sample.

        With analyst views, replace `bl = BlackLittermanModel(S, pi=pi)`
        with Q (view vector) and P (picking matrix).
        """
        from pypfopt import BlackLittermanModel

        pi = self._capm_returns(prices)   # equilibrium prior

        # Align pi index with covariance matrix columns
        pi = pi.reindex(S.columns).fillna(_RISK_FREE + _MKT_PREMIUM)

        try:
            bl    = BlackLittermanModel(S, pi=pi)
            mu_bl = bl.bl_returns()
            return mu_bl
        except Exception:
            return pi   # fall back to CAPM prior

    # ── Covariance ────────────────────────────────────────────────────────────

    def calculate_covariance(
        self,
        prices: pd.DataFrame,
        method: str = 'ledoit_wolf',
    ) -> pd.DataFrame:
        """Calculate covariance matrix."""
        if method == 'sample_cov':
            return risk_models.sample_cov(prices, frequency=252)
        if method == 'semicovariance':
            return risk_models.semicovariance(prices, frequency=252)
        if method == 'exp_cov':
            return risk_models.exp_cov(prices, frequency=252)
        # default: Ledoit-Wolf shrinkage
        return risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # ── Main optimiser ────────────────────────────────────────────────────────

    def optimize_portfolio(
        self,
        tickers: List[str],
        current_regime: str,
        optimization_method: str = 'max_sharpe',
        expected_return_method: str = 'capm',
        position_scores: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Optimize portfolio allocation conditioned on regime.

        Parameters
        ----------
        expected_return_method : str
            'mean_historical' | 'ema' | 'capm' | 'black_litterman'
        """
        prices = self.fetch_price_data(tickers, period='2y')
        if prices.empty:
            raise ValueError("Could not fetch price data for any ticker.")

        # Covariance (always Ledoit-Wolf)
        S  = self.calculate_covariance(prices, method='ledoit_wolf')

        # Expected returns — method-dependent
        if expected_return_method == 'black_litterman':
            mu = self._black_litterman_returns(prices, S)
        else:
            mu = self.calculate_expected_returns(prices, method=expected_return_method)

        # Align mu / S to the same tickers
        common = list(set(mu.index) & set(S.columns) & set(prices.columns))
        if not common:
            raise ValueError("No common tickers after computing mu/S.")
        mu = mu.loc[common]
        S  = S.loc[common, common]

        # Regime constraints
        constraints = self.regime_constraints.get(
            current_regime,
            self.regime_constraints['High Vol'],
        )
        max_equity = constraints['max_equity_exposure']

        ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.30))

        if optimization_method == 'max_sharpe':
            ef.max_sharpe(risk_free_rate=_RISK_FREE)
        elif optimization_method == 'min_volatility':
            ef.min_volatility()
        elif optimization_method == 'max_quadratic_utility':
            ef.max_quadratic_utility(risk_aversion=1)
        else:
            ef.max_sharpe(risk_free_rate=_RISK_FREE)

        cleaned_weights = ef.clean_weights()
        performance     = ef.portfolio_performance(verbose=False, risk_free_rate=_RISK_FREE)
        expected_return, volatility, sharpe_ratio = performance

        # Apply regime equity cap
        total_eq = sum(cleaned_weights.values())
        if total_eq > max_equity:
            scale = max_equity / total_eq
            cleaned_weights = {k: v * scale for k, v in cleaned_weights.items()}

        cash_weight = 1.0 - sum(cleaned_weights.values())
        if cash_weight > 0:
            cleaned_weights['CASH'] = cash_weight

        return {
            'weights':             cleaned_weights,
            'expected_return':     expected_return * 100,
            'volatility':          volatility * 100,
            'sharpe_ratio':        sharpe_ratio,
            'regime':              current_regime,
            'max_equity_allowed':  max_equity * 100,
            'cash_allocation':     cash_weight * 100,
            'return_method':       expected_return_method,
        }

    # ── Rebalancing ───────────────────────────────────────────────────────────

    def calculate_rebalancing_trades(
        self,
        current_positions: pd.DataFrame,
        optimal_weights: Dict[str, float],
        current_prices: Dict[str, float],
        total_portfolio_value: float,
    ) -> pd.DataFrame:
        """Calculate trades needed to reach optimal weights."""
        trades = []

        current_weights: Dict[str, float] = {}
        for _, pos in current_positions.iterrows():
            ticker = pos['ticker']
            value  = pos['shares'] * current_prices.get(ticker, 0)
            current_weights[ticker] = value / total_portfolio_value

        all_tickers = set(list(current_weights.keys()) + list(optimal_weights.keys()))

        for ticker in all_tickers:
            if ticker == 'CASH':
                continue
            cw = current_weights.get(ticker, 0)
            ow = optimal_weights.get(ticker, 0)
            vd = (ow - cw) * total_portfolio_value

            if abs(vd) > 100:
                price = current_prices.get(ticker, 0)
                if price > 0:
                    trades.append({
                        'Ticker':         ticker,
                        'Action':         'BUY' if vd > 0 else 'SELL',
                        'Shares':         abs(vd / price),
                        'Price':          price,
                        'Value':          abs(vd),
                        'Current Weight': cw * 100,
                        'Target Weight':  ow * 100,
                        'Weight Change':  (ow - cw) * 100,
                    })

        cw_cash = current_weights.get('CASH', 0)
        ow_cash = optimal_weights.get('CASH', 0)
        if abs(ow_cash - cw_cash) > 0.01:
            cash_diff = (ow_cash - cw_cash) * total_portfolio_value
            trades.append({
                'Ticker':         'CASH',
                'Action':         'INCREASE' if cash_diff > 0 else 'DECREASE',
                'Shares':         1,
                'Price':          1,
                'Value':          abs(cash_diff),
                'Current Weight': cw_cash * 100,
                'Target Weight':  ow_cash * 100,
                'Weight Change':  (ow_cash - cw_cash) * 100,
            })

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df = trades_df.sort_values('Value', ascending=False)
        return trades_df

    # ── Portfolio metrics ─────────────────────────────────────────────────────

    def calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        prices: pd.DataFrame,
    ) -> Dict:
        """Calculate various portfolio metrics."""
        weights_no_cash = {k: v for k, v in weights.items() if k != 'CASH'}
        total_w = sum(weights_no_cash.values())
        if total_w == 0:
            return {}

        weights_norm = {k: v / total_w for k, v in weights_no_cash.items()}
        returns      = prices.pct_change().dropna()
        port_ret     = pd.Series(0.0, index=returns.index)

        for ticker, w in weights_norm.items():
            if ticker in returns.columns:
                port_ret += returns[ticker] * w

        total_return = (1 + port_ret).prod() - 1
        volatility   = port_ret.std() * np.sqrt(252)
        sharpe       = (port_ret.mean() / port_ret.std() * np.sqrt(252)) if port_ret.std() > 0 else 0

        cumulative  = (1 + port_ret).cumprod()
        running_max = cumulative.expanding().max()
        max_dd      = ((cumulative - running_max) / running_max).min()

        return {
            'total_return':          total_return * 100,
            'annualized_volatility': volatility * 100,
            'sharpe_ratio':          sharpe,
            'max_drawdown':          max_dd * 100,
        }
