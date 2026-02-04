"""
Portfolio Optimizer
Regime-aware optimization using PyPortfolioOpt
"""

import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation
from typing import Dict, Tuple, List
from datetime import datetime, timedelta


class RegimeAwareOptimizer:
    """Optimize portfolio based on current regime"""

    def __init__(self):
        # Regime-based constraints
        self.regime_constraints = {
            'Low Vol': {
                'max_equity_exposure': 1.0,  # 100% equities allowed
                'risk_free_min': 0.0,
                'description': 'Aggressive allocation in calm markets'
            },
            'High Vol': {
                'max_equity_exposure': 0.6,  # Max 60% equities
                'risk_free_min': 0.4,  # Min 40% cash
                'description': 'Defensive allocation in volatile markets'
            },
            'Trending': {
                'max_equity_exposure': 0.9,  # 90% equities
                'risk_free_min': 0.1,
                'description': 'High exposure to capture trend'
            },
            'Mean Reversion': {
                'max_equity_exposure': 0.8,  # 80% equities
                'risk_free_min': 0.2,
                'description': 'Moderate allocation in choppy markets'
            }
        }

    def fetch_price_data(self, tickers: List[str], period: str = '2y') -> pd.DataFrame:
        """Fetch historical price data for all tickers"""
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

        # Forward fill missing data
        prices = prices.fillna(method='ffill')

        return prices

    def calculate_expected_returns(
        self,
        prices: pd.DataFrame,
        method: str = 'mean_historical_return'
    ) -> pd.Series:
        """Calculate expected returns"""
        if method == 'mean_historical_return':
            mu = expected_returns.mean_historical_return(prices, frequency=252)
        elif method == 'ema_historical_return':
            mu = expected_returns.ema_historical_return(prices, frequency=252)
        elif method == 'capm_return':
            mu = expected_returns.capm_return(prices, frequency=252)
        else:
            mu = expected_returns.mean_historical_return(prices, frequency=252)

        return mu

    def calculate_covariance(
        self,
        prices: pd.DataFrame,
        method: str = 'sample_cov'
    ) -> pd.DataFrame:
        """Calculate covariance matrix"""
        if method == 'sample_cov':
            S = risk_models.sample_cov(prices, frequency=252)
        elif method == 'semicovariance':
            S = risk_models.semicovariance(prices, frequency=252)
        elif method == 'exp_cov':
            S = risk_models.exp_cov(prices, frequency=252)
        elif method == 'ledoit_wolf':
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        else:
            S = risk_models.sample_cov(prices, frequency=252)

        return S

    def optimize_portfolio(
        self,
        tickers: List[str],
        current_regime: str,
        optimization_method: str = 'max_sharpe',
        position_scores: Dict[str, float] = None
    ) -> Dict:
        """
        Optimize portfolio allocation based on regime

        Args:
            tickers: List of tickers
            current_regime: Current market regime
            optimization_method: 'max_sharpe', 'min_volatility', 'max_quadratic_utility'
            position_scores: Optional dict of position scores to tilt allocation

        Returns:
            Dictionary with optimal weights and metrics
        """
        # Fetch price data
        prices = self.fetch_price_data(tickers, period='2y')

        if prices.empty:
            raise ValueError("Could not fetch price data")

        # Calculate expected returns and covariance
        mu = self.calculate_expected_returns(prices, method='mean_historical_return')
        S = self.calculate_covariance(prices, method='ledoit_wolf')

        # Get regime constraints
        constraints = self.regime_constraints.get(current_regime, self.regime_constraints['High Vol'])

        # Initialize efficient frontier
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.30))  # Max 30% per position

        # Add regime-based constraint for total equity exposure
        max_equity = constraints['max_equity_exposure']

        # Optimize based on method
        if optimization_method == 'max_sharpe':
            weights = ef.max_sharpe(risk_free_rate=0.04)
        elif optimization_method == 'min_volatility':
            weights = ef.min_volatility()
        elif optimization_method == 'max_quadratic_utility':
            ef.max_quadratic_utility(risk_aversion=1)
            weights = ef.clean_weights()
        else:
            weights = ef.max_sharpe(risk_free_rate=0.04)

        # Clean weights
        cleaned_weights = ef.clean_weights()

        # Calculate performance
        performance = ef.portfolio_performance(verbose=False, risk_free_rate=0.04)
        expected_return = performance[0]
        volatility = performance[1]
        sharpe_ratio = performance[2]

        # Add cash allocation based on regime
        total_equity_weight = sum(cleaned_weights.values())

        if total_equity_weight > max_equity:
            # Scale down to meet regime constraint
            scale_factor = max_equity / total_equity_weight
            cleaned_weights = {k: v * scale_factor for k, v in cleaned_weights.items()}

        # Add cash
        cash_weight = 1 - sum(cleaned_weights.values())
        if cash_weight > 0:
            cleaned_weights['CASH'] = cash_weight

        result = {
            'weights': cleaned_weights,
            'expected_return': expected_return * 100,  # As percentage
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'regime': current_regime,
            'max_equity_allowed': max_equity * 100,
            'cash_allocation': cash_weight * 100
        }

        return result

    def calculate_rebalancing_trades(
        self,
        current_positions: pd.DataFrame,
        optimal_weights: Dict[str, float],
        current_prices: Dict[str, float],
        total_portfolio_value: float
    ) -> pd.DataFrame:
        """
        Calculate trades needed to rebalance to optimal weights

        Args:
            current_positions: DataFrame with current holdings
            optimal_weights: Dictionary of optimal weights
            current_prices: Dictionary of current prices
            total_portfolio_value: Total portfolio value

        Returns:
            DataFrame with rebalancing trades
        """
        trades = []

        # Calculate current weights
        current_weights = {}
        for _, pos in current_positions.iterrows():
            ticker = pos['ticker']
            value = pos['shares'] * current_prices.get(ticker, 0)
            current_weights[ticker] = value / total_portfolio_value

        # Calculate target values
        all_tickers = set(list(current_weights.keys()) + list(optimal_weights.keys()))

        for ticker in all_tickers:
            if ticker == 'CASH':
                continue

            current_weight = current_weights.get(ticker, 0)
            optimal_weight = optimal_weights.get(ticker, 0)
            weight_diff = optimal_weight - current_weight

            current_value = current_weight * total_portfolio_value
            optimal_value = optimal_weight * total_portfolio_value
            value_diff = optimal_value - current_value

            if abs(value_diff) > 100:  # Only if difference > $100
                current_price = current_prices.get(ticker, 0)

                if current_price > 0:
                    shares_diff = value_diff / current_price

                    action = "BUY" if shares_diff > 0 else "SELL"

                    trades.append({
                        'Ticker': ticker,
                        'Action': action,
                        'Shares': abs(shares_diff),
                        'Price': current_price,
                        'Value': abs(value_diff),
                        'Current Weight': current_weight * 100,
                        'Target Weight': optimal_weight * 100,
                        'Weight Change': weight_diff * 100
                    })

        # Add cash adjustment
        current_cash_weight = current_weights.get('CASH', 0)
        optimal_cash_weight = optimal_weights.get('CASH', 0)

        if abs(optimal_cash_weight - current_cash_weight) > 0.01:  # If >1% difference
            cash_diff = (optimal_cash_weight - current_cash_weight) * total_portfolio_value

            trades.append({
                'Ticker': 'CASH',
                'Action': 'INCREASE' if cash_diff > 0 else 'DECREASE',
                'Shares': 1,  # N/A for cash
                'Price': 1,
                'Value': abs(cash_diff),
                'Current Weight': current_cash_weight * 100,
                'Target Weight': optimal_cash_weight * 100,
                'Weight Change': (optimal_cash_weight - current_cash_weight) * 100
            })

        trades_df = pd.DataFrame(trades)

        if not trades_df.empty:
            trades_df = trades_df.sort_values('Value', ascending=False)

        return trades_df

    def calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        prices: pd.DataFrame
    ) -> Dict:
        """Calculate various portfolio metrics"""
        # Remove cash from prices
        weights_no_cash = {k: v for k, v in weights.items() if k != 'CASH'}

        # Renormalize weights
        total_weight = sum(weights_no_cash.values())
        if total_weight > 0:
            weights_normalized = {k: v / total_weight for k, v in weights_no_cash.items()}
        else:
            return {}

        # Calculate weighted returns
        returns = prices.pct_change().dropna()

        portfolio_returns = pd.Series(0.0, index=returns.index)

        for ticker, weight in weights_normalized.items():
            if ticker in returns.columns:
                portfolio_returns += returns[ticker] * weight

        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0

        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        metrics = {
            'total_return': total_return * 100,
            'annualized_volatility': volatility * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd * 100
        }

        return metrics
