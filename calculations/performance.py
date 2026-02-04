"""
Portfolio Performance Analytics
Calculate returns, volatility, Sharpe ratio, drawdowns
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime, timedelta


class PerformanceAnalytics:
    """Calculate portfolio performance metrics"""

    def __init__(self):
        pass

    def calculate_returns(self, historical_data: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.Series:
        """
        Calculate portfolio returns based on historical data and weights

        Args:
            historical_data: Dictionary mapping ticker to historical DataFrame
            weights: Dictionary mapping ticker to weight (as decimal)

        Returns:
            Series with portfolio returns
        """
        # Align all data to common dates
        all_prices = pd.DataFrame()

        for ticker, hist in historical_data.items():
            if hist is not None and not hist.empty:
                all_prices[ticker] = hist['Close']

        # Forward fill missing data
        all_prices = all_prices.fillna(method='ffill')

        # Calculate returns for each asset
        returns = all_prices.pct_change()

        # Calculate weighted portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns.index)

        for ticker, weight in weights.items():
            if ticker in returns.columns:
                portfolio_returns += returns[ticker] * weight

        return portfolio_returns.dropna()

    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns)

        Args:
            returns: Series of returns
            annualize: If True, annualize the volatility

        Returns:
            Volatility (annualized if specified)
        """
        vol = returns.std()

        if annualize:
            vol = vol * np.sqrt(252)  # Assuming daily returns

        return vol

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.04,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe ratio

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (default 4%)
            annualize: If True, annualize the Sharpe ratio

        Returns:
            Sharpe ratio
        """
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()

        if std_excess == 0:
            return 0.0

        sharpe = mean_excess / std_excess

        if annualize:
            sharpe = sharpe * np.sqrt(252)

        return sharpe

    def calculate_max_drawdown(self, returns: pd.Series) -> Tuple[float, datetime, datetime]:
        """
        Calculate maximum drawdown

        Args:
            returns: Series of returns

        Returns:
            Tuple of (max_drawdown, start_date, end_date)
        """
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()

        # Calculate running maximum
        running_max = cumulative.expanding().max()

        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max

        # Find maximum drawdown
        max_dd = drawdown.min()

        # Find dates
        end_date = drawdown.idxmin()
        start_date = cumulative[:end_date].idxmax()

        return max_dd, start_date, end_date

    def calculate_cagr(self, returns: pd.Series) -> float:
        """
        Calculate Compound Annual Growth Rate

        Args:
            returns: Series of returns

        Returns:
            CAGR as decimal
        """
        cumulative_return = (1 + returns).prod()
        n_years = len(returns) / 252  # Assuming daily returns

        if n_years == 0:
            return 0.0

        cagr = (cumulative_return ** (1 / n_years)) - 1

        return cagr

    def calculate_all_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.04
    ) -> Dict:
        """
        Calculate all performance metrics

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary with all metrics
        """
        total_return = (1 + returns).prod() - 1
        cagr = self.calculate_cagr(returns)
        volatility = self.calculate_volatility(returns, annualize=True)
        sharpe = self.calculate_sharpe_ratio(returns, risk_free_rate, annualize=True)
        max_dd, dd_start, dd_end = self.calculate_max_drawdown(returns)

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) * 100

        # Best and worst day
        best_day = returns.max()
        worst_day = returns.min()

        metrics = {
            'total_return': total_return * 100,  # As percentage
            'cagr': cagr * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd * 100,
            'drawdown_start': dd_start,
            'drawdown_end': dd_end,
            'win_rate': win_rate,
            'best_day': best_day * 100,
            'worst_day': worst_day * 100,
            'num_periods': len(returns)
        }

        return metrics
