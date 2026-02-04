"""
Portfolio Position Loader
Handles loading positions from CSV and fetching current market data
"""

import pandas as pd
import yfinance as yf
from typing import Dict, Tuple
from datetime import datetime, timedelta


class PortfolioLoader:
    """Load and manage portfolio positions"""

    def __init__(self):
        self.positions = None
        self.market_data = None

    def load_from_csv(self, csv_file) -> pd.DataFrame:
        """
        Load portfolio positions from CSV file

        Expected CSV format:
        ticker,shares,cost_basis,purchase_date
        AAPL,100,150.00,2023-01-15
        MSFT,50,300.00,2023-02-20
        """
        try:
            df = pd.read_csv(csv_file)

            # Validate required columns
            required_cols = ['ticker', 'shares', 'cost_basis']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")

            # Convert types
            df['shares'] = pd.to_numeric(df['shares'])
            df['cost_basis'] = pd.to_numeric(df['cost_basis'])

            # Handle purchase_date if present
            if 'purchase_date' in df.columns:
                df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')

            self.positions = df
            return df

        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")

    def create_sample_portfolio(self) -> pd.DataFrame:
        """Create a sample portfolio for demo purposes"""
        sample_data = {
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'SPY'],
            'shares': [100, 50, 30, 25, 200],
            'cost_basis': [150.00, 300.00, 120.00, 400.00, 400.00],
            'purchase_date': [
                '2023-01-15',
                '2023-02-20',
                '2023-03-10',
                '2023-04-05',
                '2023-05-12'
            ]
        }

        df = pd.DataFrame(sample_data)
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])

        self.positions = df
        return df

    def fetch_current_prices(self, tickers: list) -> Dict[str, float]:
        """
        Fetch current prices for list of tickers

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to current price
        """
        prices = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                # Get most recent price
                hist = stock.history(period='5d')
                if not hist.empty:
                    prices[ticker] = hist['Close'].iloc[-1]
                else:
                    prices[ticker] = None
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                prices[ticker] = None

        return prices

    def fetch_historical_data(
        self,
        tickers: list,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data for tickers

        Args:
            tickers: List of ticker symbols
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: today)

        Returns:
            Dictionary mapping ticker to historical price DataFrame
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        historical_data = {}

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                if not hist.empty:
                    historical_data[ticker] = hist
                else:
                    historical_data[ticker] = None
            except Exception as e:
                print(f"Error fetching historical data for {ticker}: {e}")
                historical_data[ticker] = None

        return historical_data

    def calculate_position_metrics(self, positions: pd.DataFrame, current_prices: Dict) -> pd.DataFrame:
        """
        Calculate P&L and metrics for each position

        Args:
            positions: DataFrame with position data
            current_prices: Dictionary of current prices

        Returns:
            DataFrame with calculated metrics
        """
        df = positions.copy()

        # Add current prices
        df['current_price'] = df['ticker'].map(current_prices)

        # Calculate values
        df['cost_value'] = df['shares'] * df['cost_basis']
        df['current_value'] = df['shares'] * df['current_price']

        # Calculate P&L
        df['total_pnl'] = df['current_value'] - df['cost_value']
        df['total_pnl_pct'] = (df['total_pnl'] / df['cost_value'] * 100)

        # Calculate position weights
        total_value = df['current_value'].sum()
        df['weight_pct'] = (df['current_value'] / total_value * 100)

        return df

    def get_portfolio_summary(self, position_metrics: pd.DataFrame) -> Dict:
        """
        Calculate portfolio-level summary metrics

        Args:
            position_metrics: DataFrame with position metrics

        Returns:
            Dictionary with summary metrics
        """
        total_cost = position_metrics['cost_value'].sum()
        total_value = position_metrics['current_value'].sum()
        total_pnl = position_metrics['total_pnl'].sum()
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

        # Number of positions
        num_positions = len(position_metrics)

        # Winners and losers
        winners = (position_metrics['total_pnl'] > 0).sum()
        losers = (position_metrics['total_pnl'] < 0).sum()

        # Largest position
        largest_position = position_metrics.loc[position_metrics['current_value'].idxmax()]

        summary = {
            'total_cost': total_cost,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'num_positions': num_positions,
            'winners': winners,
            'losers': losers,
            'largest_position_ticker': largest_position['ticker'],
            'largest_position_weight': largest_position['weight_pct']
        }

        return summary
