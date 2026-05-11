"""
Portfolio Position Loader
Handles loading positions from CSV and fetching current market data via Alpaca.
"""

import pandas as pd
from typing import Dict
from datetime import datetime, timedelta

from data import alpaca_client as alpaca


class PortfolioLoader:
    """Load and manage portfolio positions"""

    def __init__(self):
        self.positions   = None
        self.market_data = None

    def load_from_csv(self, csv_file) -> pd.DataFrame:
        """
        Load portfolio positions from CSV file.
        Required columns: ticker, shares, cost_basis
        Optional: purchase_date
        """
        try:
            df = pd.read_csv(csv_file)
            required_cols = ['ticker', 'shares', 'cost_basis']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            df['shares']     = pd.to_numeric(df['shares'])
            df['cost_basis'] = pd.to_numeric(df['cost_basis'])
            if 'purchase_date' in df.columns:
                df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
            self.positions = df
            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")

    def create_sample_portfolio(self) -> pd.DataFrame:
        """Create a sample portfolio for demo purposes."""
        sample_data = {
            'ticker':        ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'SPY'],
            'shares':        [100, 50, 30, 25, 200],
            'cost_basis':    [150.00, 300.00, 120.00, 400.00, 400.00],
            'purchase_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12'],
        }
        df = pd.DataFrame(sample_data)
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        self.positions = df
        return df

    def fetch_current_prices(self, tickers: list) -> Dict[str, float]:
        """
        Fetch current prices for a list of tickers.
        Tries Alpaca latest bar first; falls back to yfinance if Alpaca
        returns no data (e.g. no API keys / IEX tier limitation).
        Returns dict mapping ticker → price (None if unavailable).
        """
        prices = alpaca.get_latest_prices(tickers)
        missing = [t for t in tickers if prices.get(t.upper()) is None]
        if missing:
            try:
                import yfinance as yf
                hist = yf.download(
                    " ".join(missing), period="5d", auto_adjust=True,
                    progress=False, threads=False,
                )
                if not hist.empty:
                    close = hist["Close"] if "Close" in hist.columns else hist
                    if hasattr(close, "columns"):
                        for t in missing:
                            col = t.upper()
                            if col in close.columns:
                                s = close[col].dropna()
                                if not s.empty:
                                    prices[col] = float(s.iloc[-1])
                    else:
                        # single ticker
                        s = close.dropna()
                        if not s.empty and len(missing) == 1:
                            prices[missing[0].upper()] = float(s.iloc[-1])
            except Exception:
                pass
        return {t: prices.get(t.upper()) for t in tickers}

    def fetch_historical_data(
        self,
        tickers: list,
        start_date: datetime = None,
        end_date:   datetime = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical daily bars for each ticker via Alpaca.
        Returns dict mapping ticker → OHLCV DataFrame.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        historical_data = {}
        for ticker in tickers:
            try:
                hist = alpaca.get_bars(ticker, start=start_date, end=end_date)
                historical_data[ticker] = hist if not hist.empty else None
            except Exception as e:
                print(f"Error fetching historical data for {ticker}: {e}")
                historical_data[ticker] = None

        return historical_data

    def calculate_position_metrics(self, positions: pd.DataFrame, current_prices: Dict) -> pd.DataFrame:
        """Calculate P&L and metrics for each position."""
        df = positions.copy()
        df['current_price'] = df['ticker'].map(current_prices)
        df['cost_value']    = df['shares'] * df['cost_basis']
        df['current_value'] = df['shares'] * df['current_price']
        df['total_pnl']     = df['current_value'] - df['cost_value']
        df['total_pnl_pct'] = df['total_pnl'] / df['cost_value'].replace(0, float('nan')) * 100
        total_value         = df['current_value'].sum()
        df['weight_pct']    = df['current_value'] / total_value * 100 if total_value != 0 else 0.0
        return df

    def get_portfolio_summary(self, position_metrics: pd.DataFrame) -> Dict:
        """Calculate portfolio-level summary metrics."""
        total_cost  = position_metrics['cost_value'].sum()
        total_value = position_metrics['current_value'].sum()
        total_pnl   = position_metrics['total_pnl'].sum()

        largest_pos = position_metrics.loc[position_metrics['current_value'].idxmax()]

        return {
            'total_cost':               total_cost,
            'total_value':              total_value,
            'total_pnl':                total_pnl,
            'total_pnl_pct':            (total_pnl / total_cost * 100) if total_cost > 0 else 0,
            'num_positions':            len(position_metrics),
            'winners':                  (position_metrics['total_pnl'] > 0).sum(),
            'losers':                   (position_metrics['total_pnl'] < 0).sum(),
            'largest_position_ticker':  largest_pos['ticker'],
            'largest_position_weight':  largest_pos['weight_pct'],
        }
