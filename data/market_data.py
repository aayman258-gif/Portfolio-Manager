"""
Market Data Loader
Fetch market indices and macro indicators via Alpaca.
VIX data uses yfinance (not available through Alpaca).
"""

import pandas as pd
import numpy as np
import yfinance as yf           # kept for VIX only
from datetime import datetime, timedelta
from typing import Dict, Tuple

from data import alpaca_client as alpaca


class MarketDataLoader:
    """Load market data and macro indicators"""

    def __init__(self):
        self.indices = {
            'SPY': 'S&P 500',
            'QQQ': 'Nasdaq 100',
            'IWM': 'Russell 2000',
            'DIA': 'Dow Jones'
        }

    def load_index_data(self, ticker: str = 'SPY', period: str = '2y') -> pd.DataFrame:
        """Load daily OHLCV for a market index. Falls back to yfinance on Alpaca error."""
        try:
            data = alpaca.get_bars(ticker, period=period)
            if not data.empty:
                return data
        except Exception:
            pass

        # yfinance fallback
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            return data
        except Exception as e:
            raise ValueError(f"Error loading {ticker}: {str(e)}")

    def load_vix_data(self, period: str = '2y') -> pd.Series:
        """Load VIX volatility index (yfinance — not available on Alpaca)."""
        try:
            vix = yf.download('^VIX', period=period, progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            return vix['Close'] if 'Close' in vix.columns else vix['Adj Close']
        except Exception as e:
            raise ValueError(f"Error loading VIX: {str(e)}")

    def align_data(self, spy_data: pd.DataFrame, vix_data: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align SPY and VIX data to matching dates."""
        if 'Close' in spy_data.columns:
            spy_close = spy_data['Close']
        elif 'Adj Close' in spy_data.columns:
            spy_close = spy_data['Adj Close']
        else:
            raise ValueError(f"No 'Close' column. Available: {spy_data.columns.tolist()}")

        common_dates = spy_close.index.intersection(vix_data.index)
        return spy_close.loc[common_dates], vix_data.loc[common_dates]

    def get_current_market_stats(self, spy_data: pd.DataFrame, vix_current: float) -> Dict:
        """Calculate current market statistics."""
        prices = spy_data['Close'] if 'Close' in spy_data.columns else spy_data['Adj Close']
        current_price = prices.iloc[-1]

        returns_1d  = (prices.iloc[-1] / prices.iloc[-2]  - 1) * 100 if len(prices) > 1  else 0
        returns_5d  = (prices.iloc[-1] / prices.iloc[-6]  - 1) * 100 if len(prices) > 5  else 0
        returns_1m  = (prices.iloc[-1] / prices.iloc[-22] - 1) * 100 if len(prices) > 22 else 0
        returns_3m  = (prices.iloc[-1] / prices.iloc[-66] - 1) * 100 if len(prices) > 66 else 0
        returns_ytd = (
            (prices.iloc[-1] / prices[prices.index.year == prices.index[-1].year][0] - 1) * 100
            if len(prices) > 0 else 0
        )

        returns  = prices.pct_change()
        vol_20d  = returns.tail(20).std() * np.sqrt(252) * 100
        vol_60d  = returns.tail(60).std() * np.sqrt(252) * 100

        rolling_max       = prices.rolling(window=252).max()
        distance_from_high = (prices.iloc[-1] / rolling_max.iloc[-1] - 1) * 100

        ma_50  = prices.rolling(window=50).mean().iloc[-1]
        ma_200 = prices.rolling(window=200).mean().iloc[-1]

        return {
            'current_price':      current_price,
            'returns_1d':         returns_1d,
            'returns_5d':         returns_5d,
            'returns_1m':         returns_1m,
            'returns_3m':         returns_3m,
            'returns_ytd':        returns_ytd,
            'vix_current':        vix_current,
            'vol_20d':            vol_20d,
            'vol_60d':            vol_60d,
            'distance_from_high': distance_from_high,
            'ma_50':              ma_50,
            'ma_200':             ma_200,
            'above_ma50':         current_price > ma_50,
            'above_ma200':        current_price > ma_200,
        }

    def get_sector_performance(self, period: str = '1mo') -> pd.DataFrame:
        """Get sector ETF performance."""
        sector_etfs = {
            'XLK':  'Technology',
            'XLF':  'Financials',
            'XLV':  'Healthcare',
            'XLE':  'Energy',
            'XLI':  'Industrials',
            'XLC':  'Communications',
            'XLY':  'Consumer Discretionary',
            'XLP':  'Consumer Staples',
            'XLU':  'Utilities',
            'XLRE': 'Real Estate',
            'XLB':  'Materials',
        }

        performance = []
        for ticker, name in sector_etfs.items():
            try:
                data = self.load_index_data(ticker, period=period)
                if not data.empty and 'Close' in data.columns:
                    ret = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    performance.append({'Sector': name, 'Ticker': ticker, 'Return': ret})
            except Exception:
                continue

        return pd.DataFrame(performance).sort_values('Return', ascending=False)
