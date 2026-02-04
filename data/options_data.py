"""
Options Data Loader
Fetch live options chains using yfinance API
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class OptionsDataLoader:
    """Load live options data from public APIs"""

    def __init__(self):
        self.ticker_cache = {}

    def get_options_expirations(self, ticker: str) -> List[str]:
        """
        Get available expiration dates for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of expiration date strings (YYYY-MM-DD)
        """
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            return list(expirations)
        except Exception as e:
            print(f"Error fetching expirations for {ticker}: {e}")
            return []

    def get_options_chain(
        self,
        ticker: str,
        expiration: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Get options chain for a specific expiration

        Args:
            ticker: Stock ticker symbol
            expiration: Expiration date (YYYY-MM-DD). If None, uses nearest expiration

        Returns:
            (calls_df, puts_df, underlying_price)
        """
        try:
            stock = yf.Ticker(ticker)

            # Get current stock price
            hist = stock.history(period='1d')
            if hist.empty:
                raise ValueError(f"Could not fetch current price for {ticker}")

            underlying_price = hist['Close'].iloc[-1]

            # Get expiration dates
            expirations = stock.options

            if not expirations:
                raise ValueError(f"No options available for {ticker}")

            # Use provided expiration or nearest one
            if expiration is None:
                expiration = expirations[0]
            elif expiration not in expirations:
                raise ValueError(f"Expiration {expiration} not available for {ticker}")

            # Get options chain
            options = stock.option_chain(expiration)

            calls = options.calls
            puts = options.puts

            # Add calculated fields
            calls = self._enrich_options_data(calls, underlying_price, 'call')
            puts = self._enrich_options_data(puts, underlying_price, 'put')

            return calls, puts, underlying_price

        except Exception as e:
            print(f"Error fetching options chain: {e}")
            return pd.DataFrame(), pd.DataFrame(), 0.0

    def _enrich_options_data(
        self,
        options_df: pd.DataFrame,
        underlying_price: float,
        option_type: str
    ) -> pd.DataFrame:
        """Add calculated fields to options dataframe"""
        if options_df.empty:
            return options_df

        df = options_df.copy()

        # Calculate moneyness
        if option_type == 'call':
            df['moneyness'] = underlying_price - df['strike']
            df['moneyness_pct'] = (underlying_price / df['strike'] - 1) * 100
            df['ITM'] = df['moneyness'] > 0
        else:
            df['moneyness'] = df['strike'] - underlying_price
            df['moneyness_pct'] = (df['strike'] / underlying_price - 1) * 100
            df['ITM'] = df['moneyness'] > 0

        # Intrinsic value
        df['intrinsicValue'] = df['moneyness'].apply(lambda x: max(x, 0))

        # Time value
        df['timeValue'] = df['lastPrice'] - df['intrinsicValue']

        # Bid-ask spread
        df['bidAskSpread'] = df['ask'] - df['bid']
        df['bidAskSpreadPct'] = (df['bidAskSpread'] / df['lastPrice'] * 100).fillna(0)

        # Mid price
        df['midPrice'] = (df['bid'] + df['ask']) / 2

        return df

    def get_option_quote(
        self,
        ticker: str,
        strike: float,
        expiration: str,
        option_type: str
    ) -> Dict:
        """
        Get specific option quote

        Args:
            ticker: Stock ticker
            strike: Strike price
            expiration: Expiration date
            option_type: 'call' or 'put'

        Returns:
            Dictionary with option data
        """
        try:
            calls, puts, underlying = self.get_options_chain(ticker, expiration)

            if option_type == 'call':
                df = calls
            else:
                df = puts

            # Find matching strike
            option = df[df['strike'] == strike]

            if option.empty:
                # Find nearest strike
                df['strike_diff'] = abs(df['strike'] - strike)
                option = df.nsmallest(1, 'strike_diff')
                print(f"Exact strike {strike} not found, using nearest: {option['strike'].iloc[0]}")

            if option.empty:
                return {}

            # Convert to dict
            result = option.iloc[0].to_dict()
            result['underlying_price'] = underlying
            result['ticker'] = ticker
            result['expiration'] = expiration
            result['option_type'] = option_type

            return result

        except Exception as e:
            print(f"Error fetching option quote: {e}")
            return {}

    def get_atm_options(
        self,
        ticker: str,
        expiration: str = None,
        num_strikes: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Get ATM and near-ATM options

        Args:
            ticker: Stock ticker
            expiration: Expiration date (uses nearest if None)
            num_strikes: Number of strikes above and below ATM to return

        Returns:
            (atm_calls, atm_puts, underlying_price)
        """
        calls, puts, underlying = self.get_options_chain(ticker, expiration)

        if calls.empty or puts.empty:
            return calls, puts, underlying

        # Find ATM strike (closest to underlying price)
        calls['distance_from_atm'] = abs(calls['strike'] - underlying)
        puts['distance_from_atm'] = abs(puts['strike'] - underlying)

        atm_calls = calls.nsmallest(num_strikes, 'distance_from_atm')
        atm_puts = puts.nsmallest(num_strikes, 'distance_from_atm')

        return atm_calls, atm_puts, underlying

    def get_options_summary(self, ticker: str) -> Dict:
        """
        Get summary of available options for a ticker

        Args:
            ticker: Stock ticker

        Returns:
            Summary dictionary
        """
        try:
            stock = yf.Ticker(ticker)

            # Get current price
            hist = stock.history(period='1d')
            current_price = hist['Close'].iloc[-1] if not hist.empty else 0

            # Get expirations
            expirations = list(stock.options)

            # Get nearest expiration chain
            if expirations:
                calls, puts, _ = self.get_options_chain(ticker, expirations[0])

                summary = {
                    'ticker': ticker,
                    'current_price': current_price,
                    'num_expirations': len(expirations),
                    'nearest_expiration': expirations[0],
                    'furthest_expiration': expirations[-1],
                    'num_call_strikes': len(calls),
                    'num_put_strikes': len(puts),
                    'total_call_volume': calls['volume'].sum() if not calls.empty else 0,
                    'total_put_volume': puts['volume'].sum() if not puts.empty else 0,
                    'total_call_oi': calls['openInterest'].sum() if not calls.empty else 0,
                    'total_put_oi': puts['openInterest'].sum() if not puts.empty else 0,
                }

                # Put/Call ratio
                if summary['total_call_volume'] > 0:
                    summary['put_call_ratio'] = summary['total_put_volume'] / summary['total_call_volume']
                else:
                    summary['put_call_ratio'] = 0

                return summary
            else:
                return {
                    'ticker': ticker,
                    'current_price': current_price,
                    'error': 'No options available'
                }

        except Exception as e:
            return {
                'ticker': ticker,
                'error': str(e)
            }

    def calculate_implied_volatility_smile(
        self,
        ticker: str,
        expiration: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get IV smile/skew data

        Args:
            ticker: Stock ticker
            expiration: Expiration date

        Returns:
            (calls_iv_data, puts_iv_data) with strike and IV
        """
        calls, puts, underlying = self.get_options_chain(ticker, expiration)

        if calls.empty or puts.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Select relevant columns for IV smile
        calls_iv = calls[['strike', 'impliedVolatility', 'moneyness_pct']].copy()
        puts_iv = puts[['strike', 'impliedVolatility', 'moneyness_pct']].copy()

        # Sort by strike
        calls_iv = calls_iv.sort_values('strike')
        puts_iv = puts_iv.sort_values('strike')

        return calls_iv, puts_iv

    def get_high_volume_options(
        self,
        ticker: str,
        expiration: str = None,
        min_volume: int = 100,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get options with highest volume

        Args:
            ticker: Stock ticker
            expiration: Expiration date
            min_volume: Minimum volume threshold
            top_n: Number of top results to return

        Returns:
            DataFrame with high volume options
        """
        calls, puts, underlying = self.get_options_chain(ticker, expiration)

        # Combine calls and puts
        calls['type'] = 'call'
        puts['type'] = 'put'

        all_options = pd.concat([calls, puts], ignore_index=True)

        # Filter by volume
        high_vol = all_options[all_options['volume'] >= min_volume]

        # Sort by volume
        high_vol = high_vol.sort_values('volume', ascending=False)

        # Select relevant columns
        result = high_vol[[
            'type', 'strike', 'lastPrice', 'bid', 'ask',
            'volume', 'openInterest', 'impliedVolatility',
            'ITM', 'moneyness_pct'
        ]].head(top_n)

        result['underlying_price'] = underlying

        return result
