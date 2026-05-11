"""
Options Data Loader
Fetches live options chains via Alpaca Options API (primary)
with automatic yfinance fallback when keys are unavailable.
"""

import pandas as pd
from typing import Dict, List, Tuple

from data import alpaca_options_client as _alpaca_opt


class OptionsDataLoader:
    """Load live options data — Alpaca primary, yfinance fallback."""

    def get_options_expirations(self, ticker: str) -> List[str]:
        """Return sorted list of available expiration dates (YYYY-MM-DD)."""
        return _alpaca_opt.get_option_expirations(ticker)

    def get_options_chain(
        self,
        ticker: str,
        expiration: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Get options chain for a specific expiration.

        Returns (calls_df, puts_df, underlying_price).
        DataFrames include: contractSymbol, strike, lastPrice, bid, ask,
        midPrice, volume, openInterest, impliedVolatility, expiration,
        moneyness, moneyness_pct, ITM, intrinsicValue, timeValue,
        bidAskSpread, bidAskSpreadPct, delta, gamma, theta, vega.
        """
        return _alpaca_opt.get_option_chain(ticker, expiration)

    def get_option_quote(
        self,
        ticker: str,
        strike: float,
        expiration: str,
        option_type: str,
    ) -> Dict:
        """
        Get real-time quote + greeks for a specific contract.

        Returns dict with bid, ask, mid, last, iv, delta, gamma, theta, vega.
        Falls back to nearest-strike lookup from the full chain if snapshot fails.
        """
        snap = _alpaca_opt.get_option_snapshot(ticker, strike, expiration, option_type)
        if snap:
            return snap

        # Fallback: pull from chain and find nearest strike
        calls, puts, underlying = self.get_options_chain(ticker, expiration)
        df = calls if option_type == "call" else puts
        if df.empty:
            return {}

        df = df.copy()
        df["_dist"] = (df["strike"] - strike).abs()
        row = df.nsmallest(1, "_dist").iloc[0]
        result = row.to_dict()
        result["underlying_price"] = underlying
        result["ticker"]           = ticker
        result["expiration"]       = expiration
        result["option_type"]      = option_type
        return result

    def get_atm_options(
        self,
        ticker: str,
        expiration: str = None,
        num_strikes: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Return ATM and near-ATM options (num_strikes closest to the spot).
        """
        calls, puts, underlying = self.get_options_chain(ticker, expiration)
        if calls.empty or puts.empty:
            return calls, puts, underlying

        calls = calls.copy()
        puts  = puts.copy()
        calls["distance_from_atm"] = (calls["strike"] - underlying).abs()
        puts["distance_from_atm"]  = (puts["strike"]  - underlying).abs()

        return (
            calls.nsmallest(num_strikes, "distance_from_atm"),
            puts.nsmallest(num_strikes,  "distance_from_atm"),
            underlying,
        )

    def get_options_summary(self, ticker: str) -> Dict:
        """High-level summary: expirations count, nearest chain stats, P/C ratio."""
        try:
            expirations = self.get_options_expirations(ticker)
            if not expirations:
                return {"ticker": ticker, "error": "No options available"}

            calls, puts, current_price = self.get_options_chain(ticker, expirations[0])

            summary = {
                "ticker":               ticker,
                "current_price":        current_price,
                "num_expirations":      len(expirations),
                "nearest_expiration":   expirations[0],
                "furthest_expiration":  expirations[-1],
                "num_call_strikes":     len(calls),
                "num_put_strikes":      len(puts),
                "total_call_volume":    int(calls["volume"].sum()) if not calls.empty else 0,
                "total_put_volume":     int(puts["volume"].sum())  if not puts.empty  else 0,
                "total_call_oi":        int(calls["openInterest"].sum()) if not calls.empty else 0,
                "total_put_oi":         int(puts["openInterest"].sum())  if not puts.empty  else 0,
            }
            cv = summary["total_call_volume"]
            pv = summary["total_put_volume"]
            summary["put_call_ratio"] = pv / cv if cv > 0 else 0
            return summary

        except Exception as exc:
            return {"ticker": ticker, "error": str(exc)}

    def calculate_implied_volatility_smile(
        self,
        ticker: str,
        expiration: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (calls_iv, puts_iv) DataFrames with strike, IV, moneyness_pct."""
        calls, puts, _ = self.get_options_chain(ticker, expiration)
        if calls.empty or puts.empty:
            return pd.DataFrame(), pd.DataFrame()

        cols = ["strike", "impliedVolatility", "moneyness_pct"]
        calls_iv = calls[cols].sort_values("strike").copy()
        puts_iv  = puts[cols].sort_values("strike").copy()
        return calls_iv, puts_iv

    def get_high_volume_options(
        self,
        ticker: str,
        expiration: str = None,
        min_volume: int = 100,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Top options by volume across calls and puts."""
        calls, puts, underlying = self.get_options_chain(ticker, expiration)
        if calls.empty and puts.empty:
            return pd.DataFrame()

        calls = calls.copy(); calls["type"] = "call"
        puts  = puts.copy();  puts["type"]  = "put"
        combined = pd.concat([calls, puts], ignore_index=True)

        keep_cols = [
            "type", "strike", "lastPrice", "bid", "ask",
            "volume", "openInterest", "impliedVolatility",
            "ITM", "moneyness_pct",
        ]
        keep_cols = [c for c in keep_cols if c in combined.columns]

        result = (
            combined[combined["volume"] >= min_volume]
            .sort_values("volume", ascending=False)
            [keep_cols]
            .head(top_n)
            .copy()
        )
        result["underlying_price"] = underlying
        return result
