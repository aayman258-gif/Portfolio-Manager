"""
Options Analytics
Greeks calculation and Black-Scholes pricing
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple
from datetime import datetime, timedelta


class OptionsAnalytics:
    """Calculate Greeks and fair value for options positions"""

    def __init__(self, risk_free_rate: float = 0.045):
        self.risk_free_rate = risk_free_rate

    def black_scholes(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = 'call',
        r: float = None
    ) -> float:
        """
        Black-Scholes option pricing

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            sigma: Implied volatility (annualized, as decimal)
            option_type: 'call' or 'put'
            r: Risk-free rate (uses instance default if None)

        Returns:
            Option theoretical price
        """
        if r is None:
            r = self.risk_free_rate

        if T <= 0:
            # At expiration, return intrinsic value
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = 'call',
        r: float = None
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option

        Returns:
            Dictionary with Delta, Gamma, Theta, Vega, Rho
        """
        if r is None:
            r = self.risk_free_rate

        if T <= 0:
            return {
                'delta': 1.0 if (option_type == 'call' and S > K) else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Theta
        if option_type == 'call':
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
            ) / 365  # Daily theta
        else:
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
            ) / 365  # Daily theta

        # Vega (same for call and put)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in IV

        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% change in r
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        greeks = {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

        return greeks

    def calculate_implied_volatility(
        self,
        S: float,
        K: float,
        T: float,
        market_price: float,
        option_type: str = 'call',
        r: float = None,
        max_iterations: int = 100,
        tolerance: float = 0.0001
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method

        Returns:
            Implied volatility (as decimal)
        """
        if r is None:
            r = self.risk_free_rate

        # Initial guess
        sigma = 0.3

        for i in range(max_iterations):
            # Calculate price and vega
            price = self.black_scholes(S, K, T, sigma, option_type, r)
            vega = S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)

            # Newton-Raphson update
            diff = market_price - price

            if abs(diff) < tolerance:
                return sigma

            if vega == 0:
                break

            sigma = sigma + diff / (vega * 100)  # vega is per 1% change

            # Keep sigma positive
            sigma = max(sigma, 0.01)

        return sigma

    def analyze_option_position(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str,
        contracts: int,
        premium_paid: float = None,
        current_price: float = None
    ) -> Dict:
        """
        Comprehensive analysis of an option position

        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiration (years)
            sigma: Implied volatility
            option_type: 'call' or 'put'
            contracts: Number of contracts (negative for short)
            premium_paid: Premium paid per contract (for P&L)
            current_price: Current option price (if available)

        Returns:
            Dictionary with all analytics
        """
        # Calculate theoretical value
        theoretical_value = self.black_scholes(S, K, T, sigma, option_type)

        # Calculate Greeks
        greeks = self.calculate_greeks(S, K, T, sigma, option_type)

        # Position Greeks (adjusted for number of contracts)
        position_greeks = {
            'delta': greeks['delta'] * contracts * 100,  # Per contract = 100 shares
            'gamma': greeks['gamma'] * contracts * 100,
            'theta': greeks['theta'] * contracts * 100,
            'vega': greeks['vega'] * contracts * 100,
            'rho': greeks['rho'] * contracts * 100
        }

        # Calculate P&L if premium paid is provided
        pnl = None
        pnl_pct = None
        if premium_paid is not None:
            option_price = current_price if current_price is not None else theoretical_value
            pnl = (option_price - premium_paid) * contracts * 100
            pnl_pct = ((option_price - premium_paid) / premium_paid * 100) if premium_paid > 0 else 0

        # Fair value analysis
        fair_value_analysis = None
        if current_price is not None:
            diff = current_price - theoretical_value
            diff_pct = (diff / theoretical_value * 100) if theoretical_value > 0 else 0

            if abs(diff_pct) < 5:
                rating = "Fair Value"
            elif diff_pct > 5:
                rating = "Overpriced" if contracts > 0 else "Good Short"
            else:
                rating = "Underpriced" if contracts > 0 else "Bad Short"

            fair_value_analysis = {
                'theoretical_value': theoretical_value,
                'market_price': current_price,
                'difference': diff,
                'difference_pct': diff_pct,
                'rating': rating
            }

        # Moneyness
        if option_type == 'call':
            moneyness = S - K
            moneyness_pct = (S / K - 1) * 100
            if S > K:
                status = f"ITM by ${moneyness:.2f} ({moneyness_pct:.1f}%)"
            elif S < K:
                status = f"OTM by ${-moneyness:.2f} ({-moneyness_pct:.1f}%)"
            else:
                status = "ATM"
        else:
            moneyness = K - S
            moneyness_pct = (K / S - 1) * 100
            if S < K:
                status = f"ITM by ${moneyness:.2f} ({moneyness_pct:.1f}%)"
            elif S > K:
                status = f"OTM by ${-moneyness:.2f} ({-moneyness_pct:.1f}%)"
            else:
                status = "ATM"

        result = {
            'theoretical_value': theoretical_value,
            'greeks': greeks,
            'position_greeks': position_greeks,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'fair_value_analysis': fair_value_analysis,
            'moneyness': status,
            'intrinsic_value': max(moneyness, 0),
            'time_value': theoretical_value - max(moneyness, 0),
            'days_to_expiry': int(T * 365)
        }

        return result

    def calculate_portfolio_greeks(self, positions: list) -> Dict:
        """
        Calculate total portfolio Greeks from multiple option positions

        Args:
            positions: List of position dictionaries with greeks

        Returns:
            Total portfolio Greeks
        """
        total_delta = sum(pos.get('delta', 0) for pos in positions)
        total_gamma = sum(pos.get('gamma', 0) for pos in positions)
        total_theta = sum(pos.get('theta', 0) for pos in positions)
        total_vega = sum(pos.get('vega', 0) for pos in positions)
        total_rho = sum(pos.get('rho', 0) for pos in positions)

        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega,
            'rho': total_rho
        }

    def get_greek_interpretation(self, greek_name: str, value: float) -> str:
        """Get interpretation of Greek value"""
        interpretations = {
            'delta': {
                'description': 'Price sensitivity to $1 move in underlying',
                'interpretation': f"Position moves ${abs(value):.2f} for $1 move in underlying. " +
                                ("Long bias" if value > 0 else "Short bias" if value < 0 else "Delta neutral")
            },
            'gamma': {
                'description': 'Rate of change of Delta',
                'interpretation': f"Delta changes by {abs(value):.4f} for $1 move. " +
                                ("High gamma = high sensitivity" if abs(value) > 0.1 else "Low gamma = stable delta")
            },
            'theta': {
                'description': 'Time decay per day',
                'interpretation': f"Position {'loses' if value < 0 else 'gains'} ${abs(value):.2f}/day from time decay. " +
                                ("Negative theta hurts long positions" if value < 0 else "Positive theta benefits short positions")
            },
            'vega': {
                'description': 'Sensitivity to 1% IV change',
                'interpretation': f"Position moves ${abs(value):.2f} for 1% change in IV. " +
                                ("Long vega = benefits from IV increase" if value > 0 else "Short vega = benefits from IV decrease")
            },
            'rho': {
                'description': 'Sensitivity to 1% interest rate change',
                'interpretation': f"Position moves ${abs(value):.2f} for 1% rate change (usually small impact)"
            }
        }

        return interpretations.get(greek_name, {'description': '', 'interpretation': ''})
