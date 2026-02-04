"""
Multi-Leg Options Strategy Builder
Build and analyze complex options strategies
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from calculations.options_analytics import OptionsAnalytics


class StrategyBuilder:
    """Build and analyze multi-leg options strategies"""

    def __init__(self):
        self.options_calc = OptionsAnalytics()
        self.legs = []

        # Strategy templates
        self.strategy_templates = {
            'Iron Condor': {
                'legs': [
                    {'type': 'put', 'position': 'short', 'strike_offset': -0.05},  # Short put
                    {'type': 'put', 'position': 'long', 'strike_offset': -0.10},   # Long put
                    {'type': 'call', 'position': 'short', 'strike_offset': 0.05},  # Short call
                    {'type': 'call', 'position': 'long', 'strike_offset': 0.10}    # Long call
                ],
                'description': 'Sell OTM put spread + OTM call spread'
            },
            'Bull Call Spread': {
                'legs': [
                    {'type': 'call', 'position': 'long', 'strike_offset': 0.0},    # Long ATM call
                    {'type': 'call', 'position': 'short', 'strike_offset': 0.05}   # Short OTM call
                ],
                'description': 'Buy ATM call, sell OTM call'
            },
            'Bear Put Spread': {
                'legs': [
                    {'type': 'put', 'position': 'long', 'strike_offset': 0.0},     # Long ATM put
                    {'type': 'put', 'position': 'short', 'strike_offset': -0.05}   # Short OTM put
                ],
                'description': 'Buy ATM put, sell OTM put'
            },
            'Long Straddle': {
                'legs': [
                    {'type': 'call', 'position': 'long', 'strike_offset': 0.0},    # Long ATM call
                    {'type': 'put', 'position': 'long', 'strike_offset': 0.0}      # Long ATM put
                ],
                'description': 'Buy ATM call and put'
            },
            'Long Strangle': {
                'legs': [
                    {'type': 'call', 'position': 'long', 'strike_offset': 0.05},   # Long OTM call
                    {'type': 'put', 'position': 'long', 'strike_offset': -0.05}    # Long OTM put
                ],
                'description': 'Buy OTM call and put'
            },
            'Short Strangle': {
                'legs': [
                    {'type': 'call', 'position': 'short', 'strike_offset': 0.05},  # Short OTM call
                    {'type': 'put', 'position': 'short', 'strike_offset': -0.05}   # Short OTM put
                ],
                'description': 'Sell OTM call and put'
            },
            'Iron Butterfly': {
                'legs': [
                    {'type': 'call', 'position': 'short', 'strike_offset': 0.0},   # Short ATM call
                    {'type': 'put', 'position': 'short', 'strike_offset': 0.0},    # Short ATM put
                    {'type': 'call', 'position': 'long', 'strike_offset': 0.05},   # Long OTM call
                    {'type': 'put', 'position': 'long', 'strike_offset': -0.05}    # Long OTM put
                ],
                'description': 'Sell ATM straddle, buy OTM wings'
            },
            'Call Butterfly': {
                'legs': [
                    {'type': 'call', 'position': 'long', 'strike_offset': -0.03},  # Long lower
                    {'type': 'call', 'position': 'short', 'strike_offset': 0.0, 'quantity': 2},  # Short 2x ATM
                    {'type': 'call', 'position': 'long', 'strike_offset': 0.03}   # Long upper
                ],
                'description': 'Buy lower call, sell 2x ATM calls, buy upper call'
            }
        }

    def add_leg(
        self,
        option_type: str,
        strike: float,
        position: str,
        contracts: int,
        premium: float,
        implied_vol: float
    ) -> Dict:
        """
        Add a leg to the strategy

        Args:
            option_type: 'call' or 'put'
            strike: Strike price
            position: 'long' or 'short'
            contracts: Number of contracts
            premium: Option premium (price)
            implied_vol: Implied volatility

        Returns:
            Leg dictionary
        """
        leg = {
            'option_type': option_type,
            'strike': strike,
            'position': position,
            'contracts': contracts if position == 'long' else -contracts,
            'premium': premium,
            'implied_vol': implied_vol,
            'cost': premium * contracts * 100 * (1 if position == 'long' else -1)
        }

        self.legs.append(leg)
        return leg

    def remove_leg(self, index: int):
        """Remove a leg by index"""
        if 0 <= index < len(self.legs):
            self.legs.pop(index)

    def clear_legs(self):
        """Clear all legs"""
        self.legs = []

    def calculate_strategy_cost(self) -> float:
        """Calculate net debit/credit of strategy"""
        total_cost = sum(leg['cost'] for leg in self.legs)
        return total_cost

    def calculate_combined_greeks(
        self,
        underlying_price: float,
        time_to_expiry: float
    ) -> Dict:
        """
        Calculate combined Greeks for all legs

        Args:
            underlying_price: Current underlying price
            time_to_expiry: Time to expiration in years

        Returns:
            Dictionary with combined Greeks
        """
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0

        for leg in self.legs:
            greeks = self.options_calc.calculate_greeks(
                S=underlying_price,
                K=leg['strike'],
                T=time_to_expiry,
                sigma=leg['implied_vol'],
                option_type=leg['option_type']
            )

            # Multiply by contracts (already signed for long/short)
            contracts = leg['contracts']

            total_delta += greeks['delta'] * contracts * 100
            total_gamma += greeks['gamma'] * contracts * 100
            total_theta += greeks['theta'] * contracts * 100
            total_vega += greeks['vega'] * contracts * 100
            total_rho += greeks['rho'] * contracts * 100

        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega,
            'rho': total_rho
        }

    def calculate_payoff(
        self,
        underlying_price: float,
        price_range: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate P&L at expiration across price range

        Args:
            underlying_price: Current underlying price
            price_range: Array of prices to calculate P&L for

        Returns:
            (price_range, pnl_array)
        """
        if price_range is None:
            # Default range: ±30% of current price
            price_range = np.linspace(
                underlying_price * 0.7,
                underlying_price * 1.3,
                100
            )

        pnl = np.zeros_like(price_range)

        for leg in self.legs:
            contracts = leg['contracts']
            strike = leg['strike']
            premium = leg['premium']
            option_type = leg['option_type']

            # Calculate intrinsic value at each price
            if option_type == 'call':
                intrinsic = np.maximum(price_range - strike, 0)
            else:
                intrinsic = np.maximum(strike - price_range, 0)

            # P&L = (intrinsic value - premium paid) * contracts * 100
            leg_pnl = (intrinsic - premium) * contracts * 100

            pnl += leg_pnl

        return price_range, pnl

    def calculate_current_value(
        self,
        underlying_price: float,
        time_to_expiry: float
    ) -> Dict:
        """
        Calculate current theoretical value of the strategy

        Returns:
            Dictionary with strategy metrics
        """
        current_value = 0
        initial_cost = self.calculate_strategy_cost()

        for leg in self.legs:
            theoretical_price = self.options_calc.black_scholes(
                S=underlying_price,
                K=leg['strike'],
                T=time_to_expiry,
                sigma=leg['implied_vol'],
                option_type=leg['option_type']
            )

            # Current value of this leg
            leg_value = theoretical_price * leg['contracts'] * 100

            current_value += leg_value

        # P&L = current value - initial cost
        # For credit spreads (initial_cost < 0), profit when current_value > initial_cost
        pnl = current_value - initial_cost
        pnl_pct = (pnl / abs(initial_cost) * 100) if initial_cost != 0 else 0

        return {
            'initial_cost': initial_cost,
            'current_value': current_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }

    def calculate_breakeven_points(
        self,
        underlying_price: float
    ) -> List[float]:
        """
        Calculate breakeven price(s) at expiration

        Returns:
            List of breakeven prices
        """
        price_range, pnl = self.calculate_payoff(underlying_price)

        breakevens = []

        # Find where P&L crosses zero
        for i in range(len(pnl) - 1):
            if (pnl[i] <= 0 and pnl[i+1] > 0) or (pnl[i] >= 0 and pnl[i+1] < 0):
                # Linear interpolation to find exact breakeven
                x1, y1 = price_range[i], pnl[i]
                x2, y2 = price_range[i+1], pnl[i+1]

                if y2 != y1:
                    breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                    breakevens.append(breakeven)

        return breakevens

    def calculate_max_profit_loss(
        self,
        underlying_price: float
    ) -> Dict:
        """
        Calculate maximum profit and loss at expiration

        Returns:
            Dictionary with max profit, max loss, and prices where they occur
        """
        price_range, pnl = self.calculate_payoff(underlying_price)

        max_profit = np.max(pnl)
        max_loss = np.min(pnl)

        max_profit_price = price_range[np.argmax(pnl)]
        max_loss_price = price_range[np.argmin(pnl)]

        return {
            'max_profit': max_profit,
            'max_profit_price': max_profit_price,
            'max_loss': max_loss,
            'max_loss_price': max_loss_price
        }

    def get_strategy_summary(
        self,
        underlying_price: float,
        time_to_expiry: float
    ) -> Dict:
        """
        Get comprehensive strategy analysis

        Returns:
            Complete strategy summary
        """
        if not self.legs:
            return {'error': 'No legs in strategy'}

        # Cost
        cost = self.calculate_strategy_cost()

        # Greeks
        greeks = self.calculate_combined_greeks(underlying_price, time_to_expiry)

        # Current value
        value_metrics = self.calculate_current_value(underlying_price, time_to_expiry)

        # Max profit/loss
        max_metrics = self.calculate_max_profit_loss(underlying_price)

        # Breakevens
        breakevens = self.calculate_breakeven_points(underlying_price)

        # Strategy type
        if cost < 0:
            strategy_type = "Credit Strategy (Net Premium Collected)"
        elif cost > 0:
            strategy_type = "Debit Strategy (Net Premium Paid)"
        else:
            strategy_type = "Zero Cost Strategy"

        summary = {
            'num_legs': len(self.legs),
            'strategy_type': strategy_type,
            'initial_cost': cost,
            'current_value': value_metrics['current_value'],
            'pnl': value_metrics['pnl'],
            'pnl_pct': value_metrics['pnl_pct'],
            'max_profit': max_metrics['max_profit'],
            'max_loss': max_metrics['max_loss'],
            'max_profit_price': max_metrics['max_profit_price'],
            'max_loss_price': max_metrics['max_loss_price'],
            'breakevens': breakevens,
            'greeks': greeks
        }

        return summary

    def load_template(
        self,
        template_name: str,
        underlying_price: float,
        calls_df: pd.DataFrame,
        puts_df: pd.DataFrame
    ) -> bool:
        """
        Load a pre-built strategy template

        Args:
            template_name: Name of template
            underlying_price: Current underlying price
            calls_df: DataFrame with call options data
            puts_df: DataFrame with put options data

        Returns:
            True if successful
        """
        if template_name not in self.strategy_templates:
            return False

        template = self.strategy_templates[template_name]

        self.clear_legs()

        for leg_template in template['legs']:
            option_type = leg_template['type']
            position = leg_template['position']
            strike_offset = leg_template['strike_offset']
            quantity = leg_template.get('quantity', 1)

            # Calculate target strike
            target_strike = underlying_price * (1 + strike_offset)

            # Find closest strike in options data
            if option_type == 'call':
                df = calls_df
            else:
                df = puts_df

            if df.empty:
                return False

            # Find nearest strike
            df['strike_diff'] = abs(df['strike'] - target_strike)
            nearest = df.nsmallest(1, 'strike_diff').iloc[0]

            # Add leg
            self.add_leg(
                option_type=option_type,
                strike=nearest['strike'],
                position=position,
                contracts=quantity,
                premium=nearest.get('lastPrice', nearest.get('midPrice', 0)),
                implied_vol=nearest.get('impliedVolatility', 0.3)
            )

        return True

    def get_legs_dataframe(self) -> pd.DataFrame:
        """Get legs as formatted DataFrame"""
        if not self.legs:
            return pd.DataFrame()

        legs_data = []
        for i, leg in enumerate(self.legs):
            legs_data.append({
                '#': i + 1,
                'Type': leg['option_type'].upper(),
                'Strike': leg['strike'],
                'Position': leg['position'].upper(),
                'Contracts': abs(leg['contracts']),
                'Premium': leg['premium'],
                'IV': leg['implied_vol'],
                'Cost': leg['cost']
            })

        return pd.DataFrame(legs_data)
