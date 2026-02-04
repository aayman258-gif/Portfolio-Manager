"""
Options Strategy Recommender
Regime-aware options recommendations
"""

import numpy as np
from typing import Dict, List


class OptionsRecommender:
    """Recommend options strategies based on market regime"""

    def __init__(self):
        # Strategy mappings for each regime
        self.regime_strategies = {
            'Low Vol': [
                {
                    'name': 'Iron Condor',
                    'type': 'Income/Theta',
                    'description': 'Sell OTM call and put spreads to collect premium in range-bound markets',
                    'structure': 'Short call spread + Short put spread',
                    'best_when': 'Low volatility, range-bound, expect price to stay within range',
                    'risk': 'Medium',
                    'complexity': 'Advanced'
                },
                {
                    'name': 'Cash-Secured Put',
                    'type': 'Income/Wheel',
                    'description': 'Sell OTM puts to generate income or acquire stock at discount',
                    'structure': 'Short OTM put with cash reserved',
                    'best_when': 'Want to own stock, low volatility, bullish to neutral',
                    'risk': 'Medium',
                    'complexity': 'Intermediate'
                },
                {
                    'name': 'Covered Call',
                    'type': 'Income/Enhancement',
                    'description': 'Sell calls against existing stock position',
                    'structure': 'Long 100 shares + Short OTM call',
                    'best_when': 'Own stock, neutral to slightly bullish, generate income',
                    'risk': 'Low',
                    'complexity': 'Beginner'
                }
            ],
            'High Vol': [
                {
                    'name': 'Long Straddle',
                    'type': 'Volatility Expansion',
                    'description': 'Buy ATM call and put to profit from large move in either direction',
                    'structure': 'Long ATM call + Long ATM put',
                    'best_when': 'Expect volatility explosion, uncertain direction, before events',
                    'risk': 'High (premium loss if no move)',
                    'complexity': 'Intermediate'
                },
                {
                    'name': 'Long Strangle',
                    'type': 'Volatility Expansion',
                    'description': 'Buy OTM call and put for cheaper volatility play',
                    'structure': 'Long OTM call + Long OTM put',
                    'best_when': 'Expect big move, lower cost than straddle',
                    'risk': 'High (premium loss)',
                    'complexity': 'Intermediate'
                },
                {
                    'name': 'Protective Put',
                    'type': 'Hedging/Insurance',
                    'description': 'Buy puts to hedge long stock position',
                    'structure': 'Long 100 shares + Long OTM put',
                    'best_when': 'Own stock, expect volatility, want downside protection',
                    'risk': 'Low (premium cost)',
                    'complexity': 'Beginner'
                }
            ],
            'Trending': [
                {
                    'name': 'Bull Call Spread',
                    'type': 'Directional/Bullish',
                    'description': 'Limited risk bullish play using call spread',
                    'structure': 'Long ITM/ATM call + Short OTM call',
                    'best_when': 'Strong uptrend, moderate bullish conviction',
                    'risk': 'Limited to debit paid',
                    'complexity': 'Intermediate'
                },
                {
                    'name': 'Bear Put Spread',
                    'type': 'Directional/Bearish',
                    'description': 'Limited risk bearish play using put spread',
                    'structure': 'Long ITM/ATM put + Short OTM put',
                    'best_when': 'Strong downtrend, moderate bearish conviction',
                    'risk': 'Limited to debit paid',
                    'complexity': 'Intermediate'
                },
                {
                    'name': 'Long Call/Put',
                    'type': 'Directional',
                    'description': 'Simple directional bet with defined risk',
                    'structure': 'Long call (bullish) or Long put (bearish)',
                    'best_when': 'Strong conviction on direction, willing to pay premium',
                    'risk': 'Limited to premium paid',
                    'complexity': 'Beginner'
                }
            ],
            'Mean Reversion': [
                {
                    'name': 'Short Strangle',
                    'type': 'Range-Bound',
                    'description': 'Sell OTM call and put, profit from range compression',
                    'structure': 'Short OTM call + Short OTM put',
                    'best_when': 'Elevated IV expected to contract, price to range',
                    'risk': 'Unlimited (manage with stops)',
                    'complexity': 'Advanced'
                },
                {
                    'name': 'Iron Butterfly',
                    'type': 'Range-Bound',
                    'description': 'Tight range bet with limited risk',
                    'structure': 'Sell ATM straddle + Buy OTM protective wings',
                    'best_when': 'Expect price to stay near current level',
                    'risk': 'Limited to wing width - credit',
                    'complexity': 'Advanced'
                },
                {
                    'name': 'Calendar Spread',
                    'type': 'Time Decay',
                    'description': 'Sell near-term, buy far-term to profit from time decay',
                    'structure': 'Short front-month + Long back-month (same strike)',
                    'best_when': 'Expect consolidation, then movement later',
                    'risk': 'Limited to debit paid',
                    'complexity': 'Advanced'
                }
            ]
        }

    def get_strategies_for_regime(self, regime: str) -> List[Dict]:
        """Get recommended strategies for a given regime"""
        return self.regime_strategies.get(regime, [])

    def calculate_strike_recommendations(
        self,
        current_price: float,
        regime: str,
        volatility: float,
        trend_direction: int = 0,
        days_to_expiry: int = 30
    ) -> Dict:
        """
        Calculate specific strike prices and expiry recommendations

        Args:
            current_price: Current underlying price
            regime: Current market regime
            volatility: Current realized volatility (annualized)
            trend_direction: 1 (up), -1 (down), 0 (sideways)
            days_to_expiry: Target days to expiration

        Returns:
            Dictionary with strike recommendations for each strategy
        """
        recommendations = {}

        # Standard deviations for strike selection
        one_std_move = current_price * volatility * np.sqrt(days_to_expiry/252)

        if regime == 'Low Vol':
            # Iron Condor strikes
            recommendations['Iron Condor'] = {
                'short_put_strike': round(current_price - 1.5 * one_std_move, 2),
                'long_put_strike': round(current_price - 2.0 * one_std_move, 2),
                'short_call_strike': round(current_price + 1.5 * one_std_move, 2),
                'long_call_strike': round(current_price + 2.0 * one_std_move, 2),
                'expiry_dte': days_to_expiry,
                'rationale': f'Sell ~1.5 SD strikes (${one_std_move:.2f} move), protect at 2 SD'
            }

            recommendations['Cash-Secured Put'] = {
                'strike': round(current_price * 0.95, 2),
                'expiry_dte': days_to_expiry,
                'rationale': 'Sell 5% OTM put, willing to own at this price'
            }

            recommendations['Covered Call'] = {
                'strike': round(current_price * 1.05, 2),
                'expiry_dte': days_to_expiry,
                'rationale': 'Sell 5% OTM call, collect premium while holding stock'
            }

        elif regime == 'High Vol':
            # Long Straddle
            recommendations['Long Straddle'] = {
                'strike': round(current_price, 2),
                'expiry_dte': min(45, days_to_expiry),
                'rationale': f'ATM strikes, expect ${one_std_move:.2f} move in {days_to_expiry} days'
            }

            # Long Strangle
            recommendations['Long Strangle'] = {
                'call_strike': round(current_price + 0.5 * one_std_move, 2),
                'put_strike': round(current_price - 0.5 * one_std_move, 2),
                'expiry_dte': min(45, days_to_expiry),
                'rationale': 'OTM strikes reduce cost, need larger move to profit'
            }

            # Protective Put
            recommendations['Protective Put'] = {
                'strike': round(current_price * 0.95, 2),
                'expiry_dte': 60,
                'rationale': '5% OTM protection, covers downside below this level'
            }

        elif regime == 'Trending':
            if trend_direction > 0:  # Uptrend
                recommendations['Bull Call Spread'] = {
                    'long_strike': round(current_price, 2),
                    'short_strike': round(current_price * 1.05, 2),
                    'expiry_dte': days_to_expiry,
                    'rationale': 'Buy ATM, sell 5% OTM, profit if continues up'
                }
                recommendations['Long Call'] = {
                    'strike': round(current_price * 1.02, 2),
                    'expiry_dte': days_to_expiry,
                    'rationale': 'Slightly OTM call for momentum play'
                }
            elif trend_direction < 0:  # Downtrend
                recommendations['Bear Put Spread'] = {
                    'long_strike': round(current_price, 2),
                    'short_strike': round(current_price * 0.95, 2),
                    'expiry_dte': days_to_expiry,
                    'rationale': 'Buy ATM, sell 5% OTM, profit if continues down'
                }
                recommendations['Long Put'] = {
                    'strike': round(current_price * 0.98, 2),
                    'expiry_dte': days_to_expiry,
                    'rationale': 'Slightly OTM put for downside momentum'
                }
            else:
                recommendations['Long Call'] = {
                    'strike': round(current_price, 2),
                    'expiry_dte': days_to_expiry,
                    'rationale': 'ATM - wait for trend confirmation'
                }

        elif regime == 'Mean Reversion':
            # Short Strangle
            recommendations['Short Strangle'] = {
                'call_strike': round(current_price + one_std_move, 2),
                'put_strike': round(current_price - one_std_move, 2),
                'expiry_dte': days_to_expiry,
                'rationale': f'Sell 1 SD strikes, expect ${one_std_move:.2f} range'
            }

            # Iron Butterfly
            recommendations['Iron Butterfly'] = {
                'short_strike': round(current_price, 2),
                'long_call_strike': round(current_price + one_std_move, 2),
                'long_put_strike': round(current_price - one_std_move, 2),
                'expiry_dte': days_to_expiry,
                'rationale': 'Tight range bet centered at current price'
            }

            # Calendar Spread
            recommendations['Calendar Spread'] = {
                'strike': round(current_price, 2),
                'short_expiry_dte': days_to_expiry,
                'long_expiry_dte': min(days_to_expiry * 2, 90),
                'rationale': 'Sell front-month theta, own back-month vega'
            }

        return recommendations

    def get_position_sizing_guide(
        self,
        regime: str,
        account_size: float,
        risk_per_trade: float = 0.02
    ) -> Dict:
        """Get position sizing for options based on regime"""

        regime_risk = {
            'Low Vol': {
                'max_risk_per_trade': min(risk_per_trade * 1.5, 0.03),  # Can risk more in calm markets
                'max_positions': 5,
                'notes': 'Low risk environment, can scale up premium selling'
            },
            'High Vol': {
                'max_risk_per_trade': risk_per_trade * 0.5,  # Reduce risk in volatile markets
                'max_positions': 3,
                'notes': 'High risk environment, reduce size and be selective'
            },
            'Trending': {
                'max_risk_per_trade': risk_per_trade,
                'max_positions': 4,
                'notes': 'Moderate risk, focus on directional plays'
            },
            'Mean Reversion': {
                'max_risk_per_trade': risk_per_trade,
                'max_positions': 4,
                'notes': 'Moderate risk, focus on range-bound strategies'
            }
        }

        sizing = regime_risk.get(regime, regime_risk['High Vol'])
        sizing['max_dollar_risk'] = account_size * sizing['max_risk_per_trade']

        return sizing
