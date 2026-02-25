"""
Probability utilities for options strategies.
Risk-neutral lognormal integrals for PoP, EV, and touch probability.
"""

import numpy as np
from scipy.stats import norm


def probability_of_profit(S, strategy_builder, sigma, r, T, q=0.0, n_points=500) -> float:
    """P(payoff at expiry > 0) under risk-neutral lognormal distribution.

    Args:
        S: Current underlying price
        strategy_builder: StrategyBuilder instance with legs loaded
        sigma: Annualised implied volatility (decimal)
        r: Risk-free rate (decimal)
        T: Time to expiry in years
        q: Continuous dividend yield
        n_points: Integration grid size

    Returns:
        Probability as a float in [0, 1]
    """
    if T <= 0 or sigma <= 0:
        _, payoffs_now = strategy_builder.calculate_payoff(S, np.array([S]))
        return 1.0 if payoffs_now[0] > 0 else 0.0

    price_range = np.linspace(S * 0.50, S * 2.0, n_points)
    _, payoffs = strategy_builder.calculate_payoff(S, price_range)

    mu_ln = np.log(S) + (r - q - 0.5 * sigma ** 2) * T
    sigma_ln = sigma * np.sqrt(T)

    pdf = norm.pdf(np.log(price_range), mu_ln, sigma_ln) / price_range  # log-normal Jacobian
    total = np.trapz(pdf, price_range)
    if total > 0:
        pdf = pdf / total

    return float(np.clip(np.trapz(pdf * (payoffs > 0).astype(float), price_range), 0.0, 1.0))


def expected_value(S, strategy_builder, sigma, r, T, q=0.0, n_points=500) -> float:
    """Risk-neutral expected payoff at expiry (before discounting).

    Args:
        S: Current underlying price
        strategy_builder: StrategyBuilder instance with legs loaded
        sigma: Annualised implied volatility (decimal)
        r: Risk-free rate (decimal)
        T: Time to expiry in years
        q: Continuous dividend yield
        n_points: Integration grid size

    Returns:
        Expected dollar payoff
    """
    if T <= 0 or sigma <= 0:
        _, payoffs_now = strategy_builder.calculate_payoff(S, np.array([S]))
        return float(payoffs_now[0])

    price_range = np.linspace(S * 0.50, S * 2.0, n_points)
    _, payoffs = strategy_builder.calculate_payoff(S, price_range)

    mu_ln = np.log(S) + (r - q - 0.5 * sigma ** 2) * T
    sigma_ln = sigma * np.sqrt(T)

    pdf = norm.pdf(np.log(price_range), mu_ln, sigma_ln) / price_range
    total = np.trapz(pdf, price_range)
    if total > 0:
        pdf = pdf / total

    return float(np.trapz(pdf * payoffs, price_range))


def probability_of_touching(S, K, sigma, r, T, q=0.0) -> float:
    """P(underlying touches barrier K at any time before expiry).

    Uses the reflection principle with drift (exact for GBM).

    Args:
        S: Current underlying price
        K: Barrier (strike) price
        sigma: Annualised implied volatility
        r: Risk-free rate
        T: Time to expiry in years
        q: Continuous dividend yield

    Returns:
        Touch probability in [0, 1]
    """
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if K <= 0 or S <= 0:
        return 0.0

    mu_ln = r - q - 0.5 * sigma ** 2
    log_ratio = np.log(K / S)

    # Clamp sigma to avoid division by zero
    sigma_t = max(sigma * np.sqrt(T), 1e-10)

    d2 = (-log_ratio + mu_ln * T) / sigma_t
    drift_exp_arg = 2 * mu_ln * log_ratio / (sigma ** 2) if sigma > 0 else 0.0
    # Cap exponent to avoid overflow
    drift_exp = np.exp(np.clip(drift_exp_arg, -500, 500))

    touch = norm.cdf(-d2) + drift_exp * norm.cdf(-d2 - 2 * abs(log_ratio) / sigma_t)
    return float(np.clip(touch, 0.0, 1.0))
