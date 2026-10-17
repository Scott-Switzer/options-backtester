"""Implied volatility solver: invert Black-Scholes on an observed price.

Uses Newton-Raphson with a bisection fallback, so it converges for any
arbitrage-free price. Raises ValueError if the price violates static
arbitrage bounds (below intrinsic value or above the no-arbitrage cap).
"""

from __future__ import annotations

import math

from .black_scholes import black_scholes_price, vega


def _arbitrage_bounds(
    S: float, K: float, T: float, r: float, option_type: str
) -> tuple[float, float]:
    disc = math.exp(-r * T)
    if option_type == "call":
        lower = max(S - K * disc, 0.0)
        upper = S
    else:
        lower = max(K * disc - S, 0.0)
        upper = K * disc
    return lower, upper


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Solve for the Black-Scholes implied volatility of an observed price."""
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if market_price < 0:
        raise ValueError("market_price cannot be negative")
    lower, upper = _arbitrage_bounds(S, K, T, r, option_type)
    if market_price < lower - 1e-12:
        raise ValueError(
            f"price {market_price:.6f} below intrinsic lower bound {lower:.6f}"
        )
    if market_price > upper + 1e-12:
        raise ValueError(f"price {market_price:.6f} above upper bound {upper:.6f}")

    # Newton-Raphson from a reasonable seed.
    sigma = 0.3
    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        v = vega(S, K, T, r, sigma)
        if v < 1e-12:
            break
        sigma -= diff / v
        if sigma <= 0:
            sigma = 0.05
            break

    # Bisection fallback: bracket [1e-6, 5.0] and narrow.
    lo, hi = 1e-6, 5.0
    f_lo = black_scholes_price(S, K, T, r, lo, option_type) - market_price
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = black_scholes_price(S, K, T, r, mid, option_type) - market_price
        if abs(f_mid) < tol:
            return mid
        if (f_lo < 0) == (f_mid < 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
