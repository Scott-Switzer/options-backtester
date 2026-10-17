"""Black-Scholes pricing and Greeks for European options.

Pure Python, standard library only. Conventions:

- Prices are per-share (multiply by 100 for a standard US equity contract).
- ``option_type`` is ``"call"`` or ``"put"``.
- ``T`` is time to expiry in years (e.g. 30/365).
- ``r`` is the continuously compounded risk-free rate.
- ``sigma`` is annualized volatility.
- Theta is returned per calendar day (annual theta / 365).
"""

from __future__ import annotations

import math

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T, sigma must all be positive")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """European option price per share."""
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)
    if option_type == "call":
        return S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    return K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Rate of change of option price with respect to the underlying."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return _norm_cdf(d1)
    if option_type == "put":
        return _norm_cdf(d1) - 1.0
    raise ValueError("option_type must be 'call' or 'put'")


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Rate of change of delta with respect to the underlying (same for calls and puts)."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Time decay per calendar day (negative for long options)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)
    time_term = -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if option_type == "call":
        annual = time_term - r * K * disc * _norm_cdf(d2)
    elif option_type == "put":
        annual = time_term + r * K * disc * _norm_cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return annual / 365.0


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Sensitivity to a 1.0 (100 vol point) change in volatility; divide by 100 for 1 vol point."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T)


def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Sensitivity to a 1.0 (100bp x100) change in the risk-free rate; divide by 100 for 1bp x100."""
    _, d2 = _d1_d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)
    if option_type == "call":
        return K * T * disc * _norm_cdf(d2)
    if option_type == "put":
        return -K * T * disc * _norm_cdf(-d2)
    raise ValueError("option_type must be 'call' or 'put'")


def greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> dict[str, float]:
    """All Greeks plus price in one dict."""
    return {
        "price": black_scholes_price(S, K, T, r, sigma, option_type),
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option_type),
        "vega": vega(S, K, T, r, sigma),
        "rho": rho(S, K, T, r, sigma, option_type),
    }
