"""SYNTHETIC data generators. Nothing in this module is real market data.

Underlying paths follow geometric Brownian motion and option quotes are
priced off a parametric volatility smile, so the output is useful for
testing the pricing and backtesting machinery but says nothing about any
real security. Every public function and the CLI label their output as
synthetic.
"""

from __future__ import annotations

import math
import random

from .black_scholes import black_scholes_price

SYNTHETIC_TAG = "SYNTHETIC"


def generate_underlying_path(
    S0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.25,
    days: int = 252,
    dt: float = 1 / 252,
    seed: int = 42,
) -> list[float]:
    """SYNTHETIC underlying price path via geometric Brownian motion."""
    rng = random.Random(seed)
    path = [S0]
    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    for _ in range(days):
        path.append(path[-1] * math.exp(drift + vol * rng.gauss(0.0, 1.0)))
    return path


def smile_vol(K: float, S: float, base_vol: float = 0.25, skew: float = 0.12) -> float:
    """SYNTHETIC parametric smile: higher vol for downside strikes.

    vol(K) = base_vol + skew * max(0, log(S / K)).
    """
    return base_vol + skew * max(0.0, math.log(S / K))


def generate_option_chain(
    S: float,
    expiries_days: list[int],
    strikes: list[float],
    r: float = 0.04,
    base_vol: float = 0.25,
    skew: float = 0.12,
    spread_bps: float = 20.0,
    seed: int = 7,
) -> list[dict]:
    """SYNTHETIC option chain: mid prices from Black-Scholes on the smile.

    Each quote carries a bid/ask built from ``spread_bps`` (basis points of
    the mid) plus a small deterministic jitter so bid < mid < ask. Every
    record is tagged ``synthetic: True``.
    """
    rng = random.Random(seed)
    chain: list[dict] = []
    for dte in expiries_days:
        T = dte / 365.0
        for K in strikes:
            for option_type in ("call", "put"):
                sigma = smile_vol(K, S, base_vol, skew)
                mid = black_scholes_price(S, K, T, r, sigma, option_type)
                jitter = 1.0 + rng.uniform(-0.002, 0.002)
                half_spread = mid * (spread_bps / 1e4) / 2.0 * jitter
                chain.append(
                    {
                        "synthetic": True,
                        "underlying": S,
                        "strike": K,
                        "dte": dte,
                        "T": T,
                        "type": option_type,
                        "iv": sigma,
                        "mid": mid,
                        "bid": max(mid - half_spread, 0.01),
                        "ask": mid + half_spread,
                    }
                )
    return chain


def generate_chain_history(
    days: int = 60,
    S0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.25,
    expiries_days: tuple[int, ...] = (30, 60, 90),
    moneyness: tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1),
    r: float = 0.04,
    seed: int = 42,
) -> list[dict]:
    """SYNTHETIC daily option-chain snapshots along one underlying path.

    Returns a list of ``{"day": i, "underlying": S, "chain": [...]}`` dicts.
    Strikes are set as moneyness multiples of the day's underlying price and
    each expiry rolls down one day per step (floored at 1 day).
    """
    path = generate_underlying_path(S0, mu, sigma, days, seed=seed)[:days]
    history: list[dict] = []
    for i, S in enumerate(path):
        strikes = [round(m * S, 2) for m in moneyness]
        dtes = [max(d - i, 1) for d in expiries_days]
        chain = generate_option_chain(S, dtes, strikes, r=r, seed=seed + i)
        history.append({"day": i, "underlying": S, "chain": chain, "synthetic": True})
    return history
