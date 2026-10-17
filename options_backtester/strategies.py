"""Ready-made example strategies for the backtest engine.

Each strategy is a function ``(engine, day, S, chain) -> None`` that places
orders through the engine. They are intentionally simple and fully
functional on the synthetic chain history from ``data``.
"""

from __future__ import annotations


def _atm_strike(chain: list[dict], S: float) -> float:
    return min({q["strike"] for q in chain}, key=lambda k: abs(k - S))


def _quote(chain: list[dict], strike: float, dte: int, option_type: str):
    for q in chain:
        if q["strike"] == strike and q["dte"] == dte and q["type"] == option_type:
            return q
    return None


def long_atm_call(engine, day: int, S: float, chain: list[dict]) -> None:
    """Buy one 30-DTE ATM call on day 0 and hold to expiry.

    Baseline sanity strategy: exactly one trade, settles at intrinsic.
    """
    if day != 0 or engine.positions:
        return
    K = _atm_strike(chain, S)
    q = _quote(chain, K, 30, "call")
    if q is None:
        return
    engine._execute(q, 1, day, expiry_day=day + 30)


def short_straddle_delta_hedged(engine, day: int, S: float, chain: list[dict]) -> None:
    """Sell a 30-DTE ATM straddle on day 0, delta-hedge daily, hold to expiry.

    Classic short-volatility strategy: collect premium, hedge directional
    exposure each day by trading the underlying toward zero net delta.
    """
    if day == 0 and not engine.positions:
        K = _atm_strike(chain, S)
        qc = _quote(chain, K, 30, "call")
        qp = _quote(chain, K, 30, "put")
        if qc is None or qp is None:
            return
        engine._execute(qc, -1, day, expiry_day=day + 30)
        engine._execute(qp, -1, day, expiry_day=day + 30)
    # Re-hedge to flat delta every day while the straddle is alive.
    if engine.positions:
        engine.hedge_to_delta(0.0, day, S, chain)


def put_spread_bullish(engine, day: int, S: float, chain: list[dict]) -> None:
    """Buy a 30-DTE ATM put spread (long ATM put, short 0.9x put) on day 0.

    Defined-risk bearish/bullish-neutral structure: profits if the
    underlying stays flat or drifts up, capped loss if it falls hard.
    """
    if day != 0 or engine.positions:
        return
    strikes = sorted({q["strike"] for q in chain})
    atm = _atm_strike(chain, S)
    lower = max([k for k in strikes if k < atm], default=None)
    if lower is None:
        return
    q_long = _quote(chain, atm, 30, "put")
    q_short = _quote(chain, lower, 30, "put")
    if q_long is None or q_short is None:
        return
    engine._execute(q_long, 1, day, expiry_day=day + 30)
    engine._execute(q_short, -1, day, expiry_day=day + 30)
