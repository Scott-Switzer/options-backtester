"""Backtesting engine for options strategies.

Works on daily option-chain snapshots (see ``data.generate_chain_history``).
Orders cross the spread (buy at ask, sell at bid) and pay a per-contract
commission. Positions settle to intrinsic value at expiry. Greeks-based
strategies can delta-hedge with the underlying; the hedge itself pays no
commission (stock commission is assumed zero) but is tracked separately.

All P&L is per-share x 100 (standard US contract multiplier).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .black_scholes import black_scholes_price, delta

CONTRACT_MULT = 100.0


@dataclass
class Position:
    strike: float
    expiry_day: int  # absolute day index on which this option expires
    option_type: str  # "call" or "put"
    qty: int  # contracts, signed
    entry_mid: float = 0.0


@dataclass
class BacktestResult:
    strategy_name: str
    synthetic: bool
    start_cash: float
    end_value: float
    total_return: float
    sharpe: float
    max_drawdown: float
    num_trades: int
    equity_curve: list[float] = field(default_factory=list)

    def summary(self) -> str:
        tag = "SYNTHETIC DATA" if self.synthetic else "DATA"
        lines = [
            f"Strategy: {self.strategy_name} [{tag}]",
            f"Start cash:      ${self.start_cash:,.2f}",
            f"End value:       ${self.end_value:,.2f}",
            f"Total return:    {self.total_return * 100:+.2f}%",
            f"Sharpe (daily):  {self.sharpe:+.3f}",
            f"Max drawdown:    {self.max_drawdown * 100:.2f}%",
            f"Trades:          {self.num_trades}",
        ]
        return "\n".join(lines)


def _find_quote(chain: list[dict], strike: float, dte: int, option_type: str) -> dict | None:
    for q in chain:
        if q["strike"] == strike and q["dte"] == dte and q["type"] == option_type:
            return q
    return None


def _intrinsic(S: float, K: float, option_type: str) -> float:
    if option_type == "call":
        return max(S - K, 0.0)
    return max(K - S, 0.0)


class BacktestEngine:
    """Run a strategy over daily chain snapshots."""

    def __init__(
        self,
        history: list[dict],
        cash: float = 100_000.0,
        commission_per_contract: float = 0.65,
        r: float = 0.04,
    ):
        if not history:
            raise ValueError("history is empty")
        self.history = history
        self.start_cash = cash
        self.cash = cash
        self.commission = commission_per_contract
        self.r = r
        self.positions: list[Position] = []
        self.hedge_shares = 0.0  # delta hedge in underlying shares
        self.num_trades = 0
        self.equity_curve: list[float] = []
        self.synthetic = bool(history[0].get("synthetic", False))

    # -- order execution -------------------------------------------------
    def _execute(self, quote: dict, qty_contracts: int, day: int, expiry_day: int) -> None:
        """Execute at the touch: buy at ask, sell at bid, plus commission."""
        if qty_contracts == 0:
            return
        price = quote["ask"] if qty_contracts > 0 else quote["bid"]
        cost = qty_contracts * CONTRACT_MULT * price
        fee = abs(qty_contracts) * self.commission
        self.cash -= cost + fee
        self.positions.append(
            Position(
                strike=quote["strike"],
                expiry_day=expiry_day,
                option_type=quote["type"],
                qty=qty_contracts,
                entry_mid=quote["mid"],
            )
        )
        self.num_trades += 1

    def close_all(self, day: int, S: float, chain: list[dict]) -> None:
        """Close every open position at the touch."""
        for p in self.positions:
            dte = p.expiry_day - day
            if dte <= 0:
                continue
            q = _find_quote(chain, p.strike, dte, p.option_type)
            if q is None:
                continue
            price = q["bid"] if p.qty > 0 else q["ask"]
            proceeds = p.qty * CONTRACT_MULT * price
            fee = abs(p.qty) * self.commission
            self.cash += proceeds - fee
            self.num_trades += 1
        self.positions = []

    # -- valuation --------------------------------------------------------
    def portfolio_value(self, day: int, S: float, chain: list[dict]) -> float:
        value = self.cash + self.hedge_shares * S
        for p in self.positions:
            dte = p.expiry_day - day
            if dte <= 0:
                mark = _intrinsic(S, p.strike, p.option_type)
            else:
                q = _find_quote(chain, p.strike, dte, p.option_type)
                if q is None:
                    continue
                mark = q["mid"]
            value += p.qty * CONTRACT_MULT * mark
        return value

    def _settle_expired(self, day: int, S: float) -> None:
        """Cash-settle positions expiring today at intrinsic value."""
        remaining = []
        for p in self.positions:
            if p.expiry_day <= day:
                self.cash += p.qty * CONTRACT_MULT * _intrinsic(S, p.strike, p.option_type)
            else:
                remaining.append(p)
        self.positions = remaining

    def portfolio_delta(self, day: int, S: float, chain: list[dict]) -> float:
        """Total delta in shares (options delta x 100 x qty + hedge shares)."""
        total = self.hedge_shares
        for p in self.positions:
            dte = p.expiry_day - day
            if dte <= 0:
                continue
            q = _find_quote(chain, p.strike, dte, p.option_type)
            if q is None:
                continue
            d = delta(S, p.strike, dte / 365.0, self.r, q["iv"], p.option_type)
            total += d * CONTRACT_MULT * p.qty
        return total

    def hedge_to_delta(self, target_delta: float, day: int, S: float, chain: list[dict]) -> None:
        """Trade underlying shares so total portfolio delta equals target."""
        current = self.portfolio_delta(day, S, chain)
        trade_shares = target_delta - current
        self.hedge_shares += trade_shares
        self.cash -= trade_shares * S

    # -- main loop ---------------------------------------------------------
    def run(self, strategy, strategy_name: str = "strategy") -> BacktestResult:
        """Run ``strategy(engine, day, S, chain)`` once per day; returns metrics."""
        n = len(self.history)
        for i, snap in enumerate(self.history):
            day = snap["day"]
            S = snap["underlying"]
            chain = snap["chain"]
            self._settle_expired(day, S)
            strategy(self, day, S, chain)
            self.equity_curve.append(self.portfolio_value(day, S, chain))

        end_value = self.equity_curve[-1]
        total_return = end_value / self.start_cash - 1.0
        rets = [
            self.equity_curve[i] / self.equity_curve[i - 1] - 1.0
            for i in range(1, len(self.equity_curve))
            if self.equity_curve[i - 1] != 0
        ]
        mean = sum(rets) / len(rets) if rets else 0.0
        var = sum((x - mean) ** 2 for x in rets) / len(rets) if rets else 0.0
        sharpe = (mean / math.sqrt(var)) if var > 0 else 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for v in self.equity_curve:
            peak = max(peak, v)
            if peak > 0:
                max_dd = max(max_dd, (peak - v) / peak)
        return BacktestResult(
            strategy_name=strategy_name,
            synthetic=self.synthetic,
            start_cash=self.start_cash,
            end_value=end_value,
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=max_dd,
            num_trades=self.num_trades,
            equity_curve=self.equity_curve,
        )
