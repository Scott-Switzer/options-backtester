"""Command-line interface for the options backtester."""

from __future__ import annotations

import argparse
import json
import sys

from .black_scholes import greeks
from .implied_vol import implied_volatility
from .data import generate_option_chain, generate_chain_history, SYNTHETIC_TAG
from .surface import build_surface, plot_surface, plot_smile
from .backtest import BacktestEngine
from . import strategies


def _fmt_greeks(g: dict) -> str:
    return "\n".join(
        [
            f"price: {g['price']:.4f}",
            f"delta: {g['delta']:+.4f}",
            f"gamma: {g['gamma']:+.6f}",
            f"theta: {g['theta']:+.4f} /day",
            f"vega:  {g['vega']:+.4f} /1.00 vol",
            f"rho:   {g['rho']:+.4f} /1.00 rate",
        ]
    )


def cmd_price(args: argparse.Namespace) -> int:
    g = greeks(args.S, args.K, args.T, args.r, args.sigma, args.type)
    print(_fmt_greeks(g))
    return 0


def cmd_iv(args: argparse.Namespace) -> int:
    iv = implied_volatility(args.price, args.S, args.K, args.T, args.r, args.type)
    print(f"implied volatility: {iv:.6f}")
    return 0


def cmd_surface(args: argparse.Namespace) -> int:
    print(f"[{SYNTHETIC_TAG}] generating synthetic chain...")
    chain = generate_option_chain(
        S=args.S,
        expiries_days=[int(x) for x in args.expiries.split(",")],
        strikes=[args.S * m for m in (0.9, 0.95, 1.0, 1.05, 1.1)],
        r=args.r,
    )
    surface = build_surface(chain, args.S, args.r)
    plot_surface(surface, args.out, title="Implied Volatility Surface (synthetic)")
    print(f"surface written to {args.out}")
    return 0


def cmd_smile(args: argparse.Namespace) -> int:
    print(f"[{SYNTHETIC_TAG}] generating synthetic chain...")
    chain = generate_option_chain(
        S=args.S,
        expiries_days=[args.dte],
        strikes=[args.S * m for m in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)],
        r=args.r,
    )
    plot_smile(chain, args.dte, args.out)
    print(f"smile written to {args.out}")
    return 0


STRATEGIES = {
    "long_atm_call": strategies.long_atm_call,
    "short_straddle_delta_hedged": strategies.short_straddle_delta_hedged,
    "put_spread_bullish": strategies.put_spread_bullish,
}


def cmd_backtest(args: argparse.Namespace) -> int:
    print(f"[{SYNTHETIC_TAG}] generating {args.days} days of synthetic chain history...")
    history = generate_chain_history(
        days=args.days, S0=args.S, seed=args.seed, expiries_days=(30, 60, 90)
    )
    engine = BacktestEngine(history, cash=args.cash, commission_per_contract=args.commission)
    strat = STRATEGIES[args.strategy]
    result = engine.run(strat, strategy_name=args.strategy)
    print(result.summary())
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "strategy": result.strategy_name,
                    "synthetic": result.synthetic,
                    "start_cash": result.start_cash,
                    "end_value": result.end_value,
                    "total_return": result.total_return,
                    "sharpe": result.sharpe,
                    "max_drawdown": result.max_drawdown,
                    "num_trades": result.num_trades,
                },
                f,
                indent=2,
            )
        print(f"metrics written to {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="options-backtester",
        description="Black-Scholes pricing, IV surfaces, and options backtesting. "
        "Synthetic data is always labeled as synthetic.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("price", help="Price an option and show Greeks.")
    pp.add_argument("--S", type=float, required=True, help="Underlying price")
    pp.add_argument("--K", type=float, required=True, help="Strike")
    pp.add_argument("--T", type=float, required=True, help="Time to expiry in years")
    pp.add_argument("--r", type=float, default=0.04, help="Risk-free rate")
    pp.add_argument("--sigma", type=float, required=True, help="Volatility")
    pp.add_argument("--type", choices=("call", "put"), default="call")
    pp.set_defaults(func=cmd_price)

    pi = sub.add_parser("iv", help="Solve for implied volatility from a price.")
    pi.add_argument("--price", type=float, required=True)
    pi.add_argument("--S", type=float, required=True)
    pi.add_argument("--K", type=float, required=True)
    pi.add_argument("--T", type=float, required=True)
    pi.add_argument("--r", type=float, default=0.04)
    pi.add_argument("--type", choices=("call", "put"), default="call")
    pi.set_defaults(func=cmd_iv)

    ps = sub.add_parser("surface", help="Plot a synthetic IV surface PNG.")
    ps.add_argument("--S", type=float, default=100.0)
    ps.add_argument("--r", type=float, default=0.04)
    ps.add_argument("--expiries", default="30,60,90")
    ps.add_argument("--out", default="iv_surface.png")
    ps.set_defaults(func=cmd_surface)

    pm = sub.add_parser("smile", help="Plot a synthetic IV smile PNG for one expiry.")
    pm.add_argument("--S", type=float, default=100.0)
    pm.add_argument("--r", type=float, default=0.04)
    pm.add_argument("--dte", type=int, default=30)
    pm.add_argument("--out", default="iv_smile.png")
    pm.set_defaults(func=cmd_smile)

    pb = sub.add_parser("backtest", help="Backtest a strategy on synthetic data.")
    pb.add_argument("--strategy", choices=sorted(STRATEGIES), default="long_atm_call")
    pb.add_argument("--days", type=int, default=60)
    pb.add_argument("--S", type=float, default=100.0)
    pb.add_argument("--cash", type=float, default=100_000.0)
    pb.add_argument("--commission", type=float, default=0.65)
    pb.add_argument("--seed", type=int, default=42)
    pb.add_argument("--json", default=None, help="Write metrics JSON to this path")
    pb.set_defaults(func=cmd_backtest)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
