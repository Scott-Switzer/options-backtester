# options-backtester

Options pricing model library, implied-volatility surfaces, and strategy backtesting in Python.

## What it does

- **Model library** (`options_backtester/models.py`): eight pricing models with a unified interface — Black-Scholes, binomial (CRR), trinomial, Monte Carlo, Longstaff-Schwartz (American), Barone-Adesi-Whaley, Heston stochastic volatility, and Bjerksund-Stensland. Pure Python, standard library only.
- **Blended pricing** (`options_backtester/blender.py`): selects the best model for the option's characteristics (exercise style, time to expiry, path dependency, speed/accuracy tradeoff) and explains why. Optional weighted-blend mode averages multiple models.
- **Pricing engine** (`options_backtester/black_scholes.py`): European call/put prices and Greeks (delta, gamma, theta, vega, rho). Pure Python, no dependencies.
- **Implied volatility solver** (`options_backtester/implied_vol.py`): Newton-Raphson with bisection fallback; rejects prices that violate arbitrage bounds.
- **IV surface** (`options_backtester/surface.py`): builds a strike x expiry volatility grid from quotes and renders 3D surface and smile PNGs (matplotlib).
- **Backtesting engine** (`options_backtester/backtest.py`): runs strategies against daily option-chain snapshots. Orders cross the spread (buy at ask, sell at bid), pay per-contract commission, settle to intrinsic at expiry, and support delta hedging with the underlying.
- **Example strategies** (`options_backtester/strategies.py`): long ATM call, delta-hedged short straddle, put spread.
- **Synthetic data** (`options_backtester/data.py`): geometric-Brownian-motion underlying paths and option chains priced off a parametric vol smile. Everything synthetic is labeled as synthetic in code, CLI output, and results. This is not real market data and is for testing the machinery only.
- **Cloudflare Worker** (`worker/`): HTTP API exposing the pricing models. POST /price for single-model pricing, POST /blend for smart model selection.

## Install

```bash
pip install -r requirements.txt
```

## CLI

```bash
# Price an option and show Greeks
python -m options_backtester.cli price --S 100 --K 100 --T 0.5 --sigma 0.25 --type call

# Solve for implied volatility
python -m options_backtester.cli iv --price 10.45 --S 100 --K 100 --T 1 --type call

# Plot a synthetic IV surface
python -m options_backtester.cli surface --out iv_surface.png

# Plot a synthetic IV smile for the 30-DTE expiry
python -m options_backtester.cli smile --dte 30 --out iv_smile.png

# Backtest a strategy on synthetic data
python -m options_backtester.cli backtest --strategy short_straddle_delta_hedged --days 60
```

## Tests

```bash
python -m unittest discover -s tests
```

28 tests. Pricing is checked against textbook Black-Scholes values and put-call parity; the IV solver is round-trip tested across strikes, expiries, and both option types; the backtest engine is tested for trade execution, commission drag, settlement, and metric sanity.

## Notes

- Prices are per-share; multiply by 100 for a standard US equity contract. The backtester applies the x100 multiplier internally.
- Theta is per calendar day. Vega is per 1.00 vol point (divide by 100 for a 1 vol-point move). Rho is per 1.00 rate move.
- European exercise only. No dividends.

## License

MIT. See LICENSE.
