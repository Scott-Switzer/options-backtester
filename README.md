# options-backtester

Options pricing model library, implied-volatility surfaces, and strategy backtesting in Python.

## What it does

- **Model library** (`options_backtester/models.py`): eight pricing models with a unified interface: Black-Scholes, binomial (CRR), trinomial, Monte Carlo, Longstaff-Schwartz (American), Barone-Adesi-Whaley, Heston stochastic volatility, and Bjerksund-Stensland. Pure Python, standard library only.
- **Blended pricing** (`options_backtester/blender.py`): selects the best model for the option's characteristics (exercise style, time to expiry, path dependency, speed/accuracy tradeoff) and explains why. Optional weighted-blend mode averages multiple models.
- **Pricing engine** (`options_backtester/black_scholes.py`): European call/put prices and Greeks (delta, gamma, theta, vega, rho). Pure Python, no dependencies.
- **Implied volatility solver** (`options_backtester/implied_vol.py`): Newton-Raphson with bisection fallback; rejects prices that violate arbitrage bounds.
- **IV surface** (`options_backtester/surface.py`): builds a strike x expiry volatility grid from quotes and renders 3D surface and smile PNGs (matplotlib).
- **Backtesting engine** (`options_backtester/backtest.py`): runs strategies against daily option-chain snapshots. Orders cross the spread (buy at ask, sell at bid), pay per-contract commission, settle to intrinsic at expiry, and support delta hedging with the underlying.
- **Example strategies** (`options_backtester/strategies.py`): long ATM call, delta-hedged short straddle, put spread.
- **Synthetic data** (`options_backtester/data.py`): geometric-Brownian-motion underlying paths and option chains priced off a parametric vol smile. Everything synthetic is labeled as synthetic in code, CLI output, and results. This is not real market data and is for testing the machinery only.
- **Served price history** (`options_backtester/serving.py`): a standard-library client for the Zion thin serving layer's `/history` contract. Returns typed daily price rows (raw observed plus split/distribution adjusted bases) with the security's permno and the source release ID, following the contract's pagination. The HTTP transport is injectable for testing.
- **Cloudflare Worker** (`worker/`): HTTP API exposing the pricing models. POST /price for single-model pricing, POST /blend for smart model selection, POST /price-by-ticker for pricing off served underlying history (see below).

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

## Pricing from served market data

`POST /price-by-ticker` prices an option using the underlying's real served price history instead of a caller-supplied spot. It reads history from the Zion thin serving layer (Worker `zion-financial-serving-v2-thin-staging`) through a service binding and uses the close on or before the requested date as the spot.

```json
{"ticker": "AAPL", "date": "2025-09-26", "strike": 250, "years_to_expiry": 0.5, "sigma": 0.25, "type": "call", "model": "black_scholes"}
```

- `date` is optional and defaults to the latest served session. `expiry` (YYYY-MM-DD) may replace `years_to_expiry`. `r` defaults to 0.05 and `q` to 0.
- `model` is optional; without it the endpoint uses the same model-selection behavior as POST /blend. With it, one of `black_scholes`, `binomial`, `baw`.
- `sigma` is optional. Without it, the endpoint computes 30-day realized volatility from the served history (log returns, annualized) and labels it `realized_vol_from_served_history` in the response.
- The response carries `provenance`: the source provider, the serving release ID (`input_sha256`), the permno, the price date, the spot used, and which price basis it came from.

Setup (see `worker/wrangler.toml`):

- The binding `ZION_FINANCIAL_QUERY` to `zion-financial-serving-v2-thin-staging` is declared in the wrangler config.
- The token is a Worker secret, never a config value: `wrangler secret put ZION_SERVING_TOKEN`. Use the same token the DCF security funnel uses for the serving layer.
- The route fails closed: without the binding or the secret it returns HTTP 503 `SERVING_NOT_CONFIGURED` and fetches nothing.

Status: the wiring is in code and covered by tests, but it is not live. It goes live only when the secret is set on the options-backtester Worker and the Worker is deployed. Nothing in this repository deploys automatically.

## Tests

```bash
python -m unittest discover -s tests
cd worker && npm test
```

70 Python tests and 9 Worker tests. Pricing is checked against textbook Black-Scholes values and put-call parity; the IV solver is round-trip tested across strikes, expiries, and both option types; the backtest engine is tested for trade execution, commission drag, settlement, and metric sanity. The serving client is tested against a fake transport (pagination, row parsing, release propagation, error handling), and the Worker tests fake the serving binding, including the fail-closed 503 path and parity between /price-by-ticker and /price for the same inputs.

## Notes

- Prices are per-share; multiply by 100 for a standard US equity contract. The backtester applies the x100 multiplier internally.
- Theta is per calendar day. Vega is per 1.00 vol point (divide by 100 for a 1 vol-point move). Rho is per 1.00 rate move.
- European exercise only. No dividends.

## License

MIT. See LICENSE.
