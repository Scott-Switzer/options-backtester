"""Options backtester: Black-Scholes pricing, implied volatility surfaces,
and strategy backtesting with transaction costs.

All synthetic data is clearly labeled as synthetic. Nothing here is
presented as real market data.
"""

from .black_scholes import (
    black_scholes_price,
    delta,
    gamma,
    theta,
    vega,
    rho,
    greeks,
)
from .implied_vol import implied_volatility
from .backtest import BacktestEngine, BacktestResult
from .serving import HistoryResult, PriceRow, ServingClient, ServingError

__all__ = [
    "black_scholes_price",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "greeks",
    "implied_volatility",
    "BacktestEngine",
    "BacktestResult",
    "HistoryResult",
    "PriceRow",
    "ServingClient",
    "ServingError",
]

__version__ = "0.1.0"
