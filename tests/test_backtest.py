"""Tests for the backtesting engine (on synthetic data)."""

import unittest

from options_backtester.data import generate_chain_history
from options_backtester.backtest import BacktestEngine
from options_backtester import strategies


def _history(days=35, seed=1):
    return generate_chain_history(days=days, seed=seed, expiries_days=(30, 60))


class TestBacktest(unittest.TestCase):
    def test_long_call_completes(self):
        engine = BacktestEngine(_history(), cash=100_000.0)
        result = engine.run(strategies.long_atm_call, "long_atm_call")
        self.assertEqual(result.num_trades, 1)
        self.assertEqual(len(result.equity_curve), 35)
        # Option expired and settled; no open positions remain.
        self.assertEqual(engine.positions, [])
        self.assertTrue(result.synthetic)

    def test_straddle_hedged_runs(self):
        engine = BacktestEngine(_history(), cash=100_000.0)
        result = engine.run(strategies.short_straddle_delta_hedged, "straddle")
        # 2 opening trades + daily hedge activity recorded via trades on open only.
        self.assertGreaterEqual(result.num_trades, 2)
        self.assertEqual(len(result.equity_curve), 35)
        self.assertGreater(result.end_value, 0)

    def test_put_spread_defined_risk(self):
        engine = BacktestEngine(_history(), cash=100_000.0)
        result = engine.run(strategies.put_spread_bullish, "put_spread")
        self.assertEqual(result.num_trades, 2)
        # Max loss on a put spread is bounded: end value cannot go negative
        # from this trade alone given the starting cash.
        self.assertGreater(result.end_value, 0)

    def test_commission_reduces_pnl(self):
        h = _history()
        cheap = BacktestEngine(h, commission_per_contract=0.0)
        pricey = BacktestEngine(h, commission_per_contract=5.0)
        r_cheap = cheap.run(strategies.long_atm_call, "c")
        r_pricey = pricey.run(strategies.long_atm_call, "c")
        self.assertLess(r_pricey.end_value, r_cheap.end_value)

    def test_metrics_sane(self):
        engine = BacktestEngine(_history())
        result = engine.run(strategies.short_straddle_delta_hedged, "s")
        self.assertGreaterEqual(result.max_drawdown, 0.0)
        self.assertLessEqual(result.max_drawdown, 1.0)
        self.assertTrue(-10.0 < result.sharpe < 10.0)

    def test_empty_history_raises(self):
        with self.assertRaises(ValueError):
            BacktestEngine([])


if __name__ == "__main__":
    unittest.main()
