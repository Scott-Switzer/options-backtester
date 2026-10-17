"""Tests for the implied volatility solver."""

import unittest

from options_backtester.black_scholes import black_scholes_price
from options_backtester.implied_vol import implied_volatility


class TestImpliedVol(unittest.TestCase):
    def test_round_trip_call(self):
        S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.04, 0.25
        price = black_scholes_price(S, K, T, r, sigma, "call")
        self.assertAlmostEqual(
            implied_volatility(price, S, K, T, r, "call"), sigma, places=6
        )

    def test_round_trip_put(self):
        S, K, T, r, sigma = 95.0, 100.0, 0.25, 0.03, 0.40
        price = black_scholes_price(S, K, T, r, sigma, "put")
        self.assertAlmostEqual(
            implied_volatility(price, S, K, T, r, "put"), sigma, places=6
        )

    def test_round_trip_grid(self):
        # Across strikes, expiries, and both types.
        for S, K in ((100, 90), (100, 100), (100, 110)):
            for T in (0.1, 0.5, 2.0):
                for typ in ("call", "put"):
                    for sigma in (0.15, 0.5):
                        price = black_scholes_price(S, K, T, 0.04, sigma, typ)
                        iv = implied_volatility(price, S, K, T, 0.04, typ)
                        self.assertAlmostEqual(iv, sigma, places=5,
                                               msg=f"{S},{K},{T},{typ},{sigma}")

    def test_below_intrinsic_raises(self):
        # A call priced below S - K*e^(-rT) is an arbitrage violation.
        with self.assertRaises(ValueError):
            implied_volatility(1.0, 150.0, 100.0, 1.0, 0.05, "call")

    def test_negative_price_raises(self):
        with self.assertRaises(ValueError):
            implied_volatility(-1.0, 100.0, 100.0, 1.0, 0.05, "call")


if __name__ == "__main__":
    unittest.main()
