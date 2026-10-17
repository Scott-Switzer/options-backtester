"""Tests for Black-Scholes pricing and Greeks.

Reference values are the standard textbook case (Hull):
S=100, K=100, T=1, r=0.05, sigma=0.20 -> call 10.4506, put 5.5735.
"""

import math
import unittest

from options_backtester.black_scholes import (
    black_scholes_price,
    delta,
    gamma,
    theta,
    vega,
    rho,
    greeks,
)


class TestBlackScholes(unittest.TestCase):
    S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.20

    def test_call_price_textbook(self):
        self.assertAlmostEqual(
            black_scholes_price(self.S, self.K, self.T, self.R, self.SIG, "call"),
            10.4506,
            places=3,
        )

    def test_put_price_textbook(self):
        self.assertAlmostEqual(
            black_scholes_price(self.S, self.K, self.T, self.R, self.SIG, "put"),
            5.5735,
            places=3,
        )

    def test_put_call_parity(self):
        # C - P = S - K*e^(-rT), across a grid of inputs.
        for S in (80.0, 100.0, 120.0):
            for K in (90.0, 100.0, 110.0):
                for T in (0.25, 1.0):
                    c = black_scholes_price(S, K, T, self.R, self.SIG, "call")
                    p = black_scholes_price(S, K, T, self.R, self.SIG, "put")
                    self.assertAlmostEqual(
                        c - p, S - K * math.exp(-self.R * T), places=9
                    )

    def test_delta_bounds(self):
        for K in (80.0, 100.0, 120.0):
            dc = delta(self.S, K, self.T, self.R, self.SIG, "call")
            dp = delta(self.S, K, self.T, self.R, self.SIG, "put")
            self.assertGreater(dc, 0.0)
            self.assertLess(dc, 1.0)
            self.assertGreater(dp, -1.0)
            self.assertLess(dp, 0.0)
            # call delta - put delta == 1
            self.assertAlmostEqual(dc - dp, 1.0, places=9)

    def test_gamma_vega_positive(self):
        for K in (80.0, 100.0, 120.0):
            self.assertGreater(gamma(self.S, K, self.T, self.R, self.SIG), 0.0)
            self.assertGreater(vega(self.S, K, self.T, self.R, self.SIG), 0.0)

    def test_theta_negative_for_long(self):
        self.assertLess(theta(self.S, self.K, self.T, self.R, self.SIG, "call"), 0.0)
        self.assertLess(theta(self.S, self.K, self.T, self.R, self.SIG, "put"), 0.0)

    def test_rho_sign(self):
        self.assertGreater(rho(self.S, self.K, self.T, self.R, self.SIG, "call"), 0.0)
        self.assertLess(rho(self.S, self.K, self.T, self.R, self.SIG, "put"), 0.0)

    def test_deep_itm_call_near_intrinsic_plus_time_value(self):
        # Deep ITM call must exceed intrinsic value.
        S, K = 200.0, 100.0
        price = black_scholes_price(S, K, self.T, self.R, self.SIG, "call")
        self.assertGreater(price, S - K)

    def test_far_otm_call_near_zero(self):
        price = black_scholes_price(100.0, 500.0, self.T, self.R, self.SIG, "call")
        self.assertLess(price, 0.01)

    def test_greeks_dict_keys(self):
        g = greeks(self.S, self.K, self.T, self.R, self.SIG, "call")
        self.assertEqual(
            set(g.keys()), {"price", "delta", "gamma", "theta", "vega", "rho"}
        )

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            black_scholes_price(-100, self.K, self.T, self.R, self.SIG, "call")
        with self.assertRaises(ValueError):
            black_scholes_price(self.S, self.K, 0, self.R, self.SIG, "call")
        with self.assertRaises(ValueError):
            black_scholes_price(self.S, self.K, self.T, self.R, self.SIG, "straddle")


if __name__ == "__main__":
    unittest.main()
