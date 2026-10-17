"""Tests for the options pricing model library."""

import math
import unittest

from options_backtester import models
from options_backtester.models import (
    black_scholes, binomial, trinomial, monte_carlo, longstaff_schwartz,
    barone_adesi_whaley, heston, bjerksund_stensland,
)


class BlackScholesTests(unittest.TestCase):
    def test_hull_textbook_values(self):
        # Hull: S=100, K=100, T=1, r=0.05, sigma=0.20
        c = black_scholes(100, 100, 1.0, 0.05, 0.0, 0.20, "call")
        p = black_scholes(100, 100, 1.0, 0.05, 0.0, 0.20, "put")
        self.assertAlmostEqual(c, 10.4506, places=3)
        self.assertAlmostEqual(p, 5.5735, places=3)

    def test_put_call_parity(self):
        S, K, T, r, q, sig = 100, 105, 0.5, 0.05, 0.02, 0.25
        c = black_scholes(S, K, T, r, q, sig, "call")
        p = black_scholes(S, K, T, r, q, sig, "put")
        fwd = S * math.exp(-q * T) - K * math.exp(-r * T)
        self.assertAlmostEqual(c - p, fwd, places=9)

    def test_greeks_signs(self):
        g = models.black_scholes_greeks(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
        self.assertGreater(g["delta"], 0)
        self.assertLess(g["delta"], 1)
        self.assertGreater(g["gamma"], 0)
        self.assertLess(g["theta"], 0)
        self.assertGreater(g["vega"], 0)


class BinomialTests(unittest.TestCase):
    def test_converges_to_black_scholes(self):
        bs = black_scholes(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
        for steps, tol in [(50, 0.15), (200, 0.05), (1000, 0.02)]:
            b = binomial(100, 100, 1.0, 0.05, 0.0, 0.2, "call", steps=steps)
            self.assertLess(abs(b - bs), tol,
                            f"steps={steps}: {b} vs {bs}")

    def test_american_ge_european(self):
        eur = binomial(100, 100, 1.0, 0.05, 0.02, 0.2, "put", steps=200)
        am = binomial(100, 100, 1.0, 0.05, 0.02, 0.2, "put", steps=200,
                      american=True)
        self.assertGreaterEqual(am, eur - 1e-9)


class TrinomialTests(unittest.TestCase):
    def test_converges_to_black_scholes(self):
        bs = black_scholes(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
        t = trinomial(100, 100, 1.0, 0.05, 0.0, 0.2, "call", steps=200)
        self.assertLess(abs(t - bs), 0.1)

    def test_american_ge_european(self):
        eur = trinomial(100, 100, 1.0, 0.05, 0.0, 0.2, "put", steps=100)
        am = trinomial(100, 100, 1.0, 0.05, 0.0, 0.2, "put", steps=100,
                       american=True)
        self.assertGreaterEqual(am, eur - 1e-9)


class MonteCarloTests(unittest.TestCase):
    def test_close_to_black_scholes(self):
        bs = black_scholes(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
        mc = monte_carlo(100, 100, 1.0, 0.05, 0.0, 0.2, "call",
                         paths=50000, seed=123)
        self.assertLess(abs(mc - bs), 0.15)

    def test_reproducible_with_seed(self):
        a = monte_carlo(100, 100, 1.0, 0.05, 0.0, 0.2, "call",
                        paths=5000, seed=7)
        b = monte_carlo(100, 100, 1.0, 0.05, 0.0, 0.2, "call",
                        paths=5000, seed=7)
        self.assertEqual(a, b)


class LongstaffSchwartzTests(unittest.TestCase):
    def test_american_put_exceeds_european(self):
        eur = black_scholes(90, 100, 1.0, 0.05, 0.0, 0.25, "put")
        lsm = longstaff_schwartz(90, 100, 1.0, 0.05, 0.0, 0.25, "put",
                                 paths=20000, exercise_dates=25, seed=42)
        self.assertGreater(lsm, eur)

    def test_close_to_baw(self):
        baw = barone_adesi_whaley(100, 100, 1.0, 0.05, 0.0, 0.2, "put")
        lsm = longstaff_schwartz(100, 100, 1.0, 0.05, 0.0, 0.2, "put",
                                 paths=30000, exercise_dates=25, seed=99)
        self.assertLess(abs(lsm - baw), 0.3)


class BAWTests(unittest.TestCase):
    def test_american_ge_european(self):
        for ot in ("call", "put"):
            for q in (0.0, 0.03):
                baw = barone_adesi_whaley(100, 100, 1.0, 0.05, q, 0.2, ot)
                eur = black_scholes(100, 100, 1.0, 0.05, q, 0.2, ot)
                self.assertGreaterEqual(baw, eur - 1e-6,
                                        f"{ot} q={q}: {baw} vs {eur}")

    def test_call_no_div_equals_european(self):
        baw = barone_adesi_whaley(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
        eur = black_scholes(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
        self.assertAlmostEqual(baw, eur, places=9)


class HestonTests(unittest.TestCase):
    def test_collapses_to_bs_when_volvol_tiny(self):
        bs = black_scholes(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
        h = heston(100, 100, 1.0, 0.05, 0.0, 0.2, "call", volvol=1e-9)
        self.assertAlmostEqual(h, bs, places=3)

    def test_put_call_parity(self):
        c = heston(100, 105, 0.5, 0.05, 0.01, 0.25, "call",
                   kappa=2.0, volvol=0.3, rho=-0.5)
        p = heston(100, 105, 0.5, 0.05, 0.01, 0.25, "put",
                   kappa=2.0, volvol=0.3, rho=-0.5)
        fwd = 100 * math.exp(-0.01 * 0.5) - 105 * math.exp(-0.05 * 0.5)
        self.assertAlmostEqual(c - p, fwd, places=2)

    def test_positive(self):
        for ot in ("call", "put"):
            h = heston(100, 100, 1.0, 0.05, 0.0, 0.2, ot)
            self.assertGreater(h, 0)


class BjerksundStenslandTests(unittest.TestCase):
    def test_american_ge_european(self):
        cases = [
            (100, 100, 1.0, 0.05, 0.02, 0.2, "call"),
            (100, 100, 1.0, 0.05, 0.0, 0.2, "put"),
            (90, 100, 0.5, 0.05, 0.01, 0.3, "put"),
        ]
        for S, K, T, r, q, sig, ot in cases:
            bs = bjerksund_stensland(S, K, T, r, q, sig, ot)
            eur = black_scholes(S, K, T, r, q, sig, ot)
            self.assertGreaterEqual(bs, eur - 1e-6, f"{ot}: {bs} vs {eur}")

    def test_close_to_baw(self):
        bs = bjerksund_stensland(100, 100, 1.0, 0.05, 0.02, 0.25, "put")
        baw = barone_adesi_whaley(100, 100, 1.0, 0.05, 0.02, 0.25, "put")
        self.assertLess(abs(bs - baw) / baw, 0.15)


class RegistryTests(unittest.TestCase):
    def test_all_models_present(self):
        for name in ["black_scholes", "binomial", "trinomial", "monte_carlo",
                     "longstaff_schwartz", "baw", "heston",
                     "bjerksund_stensland"]:
            self.assertIn(name, models.MODELS)

    def test_price_with_dispatch(self):
        for name in models.MODELS:
            kw = {}
            if name == "monte_carlo":
                kw = {"paths": 5000, "seed": 1}
            if name == "longstaff_schwartz":
                kw = {"paths": 5000, "exercise_dates": 10, "seed": 1}
            p = models.price_with(name, 100, 100, 0.5, 0.05, 0.0, 0.2,
                                  "call", **kw)
            self.assertGreater(p, 0, f"{name} returned {p}")


if __name__ == "__main__":
    unittest.main()
