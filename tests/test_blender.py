"""Tests for the blended pricing engine."""

import unittest

from options_backtester import blender


class SelectModelTests(unittest.TestCase):
    def test_european_selects_black_scholes(self):
        name, _ = blender.select_model(style="european", T=0.5)
        self.assertEqual(name, "black_scholes")

    def test_american_balanced_selects_bs_model(self):
        name, _ = blender.select_model(style="american", T=0.5, speed=0.5)
        self.assertEqual(name, "bjerksund_stensland")

    def test_american_fast_selects_baw(self):
        name, _ = blender.select_model(style="american", T=0.5, speed=0.9)
        self.assertEqual(name, "baw")

    def test_american_accurate_selects_lsm(self):
        name, _ = blender.select_model(style="american", T=0.5, speed=0.1)
        self.assertEqual(name, "longstaff_schwartz")

    def test_short_dated_american_selects_binomial(self):
        name, _ = blender.select_model(style="american", T=0.05, speed=0.5)
        self.assertEqual(name, "binomial")

    def test_long_dated_european_selects_heston(self):
        name, _ = blender.select_model(style="european", T=2.0, speed=0.3)
        self.assertEqual(name, "heston")

    def test_path_dependent_selects_monte_carlo(self):
        name, _ = blender.select_model(style="european", T=1.0,
                                       path_dependent=True)
        self.assertEqual(name, "monte_carlo")

    def test_path_dependent_american_selects_lsm(self):
        name, _ = blender.select_model(style="american", T=1.0,
                                       path_dependent=True)
        self.assertEqual(name, "longstaff_schwartz")


class BlendedPriceTests(unittest.TestCase):
    def test_returns_price_model_reasoning(self):
        r = blender.blended_price(100, 100, 0.5, 0.05, 0.0, 0.2, "call")
        self.assertIn("price", r)
        self.assertIn("model", r)
        self.assertIn("reasoning", r)
        self.assertGreater(r["price"], 0)
        self.assertTrue(len(r["reasoning"]) > 20)

    def test_american_price_ge_european(self):
        # Use speed=0.9 so European leg uses Black-Scholes (not Heston),
        # keeping the comparison within constant-vol dynamics.
        eur = blender.blended_price(100, 100, 1.0, 0.05, 0.02, 0.2, "put",
                                    speed=0.9)
        am = blender.blended_price(100, 100, 1.0, 0.05, 0.02, 0.2, "put",
                                   style="american", speed=0.9)
        self.assertGreaterEqual(am["price"], eur["price"] - 1e-6)

    def test_weighted_blend(self):
        r = blender.blended_price(100, 100, 1.0, 0.05, 0.0, 0.2, "call",
                                  blend_mode=True, seed=42)
        self.assertEqual(r["model"], "weighted_blend")
        self.assertIsNotNone(r["components"])
        self.assertGreater(len(r["components"]), 1)
        # Blend should be near Black-Scholes (force BS with speed=0.9)
        bs = blender.blended_price(100, 100, 1.0, 0.05, 0.0, 0.2, "call",
                                   speed=0.9)
        self.assertLess(abs(r["price"] - bs["price"]), 0.5)

    def test_weighted_blend_american(self):
        r = blender.blended_price(100, 100, 1.0, 0.05, 0.02, 0.2, "put",
                                  style="american", blend_mode=True)
        self.assertEqual(r["model"], "weighted_blend")
        self.assertGreater(r["price"], 0)


if __name__ == "__main__":
    unittest.main()
