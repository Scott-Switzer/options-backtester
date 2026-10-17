"""Tests for synthetic data generators (properties, not values)."""

import unittest

from options_backtester.data import (
    generate_underlying_path,
    generate_option_chain,
    generate_chain_history,
    smile_vol,
    SYNTHETIC_TAG,
)


class TestSyntheticData(unittest.TestCase):
    def test_tagged_synthetic(self):
        self.assertEqual(SYNTHETIC_TAG, "SYNTHETIC")
        chain = generate_option_chain(100.0, [30], [90.0, 100.0, 110.0])
        for q in chain:
            self.assertTrue(q["synthetic"])

    def test_bid_ask_around_mid(self):
        chain = generate_option_chain(100.0, [30], [100.0])
        for q in chain:
            self.assertLess(q["bid"], q["mid"])
            self.assertLess(q["mid"], q["ask"])
            self.assertGreater(q["bid"], 0)

    def test_smile_downside_skew(self):
        # Downside strikes carry higher vol than upside strikes.
        self.assertGreater(smile_vol(90.0, 100.0), smile_vol(110.0, 100.0))

    def test_path_reproducible(self):
        p1 = generate_underlying_path(seed=99)
        p2 = generate_underlying_path(seed=99)
        self.assertEqual(p1, p2)

    def test_path_positive(self):
        for s in generate_underlying_path(days=500, seed=3):
            self.assertGreater(s, 0.0)

    def test_chain_history_shape(self):
        h = generate_chain_history(days=10, seed=5)
        self.assertEqual(len(h), 10)
        for snap in h:
            self.assertTrue(snap["synthetic"])
            self.assertGreater(len(snap["chain"]), 0)


if __name__ == "__main__":
    unittest.main()
