"""Tests for the statistics helpers Act 8 added.

Two of them carry weight in the report and had to be right:

  * ``bootstrap_mean_ci`` resamples ITEMS, not generations. A Wilson interval
    over 120 draws from 40 items would assume independence that does not hold
    and would be too narrow.
  * ``mcnemar_exact`` is the paired binary test for "did the answer change
    between two conditions on the same items", which is the actual question
    Act 8's headline contrasts ask.
"""

from __future__ import annotations

import unittest

from src.bench.stats import bootstrap_mean_ci, mcnemar_exact


class TestBootstrap(unittest.TestCase):
    def test_is_deterministic_given_the_seed(self):
        vals = [0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.25, 1.0]
        a = bootstrap_mean_ci(vals, seed=7, n_resamples=2000)
        b = bootstrap_mean_ci(vals, seed=7, n_resamples=2000)
        self.assertEqual(a["ci_low"], b["ci_low"])
        self.assertEqual(a["ci_high"], b["ci_high"])

    def test_different_seeds_give_different_intervals(self):
        vals = [0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.25, 1.0]
        a = bootstrap_mean_ci(vals, seed=1, n_resamples=2000)
        b = bootstrap_mean_ci(vals, seed=2, n_resamples=2000)
        self.assertNotEqual((a["ci_low"], a["ci_high"]), (b["ci_low"], b["ci_high"]))

    def test_interval_contains_the_mean(self):
        vals = [0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.25, 1.0]
        r = bootstrap_mean_ci(vals, seed=3)
        self.assertLessEqual(r["ci_low"], r["mean"])
        self.assertGreaterEqual(r["ci_high"], r["mean"])

    def test_constant_input_gives_a_degenerate_interval(self):
        r = bootstrap_mean_ci([1.0] * 10, seed=1)
        self.assertAlmostEqual(r["ci_low"], 1.0)
        self.assertAlmostEqual(r["ci_high"], 1.0)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            bootstrap_mean_ci([])

    def test_resampling_unit_is_the_item(self):
        """The CI must widen when the same item is repeated with different
        values -- i.e. the unit is the observation, not the row."""
        # 10 identical items, each a paired difference of 0.1
        paired = [0.1] * 10
        # 10 items, but 9 of them disagree in sign: far more variable
        mixed = [0.1] * 5 + [-0.1] * 5
        a = bootstrap_mean_ci(paired, seed=5)
        b = bootstrap_mean_ci(mixed, seed=5)
        self.assertLess(a["ci_high"] - a["ci_low"], b["ci_high"] - b["ci_low"])


class TestMcNemar(unittest.TestCase):
    def test_no_discordant_pairs_gives_p_one(self):
        r = mcnemar_exact(0, 0)
        self.assertEqual(r["p_value"], 1.0)
        self.assertEqual(r["discordant"], 0)

    def test_one_sided_split_is_strongly_significant(self):
        """9 discordant pairs all one way: p = 2 * 0.5^9 = 0.0039."""
        r = mcnemar_exact(9, 0)
        self.assertAlmostEqual(r["p_value"], 2 * (0.5 ** 9), places=9)
        self.assertLess(r["p_value"], 0.01)

    def test_balanced_split_is_not_significant(self):
        r = mcnemar_exact(5, 5)
        self.assertGreater(r["p_value"], 0.5)

    def test_matched_to_a_hand_computed_binomial(self):
        import math
        b, c = 3, 1
        n = b + c
        tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
        self.assertAlmostEqual(mcnemar_exact(b, c)["p_value"], 2 * tail, places=9)

    def test_rejects_negative_counts(self):
        with self.assertRaises(ValueError):
            mcnemar_exact(-1, 2)


class TestAct8FiguresExist(unittest.TestCase):
    """Figures are deliverables, so their absence must fail the suite."""

    def test_act8_figures_are_present(self):
        from pathlib import Path

        figs = Path("report/figures")
        for name in ("act8_hallucination_by_condition",
                     "act8_contrast_distribution",
                     "act7_icl_copy_vs_compute"):
            for ext in ("png", "pdf"):
                path = figs / f"{name}.{ext}"
                self.assertTrue(path.exists(), f"missing figure {path}")
                self.assertGreater(path.stat().st_size, 1000,
                                   f"figure {path} is suspiciously small")


if __name__ == "__main__":
    unittest.main()
