"""Tests for Act 10: the ethics instruments and the reproducibility script."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


class TestDetectionThreshold(unittest.TestCase):
    def test_threshold_is_derived_from_n(self):
        from src.eval.act10_ethics import detection_threshold

        small = detection_threshold(10)
        large = detection_threshold(400)
        self.assertGreater(small["half_width_95"], large["half_width_95"])
        self.assertAlmostEqual(small["detectable_rate_above"],
                               0.5 + small["half_width_95"], places=6)

    def test_threshold_is_symmetric_about_chance(self):
        from src.eval.act10_ethics import detection_threshold

        t = detection_threshold(25)
        self.assertAlmostEqual(t["detectable_rate_above"] + t["detectable_rate_below"],
                               2 * t["chance"], places=6)


class TestToxicityLexicon(unittest.TestCase):
    def test_lexicon_matching_is_word_bounded(self):
        from src.eval.act10_ethics import _lexicon_rate

        self.assertEqual(_lexicon_rate("you are stupid"), 1)
        self.assertEqual(_lexicon_rate("stupidity is a state of mind"), 0,
                         "'stupid' must not match inside 'stupidity'")
        self.assertEqual(_lexicon_rate("a clean sentence"), 0)


class TestEthicsResultsExist(unittest.TestCase):
    def test_results_are_present_and_internally_consistent(self):
        p = Path("results/v2/act10/ethics.json")
        if not p.exists():
            self.skipTest("Act 10c results not present")
        d = json.loads(p.read_text())
        s = d["stereotype_association"]
        t = d["toxicity_lexicon"]
        c = d["calibration"]
        # The three families must be reported together: a rate without its
        # negative control cannot be read.
        for key in ("associated_first_occupation_rate",
                    "associated_second_occupation_rate",
                    "negative_control_rate"):
            self.assertIn(key, s)
            self.assertIsNotNone(s[key])
        self.assertIn("detection_threshold_associated", s)
        self.assertIn("wilson95", t)
        self.assertLessEqual(t["rate"], 1.0)
        self.assertGreaterEqual(c["expected_calibration_error"], 0.0)
        self.assertLess(c["expected_calibration_error"], 1.0)
        # The SFT comparison must be reported as unavailable, not silently absent.
        self.assertIn("sft_comparison", d)
        self.assertEqual(d["sft_comparison"]["status"], "not possible")


class TestReproduceScript(unittest.TestCase):
    """The reproducibility script is the project's closing artefact, so its
    shape is pinned here even though running it is expensive."""

    def test_every_declared_check_is_callable(self):
        from tools.reproduce import CHECKS

        self.assertGreaterEqual(len(CHECKS), 5)
        for key, fn in CHECKS.items():
            self.assertTrue(callable(fn), key)

    def test_the_centrepiece_and_the_mechanism_are_both_covered(self):
        """At least one Act 4 fit and one Act 9 number must be re-derived."""
        from tools.reproduce import CHECKS

        self.assertIn("act4", CHECKS)
        self.assertIn("act9", CHECKS)

    def test_checks_reference_the_tracked_artefacts(self):
        from tools.reproduce import CHECKS

        for key, fn in CHECKS.items():
            check = fn.__doc__ or ""
            self.assertTrue(check, f"{key} has no docstring explaining what it checks")


if __name__ == "__main__":
    unittest.main()
