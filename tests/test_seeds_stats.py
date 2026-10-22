from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class SeedHarnessTests(unittest.TestCase):
    def test_set_seed_reproducible(self):
        """set_seed makes torch RNG reproducible."""
        from src.bench.seeds import set_seed

        set_seed(42)
        a = torch.rand(10)
        set_seed(42)
        b = torch.rand(10)
        self.assertTrue(torch.equal(a, b))

    def test_verify_determinism_identical_configs(self):
        """Two identical configs produce byte-identical metrics.json."""
        from src.bench.experiment import ExperimentConfig
        from src.bench.seeds import verify_determinism

        with tempfile.TemporaryDirectory() as directory:
            config_a = ExperimentConfig(
                act="smoke",
                experiment="determinism",
                seed=42,
                dataset_path="data/tinyshakespeare.txt",
                steps=3,
                batch_size=4,
                context_length=16,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
                eval_interval=1,
                eval_batches=1,
                device="cpu",
                tokenizer="character",
                output_root=directory,
            )
            config_b = ExperimentConfig(
                act="smoke",
                experiment="determinism-2",
                seed=42,
                dataset_path="data/tinyshakespeare.txt",
                steps=3,
                batch_size=4,
                context_length=16,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
                eval_interval=1,
                eval_batches=1,
                device="cpu",
                tokenizer="character",
                output_root=directory,
            )
            identical, message = verify_determinism(config_a, config_b)
            self.assertTrue(identical, message)

    def test_verify_determinism_different_seeds(self):
        """Different seeds produce different metrics.json."""
        from src.bench.experiment import ExperimentConfig
        from src.bench.seeds import verify_determinism

        with tempfile.TemporaryDirectory() as directory:
            config_a = ExperimentConfig(
                act="smoke",
                experiment="seed-a",
                seed=1,
                dataset_path="data/tinyshakespeare.txt",
                steps=3,
                batch_size=4,
                context_length=16,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
                eval_interval=1,
                eval_batches=1,
                device="cpu",
                tokenizer="character",
                output_root=directory,
            )
            config_b = ExperimentConfig(
                act="smoke",
                experiment="seed-b",
                seed=999,
                dataset_path="data/tinyshakespeare.txt",
                steps=3,
                batch_size=4,
                context_length=16,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
                eval_interval=1,
                eval_batches=1,
                device="cpu",
                tokenizer="character",
                output_root=directory,
            )
            identical, message = verify_determinism(config_a, config_b)
            # Different seeds should (almost certainly) produce different results
            # We check that the message indicates a difference
            self.assertFalse(identical)
            self.assertIn("differ", message)


@unittest.skipIf(torch is None, "PyTorch unavailable")
class StatsTests(unittest.TestCase):
    def test_mean_basic(self):
        from src.bench.stats import mean

        self.assertAlmostEqual(mean([1, 2, 3, 4, 5]), 3.0)
        self.assertAlmostEqual(mean([10]), 10.0)
        self.assertAlmostEqual(mean([-1, 1]), 0.0)

    def test_mean_empty_raises(self):
        from src.bench.stats import mean

        with self.assertRaises(ValueError):
            mean([])

    def test_std_basic(self):
        from src.bench.stats import std

        # Sample std of [1, 2, 3, 4, 5] = sqrt(2.5) ≈ 1.5811
        self.assertAlmostEqual(std([1, 2, 3, 4, 5]), math.sqrt(2.5), places=10)

    def test_std_single_value_raises(self):
        from src.bench.stats import std

        with self.assertRaises(ValueError):
            std([42])

    def test_wilson_interval_known_values(self):
        """Wilson interval matches published reference values.

        Reference: 50 successes out of 100 trials, 95% confidence.
        Known result: approximately (0.4038, 0.5962)
        """
        from src.bench.stats import wilson_interval

        lower, upper = wilson_interval(50, 100, confidence=0.95)
        self.assertAlmostEqual(lower, 0.4038, places=3)
        self.assertAlmostEqual(upper, 0.5962, places=3)

    def test_wilson_interval_extreme_p(self):
        """Wilson interval handles extreme proportions (p=0, p=1).

        Unlike the normal approximation, Wilson gives finite intervals at p=0 and p=1.
        """
        from src.bench.stats import wilson_interval

        # 0 successes out of 100
        lower, upper = wilson_interval(0, 100, confidence=0.95)
        self.assertGreater(lower, 0.0)  # Not exactly 0
        self.assertLess(lower, upper)
        self.assertLess(upper, 0.1)  # Upper bound should be small

        # 100 successes out of 100
        lower, upper = wilson_interval(100, 100, confidence=0.95)
        self.assertGreater(lower, 0.9)  # Lower bound should be high
        self.assertLessEqual(upper, 1.0)  # Wilson gives exactly 1.0 at p=1
        self.assertLess(lower, upper)

    def test_wilson_interval_small_n(self):
        """Wilson interval works with small n."""
        from src.bench.stats import wilson_interval

        # 3 successes out of 3
        lower, upper = wilson_interval(3, 3, confidence=0.95)
        self.assertGreater(lower, 0.0)
        self.assertLessEqual(upper, 1.0)  # Wilson gives exactly 1.0 at p=1
        self.assertLess(lower, upper)

        # 0 successes out of 3
        lower, upper = wilson_interval(0, 3, confidence=0.95)
        self.assertGreaterEqual(lower, 0.0)  # Wilson lower is exactly 0 at p=0
        self.assertLess(upper, 1.0)

    def test_wilson_interval_invalid_inputs(self):
        """Wilson interval rejects invalid inputs."""
        from src.bench.stats import wilson_interval

        with self.assertRaises(ValueError):
            wilson_interval(0, 0)  # total must be positive

        with self.assertRaises(ValueError):
            wilson_interval(-1, 10)  # negative successes

        with self.assertRaises(ValueError):
            wilson_interval(11, 10)  # successes > total

        with self.assertRaises(ValueError):
            wilson_interval(5, 10, confidence=0.0)  # confidence must be in (0,1)

        with self.assertRaises(ValueError):
            wilson_interval(5, 10, confidence=1.0)

    def test_wilson_interval_symmetry(self):
        """Wilson interval is symmetric: (k, n) and (n-k, n) give mirrored bounds."""
        from src.bench.stats import wilson_interval

        lower1, upper1 = wilson_interval(30, 100, confidence=0.95)
        lower2, upper2 = wilson_interval(70, 100, confidence=0.95)
        self.assertAlmostEqual(lower1, 1 - upper2, places=10)
        self.assertAlmostEqual(upper1, 1 - lower2, places=10)

    def test_wilson_interval_contains_point_estimate(self):
        """Wilson interval always contains the point estimate p = k/n."""
        from src.bench.stats import wilson_interval

        for k, n in [(0, 10), (3, 10), (5, 10), (7, 10), (10, 10), (50, 200)]:
            lower, upper = wilson_interval(k, n, confidence=0.95)
            p = k / n
            # Use tolerance for floating-point comparison at boundaries
            self.assertLessEqual(lower, p + 1e-9, f"lower > p for k={k}, n={n}")
            self.assertGreaterEqual(upper, p - 1e-9, f"upper < p for k={k}, n={n}")

    def test_wilson_interval_monotonic_confidence(self):
        """Higher confidence gives wider intervals."""
        from src.bench.stats import wilson_interval

        lower90, upper90 = wilson_interval(50, 100, confidence=0.90)
        lower95, upper95 = wilson_interval(50, 100, confidence=0.95)
        lower99, upper99 = wilson_interval(50, 100, confidence=0.99)

        width90 = upper90 - lower90
        width95 = upper95 - lower95
        width99 = upper99 - lower99

        self.assertLess(width90, width95)
        self.assertLess(width95, width99)


if __name__ == "__main__":
    unittest.main()
