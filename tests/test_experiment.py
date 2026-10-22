from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class ExperimentFrameworkTests(unittest.TestCase):
    def _make_config(self, **overrides):
        from src.bench.experiment import ExperimentConfig

        defaults = dict(
            act="test-act",
            experiment="test-exp",
            seed=42,
            dataset_path="data/tinyshakespeare.txt",
            steps=5,
            batch_size=4,
            context_length=16,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            eval_interval=2,
            eval_batches=2,
            device="cpu",
            tokenizer="character",
        )
        defaults.update(overrides)
        return ExperimentConfig(**defaults)

    def test_metrics_round_trip(self):
        """metrics.json round-trips through JSON and validates."""
        from src.bench.experiment import MetricsSchema

        metrics = MetricsSchema(
            act="test-act",
            experiment="test-exp",
            seed=42,
            device="cpu",
            config={"steps": 100},
            parameters=1000,
            trainable_parameters=1000,
            total_tokens=10000,
            train_tokens=9000,
            validation_tokens=1000,
            vocabulary_size=66,
            loss_history=[
                {"step": 1.0, "train_loss": 4.0, "validation_loss": 4.1, "validation_perplexity": 60.3},
                {"step": 100.0, "train_loss": 2.5, "validation_loss": 2.6, "validation_perplexity": 13.5},
            ],
            wall_clock_seconds=1.5,
            steps_per_second=66.7,
            final_step=100.0,
            final_train_loss=2.5,
            final_validation_loss=2.6,
            final_validation_perplexity=13.5,
        )
        metrics.validate()

        # Round-trip through JSON
        text = metrics.to_json()
        restored = MetricsSchema.from_json(text)
        restored.validate()

        self.assertEqual(restored.act, "test-act")
        self.assertEqual(restored.experiment, "test-exp")
        self.assertEqual(restored.seed, 42)
        self.assertEqual(restored.device, "cpu")
        self.assertEqual(restored.parameters, 1000)
        self.assertEqual(restored.total_tokens, 10000)
        self.assertEqual(len(restored.loss_history), 2)
        self.assertEqual(restored.final_step, 100.0)
        self.assertAlmostEqual(restored.final_validation_loss, 2.6)

    def test_metrics_validation_catches_mismatch(self):
        """Validation catches final metrics that don't match history."""
        from src.bench.experiment import MetricsSchema

        metrics = MetricsSchema(
            act="test",
            experiment="test",
            seed=1,
            device="cpu",
            config={},
            parameters=100,
            trainable_parameters=100,
            total_tokens=1000,
            train_tokens=900,
            validation_tokens=100,
            vocabulary_size=10,
            loss_history=[{"step": 1.0, "train_loss": 3.0, "validation_loss": 3.1, "validation_perplexity": 22.2}],
            final_step=1.0,
            final_train_loss=3.0,
            final_validation_loss=99.9,  # MISMATCH
            final_validation_perplexity=22.2,
        )
        with self.assertRaises(ValueError):
            metrics.validate()

    def test_metrics_validation_catches_empty_history(self):
        """Validation catches empty loss_history."""
        from src.bench.experiment import MetricsSchema

        metrics = MetricsSchema(
            act="test",
            experiment="test",
            seed=1,
            device="cpu",
            config={},
            parameters=100,
            trainable_parameters=100,
            total_tokens=1000,
            train_tokens=900,
            validation_tokens=100,
            vocabulary_size=10,
            loss_history=[],
        )
        with self.assertRaises(ValueError):
            metrics.validate()

    def test_run_experiment_produces_valid_metrics(self):
        """run_experiment writes config.json and metrics.json that validate."""
        from src.bench.experiment import ExperimentConfig, run_experiment, load_metrics

        with tempfile.TemporaryDirectory() as directory:
            config = self._make_config(
                act="smoke",
                experiment="round-trip",
                output_root=directory,
                steps=3,
                eval_interval=1,
                eval_batches=1,
            )
            metrics = run_experiment(config)
            metrics.validate()

            # Check files exist
            result_dir = Path(directory) / "smoke" / "round-trip"
            self.assertTrue((result_dir / "config.json").exists())
            self.assertTrue((result_dir / "metrics.json").exists())

            # Load and validate from disk
            loaded = load_metrics(result_dir / "metrics.json")
            loaded.validate()
            self.assertEqual(loaded.act, "smoke")
            self.assertEqual(loaded.experiment, "round-trip")
            self.assertEqual(loaded.seed, 42)
            self.assertGreater(loaded.parameters, 0)
            self.assertGreater(loaded.total_tokens, 0)
            self.assertGreater(len(loaded.loss_history), 0)

    def test_config_round_trip(self):
        """ExperimentConfig round-trips through dict."""
        from src.bench.experiment import ExperimentConfig

        config = self._make_config()
        restored = ExperimentConfig.from_dict(config.to_dict())
        self.assertEqual(restored.act, config.act)
        self.assertEqual(restored.experiment, config.experiment)
        self.assertEqual(restored.seed, config.seed)
        self.assertEqual(restored.steps, config.steps)

    def test_plot_metrics(self):
        """plot_metrics produces a PNG file."""
        import matplotlib
        matplotlib.use("Agg")
        from src.bench.experiment import MetricsSchema, plot_metrics

        metrics = MetricsSchema(
            act="test",
            experiment="test",
            seed=1,
            device="cpu",
            config={},
            parameters=100,
            trainable_parameters=100,
            total_tokens=1000,
            train_tokens=900,
            validation_tokens=100,
            vocabulary_size=10,
            loss_history=[
                {"step": 1.0, "train_loss": 4.0, "validation_loss": 4.1, "validation_perplexity": 60.3},
                {"step": 10.0, "train_loss": 3.0, "validation_loss": 3.1, "validation_perplexity": 22.2},
            ],
            final_step=10.0,
            final_train_loss=3.0,
            final_validation_loss=3.1,
            final_validation_perplexity=22.2,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "loss.png"
            plot_metrics(metrics, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
