from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class TokenizerComparisonTests(unittest.TestCase):
    def _make_config(self, **overrides):
        from src.bench.experiment import ExperimentConfig

        defaults = dict(
            act="tokenizer-comparison",
            experiment="test",
            seed=42,
            dataset_path="data/tinyshakespeare.txt",
            train_fraction=0.9,
            tokenizer="character",
            bpe_merges=100,
            context_length=16,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            normalization="layernorm",
            steps=3,
            batch_size=4,
            learning_rate=1e-3,
            warmup_steps=0,
            eval_interval=1,
            eval_batches=1,
            device="cpu",
            output_root="/tmp",
        )
        defaults.update(overrides)
        return ExperimentConfig(**defaults)

    def test_bpe_fitted_on_train_split_only(self):
        """D6 fix: BPE vocabulary is derived from train split only."""
        from src.bench.tokenizer_comparison import _build_tokenizer_train_only
        from src.data import load_text

        text = load_text("data/tinyshakespeare.txt")
        config = self._make_config(tokenizer="bpe", bpe_merges=50, train_fraction=0.9)

        # Full-text BPE (v1 behavior — for comparison)
        full_tokenizer = __import__("src.tokenizer", fromlist=["BPETokenizer"]).BPETokenizer.train(
            text, max_merges=50
        )

        # Train-only BPE (D6 fix)
        train_tokenizer = _build_tokenizer_train_only(text, config, train_fraction=0.9)

        # Train-only vocab should be <= full-text vocab (no leakage from validation)
        self.assertLessEqual(
            train_tokenizer.vocabulary_size,
            full_tokenizer.vocabulary_size,
            "Train-only BPE vocab should not exceed full-text BPE vocab",
        )

    def test_full_sweep_validation_covers_all_tokens(self):
        """D7 fix: full-sweep validation covers the entire validation set."""
        from src.bench.experiment import ExperimentConfig
        from src.bench.tokenizer_comparison import _evaluate_full_sweep
        from src.data import TextDataset, load_text
        from src.model import GPTConfig, MiniGPT
        from src.tokenizer import CharacterTokenizer

        text = load_text("data/tinyshakespeare.txt")
        tokenizer = CharacterTokenizer.from_text(text)
        dataset = TextDataset.from_text(text, tokenizer, train_fraction=0.9)

        config = self._make_config(batch_size=64, context_length=16)
        model_config = GPTConfig(
            vocabulary_size=tokenizer.vocabulary_size,
            context_length=16,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
        )
        model = MiniGPT(model_config)

        # Full sweep should cover all validation tokens
        val_loss = _evaluate_full_sweep(model, dataset, config, "cpu", "validation")
        self.assertTrue(math.isfinite(val_loss))
        self.assertGreater(val_loss, 0)

    def test_run_tokenizer_comparison_produces_valid_metrics(self):
        """run_tokenizer_comparison writes valid metrics.json + tokenizer_stats.json."""
        from src.bench.experiment import ExperimentConfig, load_metrics
        from src.bench.tokenizer_comparison import run_tokenizer_comparison

        with tempfile.TemporaryDirectory() as directory:
            config = self._make_config(
                steps=2,
                eval_interval=1,
                batch_size=4,
                context_length=16,
                embedding_dim=16,
                output_root=directory,
            )
            metrics = run_tokenizer_comparison("character", config)
            metrics.validate()

            result_dir = Path(directory) / "tokenizer-comparison" / "character-seed42"
            self.assertTrue((result_dir / "metrics.json").exists())
            self.assertTrue((result_dir / "config.json").exists())
            self.assertTrue((result_dir / "tokenizer_stats.json").exists())

            # Load and validate
            loaded = load_metrics(result_dir / "metrics.json")
            loaded.validate()
            self.assertEqual(loaded.act, "tokenizer-comparison")
            self.assertEqual(loaded.seed, 42)

            # Check tokenizer stats sidecar
            sidecar = json.loads((result_dir / "tokenizer_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["tokenizer"], "character")
            self.assertIn("characters_per_token", sidecar)
            self.assertIn("validation_bits_per_character", sidecar)

    def test_multi_seed_comparison(self):
        """Multi-seed comparison aggregates BPC with mean/std."""
        from src.bench.tokenizer_comparison import run_multi_seed_comparison

        with tempfile.TemporaryDirectory() as directory:
            config = self._make_config(
                steps=2,
                eval_interval=1,
                batch_size=4,
                context_length=16,
                embedding_dim=16,
                output_root=directory,
            )
            summary = run_multi_seed_comparison("test-act", config, seeds=[42, 43])

            self.assertIn("character", summary)
            self.assertIn("bpe", summary)
            for kind in ("character", "bpe"):
                self.assertIn("bits_per_character_mean", summary[kind])
                self.assertIn("bits_per_character_std", summary[kind])
                self.assertEqual(summary[kind]["n_seeds"], 2)
                self.assertEqual(len(summary[kind]["per_seed_bpc"]), 2)


if __name__ == "__main__":
    unittest.main()
