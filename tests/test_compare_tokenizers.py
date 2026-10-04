from __future__ import annotations

import argparse
import math
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class CompareTokenizersTests(unittest.TestCase):
    def test_matched_experiment_reports_character_normalised_metrics(self) -> None:
        from src.eval.compare_tokenizers import TOKENIZERS, _markdown, run_experiment

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "tiny.txt"
            text = "to be or not to be that is the question " * 40
            dataset.write_text(text, encoding="utf-8")
            args = argparse.Namespace(
                dataset_path=str(dataset), steps=2, batch_size=2, context_length=8, embedding_dim=16,
                num_heads=4, num_layers=1, learning_rate=1e-3, eval_interval=1, eval_batches=1,
                seed=5, device="cpu", bpe_merges=10, prompt="to be", sample_chars=12,
            )
            results = [run_experiment(kind, args, root, text) for kind in TOKENIZERS]

        by_name = {result["tokenizer"]: result for result in results}
        self.assertEqual(by_name["character"]["merges"], 0)
        self.assertGreater(by_name["bpe"]["merges"], 0)
        self.assertAlmostEqual(by_name["character"]["characters_per_token"], 1.0)
        self.assertGreater(by_name["bpe"]["characters_per_token"], 1.0)
        for result in results:
            expected = result["validation_loss"] / math.log(2) / result["characters_per_token"]
            self.assertAlmostEqual(result["validation_bits_per_character"], expected)
            self.assertTrue(all(len(sample["text"]) <= 12 for sample in result["samples"]))
            self.assertTrue(all(0.0 <= sample["repetition_3gram"] <= 1.0 for sample in result["samples"]))
        self.assertIn("Validation bits per character", _markdown(results))


if __name__ == "__main__":
    unittest.main()
