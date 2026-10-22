from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class TrainingTests(unittest.TestCase):
    def test_bpe_training_and_checkpoint_resume(self) -> None:
        from src.training import TrainConfig, train

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "tiny.txt"
            dataset.write_text("to be or not to be " * 30, encoding="utf-8")
            common = dict(dataset_path=str(dataset), batch_size=2, context_length=8, embedding_dim=16, num_heads=4, num_layers=1, eval_interval=1, eval_batches=1, device="cpu", seed=3)
            _, tokenizer, _ = train(TrainConfig(**common, tokenizer="bpe", bpe_merges=8, steps=1, output_dir=str(root / "first")))
            self.assertGreater(tokenizer.vocabulary_size, 1)
            checkpoint = root / "first" / "checkpoint.pt"
            _, _, metrics = train(TrainConfig(**common, tokenizer="bpe", bpe_merges=8, steps=2, output_dir=str(root / "resumed"), resume_from=str(checkpoint)))
            resumed = torch.load(root / "resumed" / "checkpoint.pt", map_location="cpu", weights_only=False)
            self.assertEqual(resumed["completed_steps"], 2)
            self.assertEqual(metrics[-1]["step"], 2.0)
