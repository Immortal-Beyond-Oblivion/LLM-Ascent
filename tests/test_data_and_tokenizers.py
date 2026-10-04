from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.tokenizer import BPETokenizer, CharacterTokenizer


class TokenizerTests(unittest.TestCase):
    def test_bpe_round_trip_and_serialization(self) -> None:
        text = "banana bandana\n"
        tokenizer = BPETokenizer.train(text, max_merges=8)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            self.assertEqual(BPETokenizer.load(path).encode(text), tokenizer.encode(text))

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_causal_batches_are_long_integer_shifted_tensors(self) -> None:
        from src.data import TextDataset

        dataset = TextDataset.from_text("abcdefghij" * 3, CharacterTokenizer.from_text("abcdefghij"))
        inputs, targets = dataset.batch("train", batch_size=3, context_length=4, generator=torch.Generator().manual_seed(1))
        self.assertEqual(inputs.dtype, torch.long)
        self.assertEqual(tuple(inputs.shape), (3, 4))
        self.assertTrue(torch.equal(inputs[:, 1:], targets[:, :-1]))
