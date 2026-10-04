from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.tokenizer import BPETokenizer, CharacterTokenizer, tokenizer_from_checkpoint


class TokenizerTests(unittest.TestCase):
    def test_bpe_encoding_replays_merges_in_rank_order(self) -> None:
        # Rank 0 merges (b, c) first, so "abc" must become ["a", "bc"], not ["ab", "c"].
        tokenizer = BPETokenizer(["<unk>", "a", "b", "c", "bc", "ab"], [("b", "c"), ("a", "b")])
        self.assertEqual(tokenizer.encode("abc"), [1, 4])
        self.assertEqual(tokenizer.decode(tokenizer.encode("abc")), "abc")

    def test_bpe_encoding_matches_training_segmentation(self) -> None:
        text = "banana bandana banana\n"
        tokenizer = BPETokenizer.train(text, max_merges=6)
        words = [list(text)]
        for pair in tokenizer.merges:
            words = [BPETokenizer._merge_word(word, pair, "".join(pair)) for word in words]
        self.assertEqual(tokenizer.encode(text), [tokenizer.token_to_id[piece] for piece in words[0]])

    def test_tokenizer_from_checkpoint_supports_bpe_character_and_legacy(self) -> None:
        bpe = BPETokenizer.train("banana bandana", max_merges=4)
        payload = {"type": "bpe", "vocabulary": list(bpe.vocabulary), "merges": [list(pair) for pair in bpe.merges]}
        restored = tokenizer_from_checkpoint({"tokenizer": payload, "vocabulary": bpe.vocabulary})
        self.assertIsInstance(restored, BPETokenizer)
        self.assertEqual(restored.encode("banana"), bpe.encode("banana"))
        self.assertEqual(restored.decode(restored.encode("banana"), skip_special_tokens=True), "banana")
        char = CharacterTokenizer.from_text("abc")
        self.assertIsInstance(tokenizer_from_checkpoint({"tokenizer": {"type": "character", "vocabulary": list(char.vocabulary)}}), CharacterTokenizer)
        self.assertIsInstance(tokenizer_from_checkpoint({"vocabulary": char.vocabulary}), CharacterTokenizer)
        with self.assertRaises(ValueError):
            tokenizer_from_checkpoint({"tokenizer": {"type": "unknown"}})

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
