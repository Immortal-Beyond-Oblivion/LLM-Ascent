"""Sanity checks for the deterministic Phase 1 character tokenizer."""

from __future__ import annotations

import unittest

from src.tokenizer import CharacterTokenizer


class CharacterTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer.from_text("cab\n")

    def test_vocabulary_has_stable_canonical_order(self) -> None:
        self.assertEqual(self.tokenizer.vocabulary, ("<unk>", "\n", "a", "b", "c"))
        self.assertEqual(self.tokenizer.unknown_token_id, 0)
        self.assertEqual(self.tokenizer.vocabulary_size, 5)

    def test_round_trip_for_known_characters(self) -> None:
        text = "cab\nabc"
        token_ids = self.tokenizer.encode(text)
        self.assertEqual(token_ids, [4, 2, 3, 1, 2, 3, 4])
        self.assertEqual(self.tokenizer.decode(token_ids), text)

    def test_unknown_character_uses_reserved_token(self) -> None:
        token_ids = self.tokenizer.encode("a?")
        self.assertEqual(token_ids, [2, 0])
        self.assertEqual(self.tokenizer.decode(token_ids), "a<unk>")
        self.assertEqual(self.tokenizer.decode(token_ids, skip_special_tokens=True), "a")

    def test_empty_text_is_supported(self) -> None:
        tokenizer = CharacterTokenizer.from_text("")
        self.assertEqual(tokenizer.vocabulary, ("<unk>",))
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.decode([]), "")

    def test_invalid_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.tokenizer.encode(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.tokenizer.decode([self.tokenizer.vocabulary_size])
        with self.assertRaises(TypeError):
            self.tokenizer.decode([1.0])

    def test_tensor_interface_has_expected_dtype_shape_and_range(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed in the active Python environment")

        token_ids = self.tokenizer.encode_tensor("cab")
        self.assertIsInstance(token_ids, torch.Tensor)
        self.assertEqual(token_ids.dtype, torch.long)
        self.assertEqual(tuple(token_ids.shape), (3,))
        self.assertTrue(torch.all((0 <= token_ids) & (token_ids < self.tokenizer.vocabulary_size)))


if __name__ == "__main__":
    unittest.main()
