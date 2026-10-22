from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.tokenizer.wordpiece import WordPieceTokenizer
from src.tokenizer.unigram import UnigramLMTokenizer


class WordPieceTests(unittest.TestCase):
    def test_train_produces_merges(self):
        text = "to be or not to be " * 30
        tok = WordPieceTokenizer.train(text, max_merges=10)
        self.assertGreater(len(tok.merges), 0)
        self.assertGreater(tok.vocabulary_size, 1)

    def test_encode_decode_roundtrip(self):
        text = "to be or not"
        tok = WordPieceTokenizer.train("to be or not to be " * 30, max_merges=10)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        self.assertEqual(decoded, text)

    def test_vocabulary_starts_with_unk(self):
        text = "hello world"
        tok = WordPieceTokenizer.train(text, max_merges=5)
        self.assertEqual(tok.vocabulary[0], "<unk>")


class UnigramTests(unittest.TestCase):
    def test_train_produces_vocab(self):
        text = "to be or not to be " * 30
        tok = UnigramLMTokenizer.train(text, target_vocab_size=50)
        self.assertGreater(tok.vocabulary_size, 1)
        self.assertGreater(tok.vocabulary_size, 1)

    def test_encode_decode_roundtrip(self):
        text = "to be"
        tok = UnigramLMTokenizer.train("to be or not to be " * 30, target_vocab_size=50)
        ids = tok.encode(text)
        self.assertGreater(len(ids), 0)
        decoded = tok.decode(ids)
        self.assertEqual(decoded, text)

    def test_vocabulary_starts_with_unk(self):
        text = "hello world"
        tok = UnigramLMTokenizer.train(text, target_vocab_size=20)
        self.assertEqual(tok.vocabulary[0], "<unk>")


class AllTokenizersTests(unittest.TestCase):
    def test_all_four_tokenizers_produce_output(self):
        text = "to be or not to be " * 30
        from src.tokenizer import CharacterTokenizer, BPETokenizer, WordPieceTokenizer, UnigramLMTokenizer

        tokenizers = {
            "character": CharacterTokenizer.from_text(text),
            "bpe": BPETokenizer.train(text, max_merges=10),
            "wordpiece": WordPieceTokenizer.train(text, max_merges=10),
            "unigram": UnigramLMTokenizer.train(text, target_vocab_size=30),
        }

        for name, tok in tokenizers.items():
            ids = tok.encode("to be or not")
            self.assertGreater(len(ids), 0, f"{name} produced no tokens")
            decoded = tok.decode(ids)
            self.assertEqual(decoded, "to be or not", f"{name} roundtrip failed")


if __name__ == "__main__":
    unittest.main()
