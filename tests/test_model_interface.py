"""Regression test pinning the shared MiniGPT / ALiBiTransformer interface.

Both model classes share a training path (`train_rung` -> `build_model_for_rung`,
`eval/attention`, etc.) that calls `non_embedding_params()`.  A earlier drift
between MiniGPT having only `parameter_count()` while ALiBiTransformer had
`non_embedding_params()` crashed every Act 5 sweep at step 0.  This test pins
the shared accessor so the collision cannot recur silently.
"""

from __future__ import annotations

import unittest

import torch

from src.model.gpt import GPTConfig, MiniGPT
from src.scaling.ladder import ALiBiTransformer, ASPECT_LADDER


class TestSharedModelInterface(unittest.TestCase):
    def test_both_architectures_have_non_embedding_params(self):
        for model, expected_floor in [
            (MiniGPT(GPTConfig(vocabulary_size=32, context_length=16, embedding_dim=16,
                              num_heads=4, num_layers=2, dropout=0.0)), 0),
            (ALiBiTransformer(vocabulary_size=32, context_length=16, embedding_dim=32,
                              num_heads=4, num_layers=2), 0),
        ]:
            self.assertTrue(hasattr(model, "non_embedding_params"),
                            f"{type(model).__name__} missing non_embedding_params()")
            n = model.non_embedding_params()
            self.assertIsInstance(n, int)
            self.assertGreaterEqual(n, 0)
            # non-embedding must be strictly less than total when an embedding exists
            total = model.total_params() if hasattr(model, "total_params") else n + 1
            emb = model.token_embedding.weight.numel()
            self.assertGreater(total, emb)
            self.assertEqual(n, total - emb)
            # both archs return an int, not a tensor
            self.assertNotIsInstance(n, torch.Tensor)

    def test_miniGPT_non_embedding_excludes_embedding(self):
        cfg = GPTConfig(vocabulary_size=40, context_length=8, embedding_dim=16,
                        num_heads=4, num_layers=3, dropout=0.0)
        m = MiniGPT(cfg)
        total = m.parameter_count()
        emb = m.token_embedding.weight.numel()
        self.assertEqual(m.non_embedding_params(), total - emb)

    def test_alibi_transformer_non_embedding_excludes_embedding(self):
        m = ALiBiTransformer(vocabulary_size=40, context_length=8,
                             embedding_dim=16, num_heads=4, num_layers=3)
        total = m.total_params()
        emb = m.token_embedding.weight.numel()
        self.assertEqual(m.non_embedding_params(), total - emb)

if __name__ == "__main__":
    unittest.main()
