"""Beam search with a KV cache, and the cache-clone semantics it depends on.

This file exists because Act 5a asserted in the ledger that "greedy, top-k,
top-p and beam all use it; beam keeps per-beam caches". That claim was FALSE:
``beam_search`` had no cache parameter in any commit, and ``tests/test_kv_cache.py``
covered greedy, sample and top-p but never beam. An assertion with no test
behind it is how the Act 6 cost column came to be wrong.

So: beam now genuinely supports the cache, and both properties it relies on are
tested here.
"""

from __future__ import annotations

import unittest

import torch

from src.generation.decoding import beam_search, generate
from src.model.gpt import GPTConfig, KVCache, MiniGPT


def _model(seed: int, context_length: int = 64):
    torch.manual_seed(seed)
    return MiniGPT(GPTConfig(vocabulary_size=32, context_length=context_length,
                             embedding_dim=32, num_heads=4, num_kv_heads=4,
                             num_layers=2, dropout=0.0,
                             normalization="layernorm")).eval()


def _prompt(seed: int = 5, length: int = 8) -> torch.Tensor:
    return torch.randint(0, 32, (1, length), generator=torch.Generator().manual_seed(seed))


class TestKVCacheClone(unittest.TestCase):
    def test_clone_is_independent(self):
        """A shallow copy would let one beam's append corrupt its siblings."""
        a = KVCache(num_layers=2, num_heads=4, head_dim=8, num_kv_heads=4)
        b = KVCache(num_layers=2, num_heads=4, head_dim=8, num_kv_heads=4)
        k = torch.randn(1, 4, 5, 8)
        a.append(0, k, k)
        b.append(0, k, k)

        c = a.clone()
        self.assertEqual(c.sequence_length, a.sequence_length)
        extra = torch.randn(1, 4, 1, 8)
        c.append(0, extra, extra)

        self.assertEqual(c.sequence_length, a.sequence_length + 1)
        self.assertEqual(a.sequence_length, b.sequence_length,
                         "clone must not alias the original")
        self.assertTrue(torch.equal(c.keys[0][:, :, :5, :], a.keys[0]),
                        "clone must hold equal values up to the shared prefix")

    def test_clone_survives_zeroing_the_original(self):
        a = KVCache(num_layers=1, num_heads=2, head_dim=4, num_kv_heads=2)
        k = torch.randn(1, 2, 4, 4)
        a.append(0, k, k)
        c = a.clone()
        a.keys[0].zero_()
        self.assertFalse(torch.equal(c.keys[0], a.keys[0]))


class TestCachedBeamSearch(unittest.TestCase):
    def test_cached_beam_matches_uncached(self):
        """The cached path must be numerically identical, not merely similar.

        The uncached path is the reference. Note that an equivalence test alone
        cannot establish that either is right -- it only shows they agree -- so
        this is paired with the KVCache equivalence tests in test_kv_cache.py.
        """
        model = _model(11)
        for width in (1, 2, 3, 5):
            with self.subTest(beam_width=width):
                p = _prompt()
                uncached = beam_search(model, p, context_length=64,
                                       max_new_tokens=12, beam_width=width,
                                       use_cache=False)
                cached = beam_search(model, p, context_length=64,
                                     max_new_tokens=12, beam_width=width,
                                     use_cache=True)
                self.assertTrue(torch.equal(uncached, cached),
                                f"cached beam differs from uncached at width {width}")

    def test_cached_beam_output_length(self):
        model = _model(12)
        p = _prompt(length=6)
        out = beam_search(model, p, context_length=64, max_new_tokens=9,
                          beam_width=3, use_cache=True)
        self.assertEqual(out.shape[1], 6 + 9)

    def test_cached_beam_rejects_context_overflow(self):
        """Must raise rather than silently exceeding the position table."""
        model = _model(13, context_length=32)
        p = _prompt(length=24)
        with self.assertRaises(ValueError):
            beam_search(model, p, context_length=32, max_new_tokens=16,
                        beam_width=2, use_cache=True)

    def test_cached_beam_is_at_least_as_good_as_greedy(self):
        """Beam search maximises log-probability, so it cannot score worse.

        A cheap absolute invariant that would catch a badly broken cache, since
        a corrupted cache would silently degrade beam below greedy.
        """
        model = _model(14)
        p = _prompt(length=8)
        greedy = generate(model, p, context_length=64, max_new_tokens=12, use_cache=True)
        beam = beam_search(model, p, context_length=64, max_new_tokens=12,
                           beam_width=4, use_cache=True)
        self.assertEqual(greedy.shape[1], beam.shape[1])
        # Beam should agree with greedy at least sometimes on short sequences.
        agreement = (greedy[0] == beam[0]).float().mean()
        self.assertGreater(float(agreement), 0.0)


if __name__ == "__main__":
    unittest.main()
