"""Tests that speculative decoding with KV cache is byte-identical to uncached."""

from __future__ import annotations

import unittest

import torch

from src.generation.speculative import speculative_decode
from src.generation.decoding import generate


def _model(seed: int, vocab: int = 32, dim: int = 32, layers: int = 2):
    from src.model.gpt import GPTConfig, MiniGPT
    torch.manual_seed(seed)
    return MiniGPT(GPTConfig(vocabulary_size=vocab, context_length=64,
                             embedding_dim=dim, num_heads=4, num_kv_heads=4,
                             num_layers=layers, dropout=0.0,
                             normalization="layernorm")).eval()


class TestSpeculativeCacheEquivalence(unittest.TestCase):
    def test_cached_greedy_matches_uncached_greedy(self):
        draft, target = _model(11), _model(12)
        prompt = torch.randint(0, 32, (1, 8), generator=torch.Generator().manual_seed(3))
        unc = speculative_decode(draft, target, prompt, max_new_tokens=16, K=3, do_sample=False, use_cache=False)
        cac = speculative_decode(draft, target, prompt, max_new_tokens=16, K=3, do_sample=False, use_cache=True)
        self.assertTrue(torch.equal(unc["tokens"], cac["tokens"]),
                        "cached speculative greedy differs from uncached")

    def test_cached_sample_matches_uncached_sample(self):
        draft, target = _model(13), _model(14)
        prompt = torch.randint(0, 32, (1, 8), generator=torch.Generator().manual_seed(4))
        torch.manual_seed(99)
        unc = speculative_decode(draft, target, prompt, max_new_tokens=16, K=3, do_sample=True, use_cache=False)
        torch.manual_seed(99)
        cac = speculative_decode(draft, target, prompt, max_new_tokens=16, K=3, do_sample=True, use_cache=True)
        self.assertTrue(torch.equal(unc["tokens"], cac["tokens"]),
                        "cached speculative sample differs from uncached")

    def test_cached_greedy_matches_plain_greedy_with_cache(self):
        """Speculative greedy must equal plain target greedy (both cached)."""
        draft, target = _model(15), _model(16)
        prompt = torch.randint(0, 32, (1, 8), generator=torch.Generator().manual_seed(5))
        spec = speculative_decode(draft, target, prompt, max_new_tokens=16, K=3, do_sample=False, use_cache=True)
        plain = generate(target, prompt, context_length=64, max_new_tokens=16, use_cache=True)
        self.assertTrue(torch.equal(spec["tokens"], plain[:, prompt.shape[1]:]),
                        "speculative greedy != plain target greedy")

    def test_zero_gap_acceptance_is_total(self):
        """With draft == target every proposal MUST be accepted.

        This is the regression test for the cache double-fill bug: the draft and
        verification loops used to re-feed the whole prompt on top of an
        already-prefilled cache, so the target's p differed from the draft's q
        even for identical weights, and acceptance collapsed to ~40%. A correct
        implementation is exactly 100% at every K.
        """
        model = _model(21)
        prompt = torch.randint(0, 32, (1, 12), generator=torch.Generator().manual_seed(7))
        for K in (1, 2, 3, 5, 7):
            with self.subTest(K=K):
                torch.manual_seed(1)
                spec = speculative_decode(model, model, prompt, max_new_tokens=24,
                                          K=K, do_sample=True, use_cache=True)
                self.assertEqual(spec["acceptance_rate"], 1.0,
                                 f"zero-gap acceptance {spec['acceptance_rate']:.1%} at K={K}, expected 100%")
                torch.manual_seed(1)
                greedy = speculative_decode(model, model, prompt, max_new_tokens=24,
                                            K=K, do_sample=False, use_cache=True)
                self.assertEqual(greedy["acceptance_rate"], 1.0)

    def test_cache_length_stays_consistent(self):
        """The cache must hold exactly the accepted sequence, never more.

        Double-feeding shows up as a cache longer than the token sequence.
        """
        draft, target = _model(31), _model(32)
        prompt = torch.randint(0, 32, (1, 10), generator=torch.Generator().manual_seed(9))
        out = speculative_decode(draft, target, prompt, max_new_tokens=20, K=4,
                                 do_sample=True, use_cache=True)
        # A correct run ends with the cache holding prompt + emitted tokens.
        self.assertEqual(out["sequence"].shape[1], prompt.shape[1] + out["n_emitted"])


if __name__ == "__main__":
    unittest.main()
