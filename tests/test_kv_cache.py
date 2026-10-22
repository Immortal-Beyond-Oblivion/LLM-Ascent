"""Tests for KV cache correctness and performance."""

from __future__ import annotations

import time
import unittest

import torch

from src.model.gpt import GPTConfig, MiniGPT
from src.generation.decoding import generate, greedy_decode


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        vocabulary_size=256,
        context_length=64,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        normalization="layernorm",
        optimized_attention=True,
    )


class TestKVCacheCorrectness(unittest.TestCase):
    """Cached generation must produce byte-identical output to uncached."""

    def test_greedy_byte_identical(self):
        torch.manual_seed(42)
        model = MiniGPT(_tiny_config())
        model.eval()
        prompt = torch.randint(0, 256, (1, 16), dtype=torch.long)

        # Uncached
        torch.manual_seed(123)
        out_uncached = greedy_decode(model, prompt, context_length=64, max_new_tokens=32)

        # Cached
        torch.manual_seed(123)
        out_cached = generate(model, prompt, context_length=64, max_new_tokens=32, use_cache=True)

        self.assertTrue(
            torch.equal(out_uncached, out_cached),
            f"Cached output differs from uncached.\n"
            f"Uncached: {out_uncached[0].tolist()}\n"
            f"Cached:   {out_cached[0].tolist()}",
        )

    def test_sample_byte_identical(self):
        torch.manual_seed(42)
        model = MiniGPT(_tiny_config())
        model.eval()
        prompt = torch.randint(0, 256, (1, 16), dtype=torch.long)

        torch.manual_seed(456)
        out_uncached = generate(
            model, prompt, context_length=64, max_new_tokens=32,
            do_sample=True, temperature=0.8, top_k=10, use_cache=False,
        )

        torch.manual_seed(456)
        out_cached = generate(
            model, prompt, context_length=64, max_new_tokens=32,
            do_sample=True, temperature=0.8, top_k=10, use_cache=True,
        )

        self.assertTrue(
            torch.equal(out_uncached, out_cached),
            "Sampled cached output differs from uncached.",
        )

    def test_top_p_byte_identical(self):
        torch.manual_seed(42)
        model = MiniGPT(_tiny_config())
        model.eval()
        prompt = torch.randint(0, 256, (1, 16), dtype=torch.long)

        torch.manual_seed(789)
        out_uncached = generate(
            model, prompt, context_length=64, max_new_tokens=32,
            do_sample=True, temperature=0.9, top_p=0.9, use_cache=False,
        )

        torch.manual_seed(789)
        out_cached = generate(
            model, prompt, context_length=64, max_new_tokens=32,
            do_sample=True, temperature=0.9, top_p=0.9, use_cache=True,
        )

        self.assertTrue(
            torch.equal(out_uncached, out_cached),
            "Top-p cached output differs from uncached.",
        )


class TestKVCachePerformance(unittest.TestCase):
    """Cache should speed up generation at longer contexts."""

    def test_speedup_at_long_context(self):
        torch.manual_seed(42)
        config = GPTConfig(
            vocabulary_size=256,
            context_length=1024,
            embedding_dim=128,
            num_heads=4,
            num_layers=4,
            dropout=0.0,
        )
        model = MiniGPT(config)
        model.eval()
        prompt = torch.randint(0, 256, (1, 512), dtype=torch.long)

        # Warmup
        _ = greedy_decode(model, prompt, context_length=1024, max_new_tokens=4)
        _ = generate(model, prompt, context_length=1024, max_new_tokens=4, use_cache=True)

        # Uncached timing
        t0 = time.perf_counter()
        _ = greedy_decode(model, prompt, context_length=1024, max_new_tokens=32)
        t_uncached = time.perf_counter() - t0

        # Cached timing
        t0 = time.perf_counter()
        _ = generate(model, prompt, context_length=1024, max_new_tokens=32, use_cache=True)
        t_cached = time.perf_counter() - t0

        speedup = t_uncached / t_cached
        print(f"\n  Context 512+32: uncached={t_uncached:.3f}s cached={t_cached:.3f}s speedup={speedup:.2f}x")
        self.assertGreater(speedup, 1.5, f"Cache speedup {speedup:.2f}x is too low")


if __name__ == "__main__":
    unittest.main()
