"""Tests for the speculative wall-clock benchmark harness.

The benchmark is the instrument behind the 5c speedup claim, so its own
guards need testing. The context guard exists because an earlier version passed
``context_length=prompt.shape[1]``, which silently truncated the context window
to the prompt length and made both arms look far cheaper than they are.
"""

from __future__ import annotations

import unittest

import torch

from src.generation.benchmark_speculative import benchmark_speculative, format_benchmark
from src.model.gpt import GPTConfig, MiniGPT


def _model(seed: int, context_length: int = 64):
    torch.manual_seed(seed)
    return MiniGPT(GPTConfig(vocabulary_size=32, context_length=context_length,
                             embedding_dim=32, num_heads=4, num_kv_heads=4,
                             num_layers=2, dropout=0.0,
                             normalization="layernorm")).eval()


class TestBenchmarkHarness(unittest.TestCase):
    def test_rejects_prompt_plus_generation_beyond_context(self):
        """Must refuse rather than silently truncate the context window."""
        m = _model(1, context_length=64)
        prompt = torch.randint(0, 32, (1, 48))
        with self.assertRaises(ValueError):
            benchmark_speculative(m, m, prompt, K_values=(3,),
                                  max_new_tokens=32, n_repeats=1)
        self.assertEqual(m.config.context_length, 64)
        self.assertEqual(prompt.shape[1] + 32, 80, "test should exceed the window")

    def test_accepts_generation_that_exactly_fills_context(self):
        m = _model(2, context_length=64)
        prompt = torch.randint(0, 32, (1, 32))
        out = benchmark_speculative(m, m, prompt, K_values=(1,),
                                    max_new_tokens=32, n_repeats=1)
        self.assertIn(1, out)

    def test_reports_every_requested_k(self):
        m = _model(3, context_length=64)
        prompt = torch.randint(0, 32, (1, 8))
        ks = (1, 2, 3, 5, 7)
        out = benchmark_speculative(m, m, prompt, K_values=ks,
                                    max_new_tokens=16, n_repeats=1)
        self.assertEqual(sorted(out), sorted(ks))
        for k in ks:
            row = out[k]
            self.assertEqual(row["K"], k)
            self.assertGreater(row["spec_seconds"], 0.0)
            self.assertGreater(row["plain_seconds"], 0.0)
            self.assertAlmostEqual(
                row["speedup"], row["plain_seconds"] / row["spec_seconds"], places=6)

    def test_zero_gap_speedup_is_never_above_one(self):
        """With draft == target no target forward is saved, so it cannot win.

        This is the regression guard for the headline 5c negative result: if a
        future change made speculative decoding look faster than plain cached
        generation at zero gap, this fails and the negative result is void.
        """
        m = _model(4, context_length=64)
        prompt = torch.randint(0, 32, (1, 8))
        out = benchmark_speculative(m, m, prompt, K_values=(1, 3, 7),
                                    max_new_tokens=32, n_repeats=2)
        for k, row in out.items():
            self.assertLessEqual(row["speedup"], 1.0,
                                 f"K={k} reported {row['speedup']:.3f}x at zero gap")
            self.assertEqual(row["acceptance_rate"], 1.0)

    def test_format_benchmark_renders_all_rows(self):
        m = _model(5, context_length=64)
        prompt = torch.randint(0, 32, (1, 8))
        out = benchmark_speculative(m, m, prompt, K_values=(1, 3),
                                    max_new_tokens=8, n_repeats=1)
        text = format_benchmark(out)
        self.assertIn("speedup", text)
        self.assertIn("accept", text)
        self.assertEqual(len(text.strip().splitlines()), 4)  # header + rule + 2 rows


if __name__ == "__main__":
    unittest.main()
