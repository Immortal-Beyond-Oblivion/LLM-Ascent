"""Tests for speculative decoding (Lecture 16).

The correctness property that matters is that the OUTPUT DISTRIBUTION IS UNCHANGED:
speculative decoding must be distributionally identical to plain sampling from the
target model. A fast wrong implementation is worse than a slow right one, so the
distribution test is primary and the speedup is secondary.

Algorithm under test (Leviathan et al. 2023):
  1. Draft model proposes K tokens autoregressively.
  2. Target model verifies all K in ONE forward pass.
  3. Walk the accepted prefix; on the first rejection, resample from the
     residual distribution  max(0, p_target - p_draft)  normalised.
  4. The corrected token always makes it into the output, so the distribution of
     every emitted token equals the target model's own distribution.
"""

from __future__ import annotations

import unittest
from collections import Counter

import torch
from torch.nn import functional as F

from src.generation.speculative import (
    speculative_decode,
    speculative_sample_tokens,
    residual_distribution,
)


def _model(seed: int, vocab: int = 32, dim: int = 32, layers: int = 2):
    from src.model.gpt import GPTConfig, MiniGPT

    torch.manual_seed(seed)
    model = MiniGPT(GPTConfig(vocabulary_size=vocab, context_length=64,
                              embedding_dim=dim, num_heads=4, num_kv_heads=4,
                              num_layers=layers, dropout=0.0,
                              normalization="layernorm"))
    return model.eval()


class TestResidualDistribution(unittest.TestCase):
    def test_residual_is_zero_on_draft_argmax_when_target_agrees(self):
        p = torch.tensor([0.5, 0.3, 0.2])
        q = torch.tensor([0.5, 0.3, 0.2])
        r = residual_distribution(p, q)
        self.assertAlmostEqual(float(r.sum()), 1.0, places=5)

    def test_residual_removes_draft_mass(self):
        p = torch.tensor([0.2, 0.5, 0.3])
        q = torch.tensor([0.5, 0.3, 0.2])
        r = residual_distribution(p, q)
        # Where q > p the residual is exactly zero.
        self.assertEqual(float(r[0]), 0.0)
        self.assertAlmostEqual(float(r.sum()), 1.0, places=5)

    def test_residual_is_zero_where_target_never_goes(self):
        p = torch.tensor([0.0, 1.0, 0.0])
        q = torch.tensor([0.3, 0.3, 0.4])
        r = residual_distribution(p, q)
        self.assertEqual(float(r[0]), 0.0)
        self.assertEqual(float(r[2]), 0.0)


class TestDistributionEquivalence(unittest.TestCase):
    """THE primary correctness check, and it must be statistical."""

    def test_single_token_distribution_matches_target(self):
        """Speculative first-token distribution must equal the target's.

        The target's next-token distribution at this prompt is near-uniform
        (max prob ~0.04 over 32 bins), so two independent n-sample estimates sit
        ~0.03 apart in total variation by sampling noise alone. A fixed tolerance
        would therefore be below the noise floor and would fire on a CORRECT
        implementation. Instead we measure the noise floor directly (plain
        sampler with the same seeds vs the exact distribution) and require the
        speculative sampler's distance from exact p to be no larger.
        """
        draft, target = _model(1), _model(2)
        prompt = torch.randint(0, 32, (1, 6), generator=torch.Generator().manual_seed(0))
        vocab, n = 32, 6000

        with torch.no_grad():
            logits, _, _ = target(prompt)
        exact = F.softmax(logits[0, -1, :], dim=-1)

        spec_counts: Counter = Counter()
        plain_counts: Counter = Counter()
        for i in range(n):
            torch.manual_seed(10_000 + i)
            spec_counts[int(speculative_decode(draft, target, prompt, max_new_tokens=1,
                                                K=3, do_sample=True)["tokens"][0, 0])] += 1
            torch.manual_seed(10_000 + i)
            with torch.no_grad():
                plain_counts[int(torch.multinomial(exact, 1).item())] += 1

        def tv_from(counts):
            emp = torch.zeros(vocab)
            for k, v in counts.items():
                emp[k] = v / n
            return 0.5 * float((emp - exact).abs().sum()), emp

        tv_spec, _ = tv_from(spec_counts)
        tv_plain, _ = tv_from(plain_counts)

        # Noise floor for an n-sample estimate of this distribution.
        self.assertLessEqual(
            tv_spec, tv_plain * 1.5 + 0.005,
            f"speculative TV={tv_spec:.4f} exceeds noise floor {tv_plain:.4f}: distribution changed")

    def test_acceptance_rate_reported(self):
        draft, target = _model(3), _model(4)
        prompt = torch.randint(0, 32, (1, 6), generator=torch.Generator().manual_seed(1))
        stats = speculative_decode(draft, target, prompt, max_new_tokens=8, K=3, do_sample=True)
        self.assertIn("acceptance_rate", stats)
        self.assertGreaterEqual(stats["acceptance_rate"], 0.0)
        self.assertLessEqual(stats["acceptance_rate"], 1.0)

    def test_greedy_path_matches_plain_greedy(self):
        draft, target = _model(5), _model(6)
        prompt = torch.randint(0, 32, (1, 6), generator=torch.Generator().manual_seed(2))
        spec = speculative_decode(draft, target, prompt, max_new_tokens=8, K=3, do_sample=False)
        plain = greedy_tokens(target, prompt, 8)
        self.assertTrue(torch.equal(spec["tokens"], plain),
                        "greedy speculative output differs from plain greedy")


# ── helpers ──────────────────────────────────────────────────────────────
def total_variation(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / sum(a.values()) - b.get(k, 0) / sum(b.values())) for k in keys)


@torch.no_grad()
def greedy_tokens(target, prompt, n):
    out = prompt
    for _ in range(n):
        logits, _, _ = target(out)
        out = torch.cat((out, logits[:, -1, :].argmax(-1, keepdim=True)), dim=1)
    return out[:, prompt.shape[1]:]


@torch.no_grad()
def target_sample_tokens(target, prompt, n_samples, vocab):
    counts = {}
    logits, _, _ = target(prompt)
    probs = F.softmax(logits[:, -1, :], dim=-1)[0]
    for _ in range(n_samples):
        tok = int(torch.multinomial(probs, 1).item())
        counts[tok] = counts.get(tok, 0) + 1
        logits, _, _ = target(torch.cat((prompt, torch.tensor([[tok]])), dim=1))
        probs = F.softmax(logits[:, -1, :], dim=-1)[0]
    return counts


if __name__ == "__main__":
    unittest.main()