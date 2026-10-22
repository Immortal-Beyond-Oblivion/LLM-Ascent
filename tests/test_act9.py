"""Tests for Act 9: the interpretability instruments.

The first version of ``measure_ce`` compared logits at position i against the
token AT position i instead of the token at i+1, and returned CE ~ 12 for a
model whose real held-out loss is ~2.5. A metric that cannot be wrong is
worth more than a metric that is fast, so the shift is pinned here by a model
that is constructed to be perfect and one that is constructed to be uniform.
"""

from __future__ import annotations

import unittest

import torch


class _NextTokenOracle(torch.nn.Module):
    """Puts all probability mass on the NEXT token: perfect at the task."""

    def __init__(self, vocab: int = 40):
        super().__init__()
        self._probe = torch.nn.Parameter(torch.zeros(1))
        self.vocab = vocab

    def forward(self, token_ids, targets=None, cache=None):
        b, T = token_ids.shape
        logits = torch.full((b, T, self.vocab), -50.0)
        for i in range(T - 1):
            logits[0, i, int(token_ids[0, i + 1])] = 50.0
        return logits, None, None


class _UniformModel(torch.nn.Module):
    """Constant logits: CE should be exactly ln(vocab)."""

    def __init__(self, vocab: int = 40):
        super().__init__()
        self._probe = torch.nn.Parameter(torch.zeros(1))
        self.vocab = vocab

    def forward(self, token_ids, targets=None, cache=None):
        b, T = token_ids.shape
        return torch.zeros(b, T, self.vocab), None, None


class TestHeldOutMetric(unittest.TestCase):
    def test_perfect_next_token_model_scores_near_zero(self):
        from src.eval.act9_interpretability import measure_ce

        seqs = [[1, 2, 3, 4, 5]]
        ce = measure_ce(_NextTokenOracle(), seqs, device="cpu")
        self.assertLess(ce, 1e-4,
                        "the metric is not comparing logits to the NEXT token")

    def test_uniform_model_scores_ln_vocab(self):
        import math

        from src.eval.act9_interpretability import measure_ce

        vocab = 40
        ce = measure_ce(_UniformModel(vocab), [[1, 2, 3, 4, 5]], device="cpu")
        self.assertAlmostEqual(ce, math.log(vocab), places=4)

    def test_the_shift_is_what_the_oracle_test_detects(self):
        """A shifted metric scores ln(V) on the oracle model, not 0.

        The uniform test alone cannot detect the shift (it returns ln(V)
        either way), which is why the oracle test exists.
        """
        import math

        from src.eval.act9_interpretability import measure_ce

        vocab = 40
        shifted = measure_ce(_NextTokenOracle(vocab), [[1, 2, 3, 4, 5]],
                              device="cpu")
        self.assertNotAlmostEqual(shifted, math.log(vocab), places=2)


class TestHeadAblationHook(unittest.TestCase):
    """The ablation must be a real intervention, and inert by default."""

    def _model(self):
        from src.model.gpt import GPTConfig, MiniGPT

        torch.manual_seed(0)
        config = GPTConfig(vocabulary_size=40, context_length=16, embedding_dim=32,
                           num_heads=4, num_layers=2, dropout=0.0)
        model = MiniGPT(config)
        model.eval()
        return model

    def test_default_state_is_inert(self):
        model = self._model()
        for attn in (b.attention for b in model.blocks):
            self.assertEqual(attn.ablate_heads, set())
            self.assertFalse(attn.record_attention)
            self.assertIsNone(attn.last_attn)

    def test_ablation_changes_the_output(self):
        model = self._model()
        x = torch.randint(0, 40, (1, 16))
        with torch.no_grad():
            before, _, _ = model(x)
        model.blocks[0].attention.ablate_heads = {1}
        with torch.no_grad():
            after, _, _ = model(x)
        self.assertFalse(torch.allclose(before, after),
                         "ablating a head did not change the output at all")

    def test_ablation_is_confined_to_the_chosen_head(self):
        """Ablating head 1 must not equal ablating head 0."""
        model = self._model()
        x = torch.randint(0, 40, (1, 16))
        outs = []
        for head in (0, 1):
            model.blocks[0].attention.ablate_heads = {head}
            with torch.no_grad():
                outs.append(model(x)[0].clone())
        self.assertFalse(torch.allclose(outs[0], outs[1]))

    def test_recorded_attention_rows_sum_to_one(self):
        model = self._model()
        for b in model.blocks:
            b.attention.record_attention = True
        x = torch.randint(0, 40, (1, 16))
        with torch.no_grad():
            model(x)
        a = model.blocks[0].attention.last_attn
        self.assertIsNotNone(a)
        sums = a.sum(-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))


class TestRolloutAndLens(unittest.TestCase):
    def test_rollout_matrices_are_row_stochastic(self):
        from src.eval.act9_interpretability import rollout_matrices

        torch.manual_seed(0)
        attns = [torch.softmax(torch.randn(6, 6), dim=-1) for _ in range(3)]
        rolls = rollout_matrices(attns)
        self.assertEqual(len(rolls), 3)
        for r in rolls:
            sums = r.sum(-1)
            self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-4))

    def test_rollout_of_a_single_layer_is_that_layer(self):
        from src.eval.act9_interpretability import rollout_matrices

        a = torch.softmax(torch.randn(4, 4), dim=-1)
        rolls = rollout_matrices([a])
        self.assertTrue(torch.allclose(rolls[0], a, atol=1e-5))

    def test_subsequence_search(self):
        from src.eval.act9_interpretability import _find_subsequence

        self.assertEqual(_find_subsequence([1, 2, 3, 4], [3, 4]), 2)
        self.assertEqual(_find_subsequence([1, 2, 3, 4], [2, 3]), 1)
        self.assertEqual(_find_subsequence([1, 2, 3, 4], [9]), -1)
        self.assertEqual(_find_subsequence([1, 2], []), -1)


class TestAct9ResultsIntegrity(unittest.TestCase):
    """The tracked Act 9 results must be self-consistent and sane."""

    def _load(self):
        import json
        from pathlib import Path

        p = Path("results/v2/act9/interpretability.json")
        if not p.exists():
            self.skipTest("Act 9 results not present")
        return json.loads(p.read_text())

    def test_held_out_loss_is_in_a_plausible_range(self):
        """A held-out CE of 12 (perplexity 1.6e5) is the signature of the
        unshifted-target bug; a real model scores ~2.5-3."""
        d = self._load()
        ce = d["held_out"]["intact_ce_nats"]
        self.assertGreater(ce, 1.0)
        self.assertLess(ce, 5.0)

    def test_z_scores_are_consistent_with_the_deltas(self):
        d = self._load()
        a = d["9a_head_ablation"]
        rows = a["per_head"]
        deltas = [r["delta_ce"] for r in rows]
        mean = sum(deltas) / len(deltas)
        var = sum((x - mean) ** 2 for x in deltas) / (len(deltas) - 1)
        sd = var ** 0.5
        for r in rows:
            self.assertAlmostEqual(r["z_score"], (r["delta_ce"] - mean) / sd,
                                   places=6)

    def test_the_null_is_the_null(self):
        """A random head must cost less than the worst real head, and the
        random-4 combination must cost less than the top-4 combination."""
        d = self._load()
        a = d["9a_head_ablation"]
        self.assertLess(a["null_single_head_mean_delta_ce"],
                        max(r["delta_ce"] for r in a["per_head"]))
        self.assertLess(a["random_4head_mean_delta_ce"],
                        a["top4_combination_delta_ce"])

    def test_rollout_hit_rate_is_zero_or_a_real_rate(self):
        d = self._load()
        b = d["9b_rollout"]
        self.assertIsNotNone(b["rollout_hit_rate"])
        self.assertGreaterEqual(b["rollout_hit_rate"], 0.0)
        self.assertLessEqual(b["rollout_hit_rate"], 1.0)
        self.assertLessEqual(b["rollout_hit_rate"], b["raw_attention_hit_rate"] + 1)

    def test_lens_ranks_are_within_the_vocabulary(self):
        d = self._load()
        c = d["9c_logit_lens"]
        for r in c["median_rank_of_asserted_number_per_layer"]:
            self.assertGreaterEqual(r, 0)
            self.assertLess(r, c["vocabulary_size"])


if __name__ == "__main__":
    unittest.main()
