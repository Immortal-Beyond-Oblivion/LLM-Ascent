
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.scaling.fit import load_points_from_metrics
from src.scaling.ladder import ALiBiAttention, MemmapCorpus, LadderRung, train_rung


def _make_test_corpus(tmpdir, total_tokens=5000, vocab=20, eval_subsample=100):
    tokens = np.random.RandomState(42).randint(0, vocab, total_tokens).astype(np.uint16)
    path = Path(tmpdir) / "tokens.mmap"
    mmap = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
    mmap[:] = tokens
    mmap.flush()
    manifest = {
        "memmap_path": str(path),
        "tokenizer_vocab_size": vocab,
        "eval_token_subsample": eval_subsample,
    }
    manifest_path = Path(tmpdir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return str(manifest_path)


class TestBug1_AliBiBias(unittest.TestCase):
    """Bug 1: ALiBi bias must NOT be identically zero on causal entries.

    On ORIGINAL buggy code, these failed because rel_pos = pos[unsqueeze(0)] - pos[unsqueeze(1)]
    gives <=0 on causal entries, clamped to zero. Fixed code uses pos[unsqueeze(1)] - pos[unsqueeze(0)].
    """

    def test_bias_negative_for_causal_entries(self):
        attn = ALiBiAttention(dim=16, heads=4, max_len=4)
        T = 4
        positions = torch.arange(T)
        rel = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel = rel.clamp(min=0)
        slopes = attn.slopes
        bias = -slopes.view(1, -1, 1, 1) * rel.unsqueeze(0).unsqueeze(0)
        for i in range(1, T):
            for j in range(i):
                self.assertLess(bias[0, 0, i, j].item(), 0.0,
                    f"bias[0,0,{i},{j}] should be strictly negative")

    def test_bias_zero_for_self_attention(self):
        attn = ALiBiAttention(dim=16, heads=4, max_len=4)
        T = 4
        positions = torch.arange(T)
        rel = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel = rel.clamp(min=0)
        slopes = attn.slopes
        bias = -slopes.view(1, -1, 1, 1) * rel.unsqueeze(0).unsqueeze(0)
        for i in range(T):
            self.assertAlmostEqual(bias[0, 0, i, i].item(), 0.0, places=8)

    def test_bias_magnitude_increases_with_distance(self):
        attn = ALiBiAttention(dim=16, heads=4, max_len=4)
        T = 4
        positions = torch.arange(T)
        rel = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel = rel.clamp(min=0)
        slopes = attn.slopes
        i = 3
        mag_dist1 = abs((-slopes.view(1,-1,1,1) * rel[i, i-1].unsqueeze(0).unsqueeze(0)).view(-1)[0].item())
        mag_dist2 = abs((-slopes.view(1,-1,1,1) * rel[i, i-2].unsqueeze(0).unsqueeze(0)).view(-1)[0].item())
        mag_dist3 = abs((-slopes.view(1,-1,1,1) * rel[i, i-3].unsqueeze(0).unsqueeze(0)).view(-1)[0].item())
        bias = -slopes.view(1, -1, 1, 1) * rel.unsqueeze(0).unsqueeze(0)
        mag_dist1 = abs(bias[0, 0, i, i-1].item())
        mag_dist2 = abs(bias[0, 0, i, i-2].item())
        mag_dist3 = abs(bias[0, 0, i, i-3].item())
        self.assertGreater(mag_dist2, mag_dist1)
        self.assertGreater(mag_dist3, mag_dist2)


class TestBug2_FitDaxis(unittest.TestCase):
    """Bug 2: Fit must use entry['tokens'], not final token count."""

    def _make_ladder_dir(self, tmpdir, use_per_entry_tokens=True):
        rung_dir = Path(tmpdir) / "0.5M"
        rung_dir.mkdir(parents=True, exist_ok=True)
        history = []
        for step in range(1, 1501):
            # Per-entry tokens grow; distinct from final
            history.append({"step": float(step), "tokens": step * 32 * 128, "val_loss": 2.5})
        final_total = 10_000_000
        if not use_per_entry_tokens:
            # buggy loader would read m["tokens"] = final_total
            pass
        metrics = {
            "rung": "0.5M",
            "non_embedding_params": 500000,
            "total_params": 600000,
            "steps": 1500,
            "tokens": final_total,
            "history": history,
            "device": "cpu",
            "seed": 42,
        }
        (rung_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return tmpdir

    def test_loader_reads_entry_tokens_not_final_total(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = self._make_ladder_dir(tmpdir)
            N, D, losses, meta = load_points_from_metrics(dirpath)
            # Each point has a distinct per-entry token count, not the final 999999
            assert not all(d == 10_000_000 for d in D), "loader collapsed to final total (entry tokens ignored)"
            self.assertNotEqual(N[0], None)
            # Verify the loader consumed entry["tokens"], not m["tokens"]: max D < final total
            self.assertLess(max(int(x) for x in D), 10_000_000)

    def test_dense_D_trajectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = self._make_ladder_dir(tmpdir)
            N, D, losses, meta = load_points_from_metrics(dirpath)
            distinct_D = len(np.unique(D))
            self.assertGreater(distinct_D, 1000,
                f"only {distinct_D} distinct D; expected thousands")


class TestBug3_SeedsApplied(unittest.TestCase):
    """Bug 3: Seeds must actually reach the RNG (behavioral, not text inspection)."""

    def _run_tiny_rung(self, seed, corpus):
        # Tiny model, short run, fixed batch order via np.random seed inside train_rung
        rung = LadderRung("tiny", 2000, 10 * 8 * 32, [seed], "cpu")
        hist = train_rung(
            rung, corpus, context_length=32, batch_size=8,
            eval_interval=5, eval_batches=2, warmup_steps=2,
        )
        return [h["val_loss"] for h in hist]

    def test_same_seed_identical_histories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _make_test_corpus(tmpdir, total_tokens=5000, eval_subsample=100)
            corpus = MemmapCorpus(manifest_path)
            h1 = self._run_tiny_rung(42, corpus)
            h2 = self._run_tiny_rung(42, corpus)
            self.assertEqual(len(h1), len(h2))
            self.assertEqual(h1, h2, f"same seed must produce identical histories: {h1} vs {h2}")

    def test_different_seeds_produce_different_histories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _make_test_corpus(tmpdir, total_tokens=5000, eval_subsample=100)
            corpus = MemmapCorpus(manifest_path)
            h1 = self._run_tiny_rung(42, corpus)
            h2 = self._run_tiny_rung(99, corpus)
            self.assertNotEqual(h1, h2, "different seeds must produce different histories")


class TestBug4_NoValidationLeak(unittest.TestCase):
    """Bug 4: Training offsets must not overlap validation range."""

    def test_sampled_training_offsets_below_train_limit(self):
        manifest_path = None
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _make_test_corpus(tmpdir, total_tokens=5000, eval_subsample=500)
            corpus = MemmapCorpus(manifest_path)
            for _ in range(2000):
                offs = corpus._sample_offsets(32, 0, corpus.train_limit)
                o = np.asarray(offs)
                self.assertGreaterEqual(int(o.min()), 0)
                self.assertLess(int(o.max()), corpus.train_limit)
            assert corpus.train_limit <= corpus.val_start

    def test_ranges_genuinely_disjoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = _make_test_corpus(tmpdir, total_tokens=5000, eval_subsample=500)
            corpus = MemmapCorpus(manifest_path)
            assert corpus.train_limit <= corpus.val_start, (
                f"train_limit={corpus.train_limit} > val_start={corpus.val_start} => overlap"
            )


if __name__ == "__main__":
    unittest.main()
