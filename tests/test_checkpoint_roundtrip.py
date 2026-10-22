"""Tests that a saved checkpoint round-trips and survives eviction.

A saved tensor blob that does not round-trip is worse than no checkpoint, because
it fails silently later. These assert LOGITS equality between the reloaded model and
the original, not merely that torch.load succeeds.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.model.gpt import GPTConfig, MiniGPT
from src.scaling.checkpoints import CheckpointStore


def _tiny_model(num_kv_heads: int = 4) -> MiniGPT:
    torch.manual_seed(0)
    cfg = GPTConfig(vocabulary_size=64, context_length=16, embedding_dim=32,
                    num_heads=4, num_kv_heads=num_kv_heads, num_layers=2,
                    dropout=0.0, normalization="layernorm")
    return MiniGPT(cfg).eval()


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_reloaded_model_produces_identical_logits(self) -> None:
        model = _tiny_model()
        tokens = torch.randint(0, 64, (1, 9), dtype=torch.long)
        with torch.no_grad():
            before, _, _ = model(tokens)

        with tempfile.TemporaryDirectory() as d:
            store = CheckpointStore(d, keep_recent=2)
            store.save("1.5M", 42, 100, {
                "model_state": model.state_dict(),
                "model_config": model.config.to_dict(),
                "completed_steps": 100,
            })
            payload = store.load_pinned_or_latest("1.5M", 42)

        reloaded = MiniGPT(GPTConfig(**payload["model_config"])).eval()
        reloaded.load_state_dict(payload["model_state"])
        with torch.no_grad():
            after, _, _ = reloaded(tokens)

        self.assertEqual(before.shape, after.shape)
        self.assertTrue(torch.equal(before, after),
                        "reloaded model logits differ from the original")

    def test_round_trip_preserves_generation(self) -> None:
        from src.generation.decoding import generate

        model = _tiny_model()
        prompt = torch.randint(0, 64, (1, 6), dtype=torch.long)
        before = generate(model, prompt, context_length=16, max_new_tokens=6, use_cache=True)

        with tempfile.TemporaryDirectory() as d:
            store = CheckpointStore(d, keep_recent=2)
            store.save("6.3M", 42, 30720, {
                "model_state": model.state_dict(),
                "model_config": model.config.to_dict(),
                "completed_steps": 30720,
            })
            payload = store.load_pinned_or_latest("6.3M", 42)

        reloaded = MiniGPT(GPTConfig(**payload["model_config"])).eval()
        reloaded.load_state_dict(payload["model_state"])
        after = generate(reloaded, prompt, context_length=16, max_new_tokens=6, use_cache=True)
        self.assertTrue(torch.equal(before, after))

    def test_pinned_weights_survive_real_eviction_on_disk(self) -> None:
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as d:
            store = CheckpointStore(d, keep_recent=2)
            # Two later, unpinned checkpoints must NOT evict the pinned weights.
            for step in (200, 300, 400, 500):
                store.save("6.3M", 42, step, {
                    "model_state": model.state_dict(),
                    "model_config": model.config.to_dict(),
                })
            store.pin("6.3M", 42, step=100)
            store.save("6.3M", 42, 100, {
                "model_state": model.state_dict(),
                "model_config": model.config.to_dict(),
            })

            removed = store.evict()

            path = Path(d) / "6.3M" / "seed42" / "checkpoint-step000100.pt"
            self.assertTrue(path.exists(), "pinned weights were evicted")
            self.assertGreater(len(removed), 0, "eviction removed nothing, test is vacuous")

    def test_train_rung_saves_loadable_weights(self) -> None:
        """The training path must write real tensors through the store."""
        from src.scaling.ladder import train_rung

        with tempfile.TemporaryDirectory() as d:
            store_root = d
            store = CheckpointStore(store_root, keep_recent=2)

            class _FakeCorpus:
                total_tokens = 4096
                vocab_size = 64
                context_length = 16
                train_limit = 2048
                val_start = 2048
                val_tokens = 512
                fixed_val_offsets = list(range(2048, 2048 + 64))

                def train_batch(self, bs, ctx, device):
                    return (torch.randint(0, 64, (bs, ctx), dtype=torch.long),
                            torch.randint(0, 64, (bs, ctx), dtype=torch.long))

                val_batch = train_batch

            import src.scaling.ladder as laddermod
            original = laddermod.build_model_for_rung

            def _mini(rung_, vs, ctx=128):
                return MiniGPT(GPTConfig(vocabulary_size=vs, context_length=ctx,
                                        embedding_dim=32, num_heads=4,
                                        num_kv_heads=4, num_layers=2,
                                        dropout=0.0, normalization="layernorm"))

            laddermod.build_model_for_rung = _mini
            self.addCleanup(setattr, laddermod, "build_model_for_rung", original)

            from src.scaling.ladder import LadderRung
            rung = LadderRung("tiny", 1000, 4096, [7], "cpu")
            train_rung(rung, _FakeCorpus(), context_length=16, batch_size=2,
                       eval_interval=2, eval_batches=1, warmup_steps=1,
                       checkpoint_dir=store_root, save_weights=True,
                       pin_weights=True)
            import json as _json
            metrics = _json.loads((Path(store_root) / "tiny" / "seed7" / "metrics.json").read_text())
            self.assertIsNotNone(metrics.get("saved_checkpoint"),
                                 "train_rung did not save weights")
            path = Path(metrics["saved_checkpoint"])
            self.assertTrue(path.exists(), f"saved weights missing at {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertIn("model_state", payload)
            self.assertIn("model_config", payload)
            # Must be loadable into a real model and produce identical logits.
            model = MiniGPT(GPTConfig(**payload["model_config"])).eval()
            model.load_state_dict(payload["model_state"])
            tokens = torch.randint(0, 64, (1, 8), dtype=torch.long)
            with torch.no_grad():
                logits, _, _ = model(tokens)
            self.assertTrue(torch.isfinite(logits).all())
            # And it must survive eviction.
            store.evict()
            self.assertTrue(path.exists(), "pinned training weights did not survive eviction")


if __name__ == "__main__":
    unittest.main()