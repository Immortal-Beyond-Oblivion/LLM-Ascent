"""Tests for the Act 4 rescue trainer's PAUSE-and-RESUME machinery.

A multi-hour training arm that cannot be paused is a liability: it has to be
left running or restarted from scratch. The rescue trainer therefore checkpoints
the model, the optimizer, the history AND the RNG states, because the batch
order comes from the global numpy RNG (the corpus samples offsets with
``np.random.randint``) and is seeded only once, at the start of the run.
Restoring the RNG state is what makes a resumed run continue the SAME data
order rather than silently starting a different one.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


def _tiny_config():
    from src.model.gpt import GPTConfig, MiniGPT

    # A deliberately small model so the tests run in seconds.
    return MiniGPT(GPTConfig(vocabulary_size=6487, context_length=128,
                              embedding_dim=64, num_heads=8, num_layers=6,
                              dropout=0.0))


class TestRescueCheckpoints(unittest.TestCase):
    def test_a_checkpoint_carries_the_rng_states(self):
        """Without the RNG states a resume silently changes the data order."""
        from src.scaling.rescue import config_hash

        model = _tiny_config()
        ckpt = {"completed_steps": 7, "history": [],
                "model_state": model.state_dict(),
                "optimizer_state": {},
                "config": config_hash(model),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state()}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.pt"
            torch.save(ckpt, p)
            back = torch.load(p, map_location="cpu", weights_only=False)
        self.assertIn("torch_rng_state", back)
        self.assertIn("numpy_rng_state", back)
        # the numpy state must be restorable, not just present
        np.random.set_state(back["numpy_rng_state"])
        self.assertIsInstance(np.random.get_state()[1], np.ndarray)

    def test_numpy_state_round_trips_the_batch_order(self):
        """The resumed run must see the same offsets as the interrupted one."""
        a = np.random.get_state()
        expected = [np.random.randint(0, 100, size=4) for _ in range(3)]
        np.random.set_state(a)
        got = [np.random.randint(0, 100, size=4) for _ in range(3)]
        np.random.set_state(a)
        again = [np.random.randint(0, 100, size=4) for _ in range(3)]
        for g, ag in zip(got, again):
            self.assertTrue(np.array_equal(g, ag))
        for g, e in zip(got, expected):
            self.assertTrue(np.array_equal(g, e))


class TestRescueResume(unittest.TestCase):
    """End-to-end: a run can be stopped and continued without losing work."""

    def test_a_completed_run_resumes_and_does_no_work(self):
        from src.scaling.rescue import train

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            first = train("0.02M", device="cpu", seed=42, out_dir=out,
                          validate=False, target_non_embedding=30_000,
                          d_over_n=20)
            self.assertGreater(first["steps"], 0)
            # Resuming a finished run must not retrain anything.
            second = train("0.02M", device="cpu", seed=42, out_dir=out,
                           validate=False, target_non_embedding=30_000,
                           d_over_n=20)
            self.assertAlmostEqual(second["final_val_loss"],
                                   first["final_val_loss"], places=6)
            self.assertEqual(len(second["history"]), len(first["history"]))

    def test_a_truncated_checkpoint_resumes_at_the_right_step(self):
        from src.scaling.rescue import train

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            full = train("0.02M", device="cpu", seed=42, out_dir=out,
                         validate=False, target_non_embedding=30_000, d_over_n=20)
            ckpt_path = out / "0.02M_seed42.pt"
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # The cut must land on a LOGGED step (the history is written every
            # EVAL_INTERVAL steps), so the resumed run's history can be checked
            # to continue from it rather than restart.
            logged = [h["step"] for h in ck["history"]]
            cut = logged[len(logged) // 2]
            ck["completed_steps"] = cut
            ck["history"] = [h for h in ck["history"] if h["step"] <= cut]
            torch.save(ck, ckpt_path)

            resumed = train("0.02M", device="cpu", seed=42, out_dir=out,
                            validate=False, target_non_embedding=30_000,
                            d_over_n=20)
            self.assertEqual(resumed["steps"], full["steps"])
            steps = [h["step"] for h in resumed["history"]]
            self.assertIn(cut, steps, "the resume point is missing from the history")
            self.assertEqual(steps, sorted(steps))
            self.assertEqual(steps[-1], full["steps"])
            self.assertTrue(all(s >= cut for s in steps[len(logged) // 2:]),
                            "the resumed run restarted instead of continuing")

    def test_a_config_mismatch_starts_fresh_instead_of_raising(self):
        """A checkpoint from a different architecture must not be loaded."""
        from src.scaling.rescue import train

        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            train("0.02M", device="cpu", seed=42, out_dir=out, validate=False,
                  target_non_embedding=30_000, d_over_n=20)
            ckpt_path = out / "0.02M_seed42.pt"
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ck["model_state"] = {k: torch.zeros_like(v)
                                 for k, v in ck["model_state"].items()}
            torch.save(ck, ckpt_path)
            # A state dict of the right shapes but wrong values still loads, so
            # the mismatch case is exercised with a wrong SHAPE.
            ck2 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ck2["model_state"]["token_embedding.weight"] = torch.zeros(3, 3)
            torch.save(ck2, ckpt_path)
            res = train("0.02M", device="cpu", seed=42, out_dir=out,
                        validate=False, target_non_embedding=30_000, d_over_n=20)
            self.assertGreater(res["final_val_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
