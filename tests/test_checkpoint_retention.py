"""Tests for checkpoint retention and pinning policy.

Delete-after-rung eviction stranded Acts 6-10 with no loadable weights. The
replacement keeps the N most recent checkpoints per rung and never touches a
pinned checkpoint. An untested policy is not a policy, so the pin behaviour is
pinned here explicitly.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.scaling.checkpoints import (
    CheckpointStore,
    PINNED_CHECKPOINTS,
    _resolve,
)


class TestCheckpointRetentionAndPinning(unittest.TestCase):
    def test_pinned_checkpoint_survives_eviction(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = CheckpointStore(root, keep_recent=2)
            # Write 5 sequential checkpoints for one rung, pinning the middle one.
            for step in (100, 200, 300, 400, 500):
                store.save("0.8M", "seed42", step, {"model": step})
            store.pin("0.8M", "seed42", step=300)

            store.evict()

            remaining = sorted(p.name for p in (root / "0.8M" / "seed42").glob("checkpoint*.pt"))
            # Pinned checkpoint must survive.
            self.assertIn("checkpoint-step000300.pt", remaining)
            # Most recent N-1 pinned-aware survivors retained.
            self.assertIn("checkpoint-step000500.pt", remaining)
            self.assertIn("checkpoint-step000400.pt", remaining)

    def test_eviction_drops_unpinned_oldest(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = CheckpointStore(root, keep_recent=2)
            for step in (100, 200, 300, 400, 500):
                store.save("0.8M", "seed42", step, {"model": step})

            store.evict()

            remaining = sorted(p.name for p in (root / "0.8M" / "seed42").glob("checkpoint*.pt"))
            # keep_recent=2 -> only 500 and 400 survive.
            self.assertEqual(remaining, ["checkpoint-step000400.pt", "checkpoint-step000500.pt"])

    def test_pin_is_recorded_in_pin_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = CheckpointStore(root, keep_recent=2)
            store.save("0.8M", "seed42", 100, {"model": 1})
            store.pin("0.8M", "seed42", step=100)
            pins = json.loads((root / "0.8M" / "seed42" / "pinned.json").read_text())
            self.assertIn("0.8M/seed42", pins)
            self.assertEqual(pins["0.8M/seed42"], 100)

    def test_load_returns_pinned_payload(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = CheckpointStore(root, keep_recent=2)
            store.save("6.3M", "seed42", 500, {"model": "six-three"})
            store.pin("6.3M", "seed42", step=500)
            store.evict()
            payload = store.load_pinned("6.3M", "seed42")
            self.assertEqual(payload["model"], "six-three")

    def test_canonical_pinned_set_declared(self) -> None:
        # Canonical set for Acts 5c/7/9 per owner decision: 1.5M, 2.6M, 6.3M.
        self.assertEqual(sorted(PINNED_CHECKPOINTS), ["1.5M", "2.6M", "6.3M"])


class TestLegacyPathResolution(unittest.TestCase):
    def test_resolve_prefers_step_named_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = CheckpointStore(root, keep_recent=2)
            store.save("9.0M", "seed42", 500, {"model": "nine"})
            path = _resolve(root / "9.0M" / "seed42", 500)
            self.assertTrue(path.exists())
            self.assertIn("step000500", path.name)


if __name__ == "__main__":
    unittest.main()