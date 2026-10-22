"""The tracked canonical checkpoints must stay loadable and self-consistent.

``checkpoints/canonical/`` holds the three canonical Act 5 models (1.5M / 2.6M /
6.3M) so a reader can regenerate every Act 5 figure without ~3.4 h of retraining.
That only helps if the files are intact and self-describing, so this verifies
the SHA-256 digests, the recorded step counts and losses, and that the weights
actually reconstruct a runnable model.

It deliberately does NOT compare against ``artifacts/``, which is git-ignored and
absent in a fresh clone; these tests must pass for anyone who checks the repo
out.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

CANON = Path("checkpoints/canonical")
INDEX = CANON / "INDEX.json"


def _load_index() -> dict:
    return json.loads(INDEX.read_text(encoding="utf-8"))


class TestCanonicalCheckpoints(unittest.TestCase):
    def test_index_exists_and_lists_the_canonical_rungs(self):
        """The three trained base rungs, plus Act 7's instruction-tuned model.

        6.3M-sft was added so that Act 7's weights are auditable like the
        other three and 10c's SFT-versus-base comparison is reproducible. It is
        a FINE-TUNE of the 6.3M rung, not a fourth base rung, so it is listed
        separately.
        """
        idx = _load_index()
        rungs = [m["rung"] for m in idx["models"]]
        self.assertEqual(rungs, ["1.5M", "2.6M", "6.3M", "6.3M-sft"])
        self.assertIn("tokenizer", idx)
        self.assertIn("architecture", idx)

    def test_the_sft_checkpoint_is_present_and_digested(self):
        import hashlib

        idx = _load_index()
        entry = next(m for m in idx["models"] if m["rung"] == "6.3M-sft")
        path = CANON / entry["file"]
        self.assertTrue(path.exists(), f"{path} missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest, entry["sha256"],
                         "the tracked SFT checkpoint does not match its digest")

    def test_tokenizer_present_and_loadable(self):
        from src.tokenizer.bpe import BPETokenizer

        tok = BPETokenizer.load(CANON / "tokenizer.json")
        self.assertGreater(tok.vocabulary_size, 0)
        # Round-trip on real text: the tokenizer must be usable standalone.
        ids = tok.encode("Beginners BBQ Class Taking Place in Missoula!")
        self.assertEqual(tok.decode(ids), "Beginners BBQ Class Taking Place in Missoula!")

    def test_every_file_matches_its_recorded_digest(self):
        """Catches truncated or hand-edited weight files."""
        for m in _load_index()["models"]:
            with self.subTest(rung=m["rung"]):
                p = CANON / m["file"]
                self.assertTrue(p.exists(), f"missing {p}")
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                self.assertEqual(digest, m["sha256"],
                                 f"{m['rung']} digest mismatch - file was altered")

    def test_weights_reconstruct_a_runnable_model(self):
        """Load each rung and confirm it produces finite logits on real text."""
        import torch

        from src.model.gpt import GPTConfig, MiniGPT
        from src.generation.decoding import generate
        from src.tokenizer.bpe import BPETokenizer

        tok = BPETokenizer.load(CANON / "tokenizer.json")
        ids = tok.encode("Beginners BBQ Class Taking Place in Missoula!")[:16]
        prompt = torch.tensor([ids], dtype=torch.long)

        for m in _load_index()["models"]:
            with self.subTest(rung=m["rung"]):
                payload = torch.load(CANON / m["file"], map_location="cpu",
                                     weights_only=False)
                self.assertEqual(payload["completed_steps"], m["completed_steps"])
                self.assertAlmostEqual(payload["val_loss"], m["val_loss"], places=4)
                self.assertNotIn("optimizer_state", payload,
                                 "canonical export must not carry optimizer state")

                model = MiniGPT(GPTConfig(**payload["model_config"])).eval()
                model.load_state_dict(payload["model_state"])
                self.assertEqual(model.non_embedding_params(), m["non_embedding_params"])

                with torch.no_grad():
                    logits, _, _ = model(prompt)
                self.assertTrue(torch.isfinite(logits).all())
                out = generate(model, prompt, context_length=model.config.context_length,
                               max_new_tokens=8, use_cache=True, do_sample=False)
                self.assertEqual(out.shape[1], prompt.shape[1] + 8)


if __name__ == "__main__":
    unittest.main()
