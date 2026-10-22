"""Guard: the test suite must never modify tracked evidence under ``results/``.

``results/`` is the project's ledger of measured outcomes. If a test run can
rewrite it, then ``git checkout results/`` followed by a test run silently
changes versioned evidence, and because the change is a timing field rather
than a number anyone would look at, it can reach a commit unnoticed.

This guard caught exactly that: ``tests/test_ngram.py`` called
``run_ngram_experiment`` with no ``output_root``, and the function hard-coded
``Path("results")``, so every suite run rewrote
``results/act0-ngram/o2-laplace-seed42/metrics.json``. It failed loudly here.

The test invokes the real entry point the suite uses and asserts the tracked
tree is byte-identical afterwards.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"


def _fingerprint(root: Path) -> dict[str, str]:
    """SHA-256 of every file under ``root``, keyed by relative path."""
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class TestResultsAreAppendOnly(unittest.TestCase):
    def test_ngram_experiment_does_not_write_into_results(self):
        from src.ngram.run_comparison import run_ngram_experiment

        before = _fingerprint(RESULTS)
        self.assertTrue(before, "results/ is missing or empty; guard cannot run")

        text = "to be or not to be " * 50
        train_text = text[:int(len(text) * 0.9)]
        val_text = text[int(len(text) * 0.9):]

        with tempfile.TemporaryDirectory() as tmp:
            run_ngram_experiment(
                order=2,
                smoothing="laplace",
                train_text=train_text,
                validation_text=val_text,
                seed=42,
                output_root=tmp,
            )

        after = _fingerprint(RESULTS)
        self.assertEqual(
            before, after,
            "the n-gram experiment modified tracked files under results/:\n"
            + "\n".join(
                f"  {k}: {before.get(k, '<new>')} -> {after.get(k, '<deleted>')}"
                for k in sorted(set(before) | set(after))
                if before.get(k) != after.get(k)
            ),
        )

    def test_no_source_file_hardcodes_a_results_write(self):
        """No module may build a results path from a hard-coded 'results'.

        The bug this guards against was ``Path("results") / act / name`` baked
        into the writer, so every caller — including tests — inherited it.
        Writers must take the root as a parameter.
        """
        offenders: list[str] = []
        for p in sorted((REPO_ROOT / "src").rglob("*.py")):
            for lineno, line in enumerate(
                p.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if 'Path("results")' in line or "Path('results')" in line:
                    offenders.append(f"{p.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
        self.assertEqual(
            offenders, [],
            "these modules hard-code a results/ path; make the root a parameter:\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
