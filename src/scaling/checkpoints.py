"""Checkpoint retention and pinning.

Delete-after-rung eviction stranded Acts 6-10 with no loadable weights (Act 5c
needs a draft/target pair, Act 7 a base model, Act 9 a model to ablate heads in).
The policy here keeps the N most recent checkpoints per rung and never touches a
pinned one.

Canonical pinned set (owner decision, 2026-10-17): 1.5M, 2.6M and 6.3M.
6.3M is chosen over 9.0M deliberately: 9.0M was capped at D=10N (89.9M tokens)
because the corpus ran out, while 6.3M reached D=20N (125.8M tokens). The 6.3M
model is therefore better trained AND cheaper to reproduce, despite being
smaller. It is the better canonical model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PINNED_CHECKPOINTS: dict[str, int] = {
    # rung -> steps retained (D tokens / (batch*context) for the canonical config)
    "1.5M": 7500,
    "2.6M": 12960,
    "6.3M": 30720,
}

_STEP_RE = re.compile(r"checkpoint-step(\d+)\.pt$")


def _resolve(directory: Path, step: int | None = None) -> Path | None:
    """Prefer an explicit step-named file; fall back to the most recent one."""
    if not directory.exists():
        return None
    if step is not None:
        exact = directory / f"checkpoint-step{step:06d}.pt"
        if exact.exists():
            return exact
    candidates = sorted(p for p in directory.glob("checkpoint-step*.pt") if _STEP_RE.match(p.name))
    if candidates:
        return candidates[-1]
    plain = directory / "checkpoint.pt"
    return plain if plain.exists() else None


class CheckpointStore:
    """Retains the `keep_recent` newest checkpoints per rung; pins are exempt."""

    def __init__(self, root: str | Path, *, keep_recent: int = 2) -> None:
        self.root = Path(root)
        self.keep_recent = max(1, int(keep_recent))

    # ── paths ──────────────────────────────────────────────────────────────
    @staticmethod
    def _seed_label(seed: int | str) -> str:
        # Accept both 42 and "seed42" so callers cannot silently produce
        # "seedseed42" directories that eviction then never matches.
        text = str(seed)
        return text if text.startswith("seed") else f"seed{text}"

    def dir_for(self, rung: str, seed: int | str) -> Path:
        return self.root / rung / self._seed_label(seed)

    # ── write ──────────────────────────────────────────────────────────────
    def save(self, rung: str, seed: int | str, step: int, payload: dict[str, Any]) -> Path:
        import torch

        directory = self.dir_for(rung, seed)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"checkpoint-step{step:06d}.pt"
        torch.save(payload, path)
        return path

    # ── pin ────────────────────────────────────────────────────────────────
    def pin(self, rung: str, seed: int | str, step: int) -> None:
        directory = self.dir_for(rung, seed)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "pinned.json"
        pins = json.loads(path.read_text()) if path.exists() else {}
        pins[f"{rung}/{self._seed_label(seed)}"] = int(step)
        path.write_text(json.dumps(pins, indent=2))

    def load_pinned_or_latest(self, rung: str, seed: int | str) -> dict[str, Any]:
        """Load the pinned checkpoint if one exists, else the most recent."""
        import torch

        directory = self.dir_for(rung, seed)
        pins_path = directory / "pinned.json"
        step = None
        if pins_path.exists():
            pins = json.loads(pins_path.read_text())
            step = pins.get(f"{rung}/{self._seed_label(seed)}")
        path = _resolve(directory, step)
        if path is None:
            raise FileNotFoundError(f"no checkpoint for {rung}/{self._seed_label(seed)}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def pinned_steps(self) -> dict[str, int]:
        merged: dict[str, int] = {}
        for pins_file in self.root.glob("*/seed*/pinned.json"):
            try:
                merged.update(json.loads(pins_file.read_text()))
            except json.JSONDecodeError:
                continue
        return merged

    # ── read ───────────────────────────────────────────────────────────────
    def load_pinned(self, rung: str, seed: int | str) -> dict[str, Any]:
        import torch

        pins = json.loads((self.dir_for(rung, seed) / "pinned.json").read_text())
        step = pins[f"{rung}/{self._seed_label(seed)}"]
        path = _resolve(self.dir_for(rung, seed), step)
        if path is None:
            raise FileNotFoundError(f"pinned checkpoint missing for {rung}/seed{seed} step {step}")
        return torch.load(path, map_location="cpu", weights_only=False)

    # ── eviction ───────────────────────────────────────────────────────────
    def evict(self) -> list[Path]:
        """Delete unpinned checkpoints beyond `keep_recent` for each rung."""
        pins = self.pinned_steps()
        removed: list[Path] = []
        for rung_dir in sorted(p for p in self.root.glob("*") if p.is_dir()):
            for seed_dir in sorted(p for p in rung_dir.glob("seed*") if p.is_dir()):
                key = f"{rung_dir.name}/{seed_dir.name}"
                pinned_step = pins.get(key)
                files = sorted(
                    (p for p in seed_dir.glob("checkpoint-step*.pt") if _STEP_RE.match(p.name)),
                    key=lambda p: int(_STEP_RE.match(p.name).group(1)),
                )
                protected: set[Path] = set()
                if pinned_step is not None:
                    protected |= {p for p in files if int(_STEP_RE.match(p.name).group(1)) == pinned_step}
                unpinned = [p for p in files if p not in protected]
                for path in unpinned[: max(0, len(unpinned) - self.keep_recent)]:
                    path.unlink()
                    removed.append(path)
        return removed


__all__ = ["CheckpointStore", "PINNED_CHECKPOINTS"]