"""Discover and cache trained checkpoints for the web interface.

Clients only ever send a model *id* (for example ``tick-204/bpe``); ids are
resolved through a server-side discovery map, so no client-supplied path is
ever opened.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from src.model import GPTConfig, MiniGPT
from src.tokenizer import tokenizer_from_checkpoint
from src.tokenizer.bpe import BPETokenizer
from src.tokenizer.character import CharacterTokenizer


@dataclass
class LoadedModel:
    id: str
    model: MiniGPT
    tokenizer: object
    info: dict


class ModelRegistry:
    """Finds model checkpoints under one or more roots and keeps a few in memory.

    Two layouts are supported, because the project has two:

    * the **training** layout, ``<root>/<rung>/<seed>/checkpoint.pt``, written
      by the training loop with the optimizer state and the training history
      alongside the weights;
    * the **canonical** layout, ``<root>/<rung>/model.pt`` with the tokenizer
      in a sibling ``tokenizer.json``, written by the exporter that pins the
      models the report is about. These carry the model weights only.

    The first version of this registry knew only the training layout and read
    ``checkpoint["model_config"]``, which the training checkpoints do not
    carry. Every model therefore failed to load and the interface offered an
    empty, disabled model list: it had never been opened in a browser.
    """

    #: The file names a checkpoint may have, in preference order.
    CHECKPOINT_NAMES = ("checkpoint.pt", "model.pt")

    def __init__(self, artifacts_dir: str | Path, *, cache_size: int = 2,
                 device: str = "cpu",
                 extra_roots: Sequence[tuple[str, str | Path]] = ()) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self.artifacts_dir = Path(artifacts_dir).resolve()
        self.device = device
        self.cache_size = cache_size
        # (id prefix, root) pairs searched after the primary root. The prefix
        # keeps the ids of two roots from colliding.
        self.extra_roots = [(prefix, Path(path).resolve())
                            for prefix, path in extra_roots]
        # Re-entrant: handlers hold this lock while generating and call get() inside it.
        self.lock = threading.RLock()
        self._cache: OrderedDict[str, LoadedModel] = OrderedDict()
        self._info: dict[tuple[str, float], dict] = {}

    def discover(self) -> dict[str, Path]:
        """Map model id -> checkpoint path for every checkpoint below the roots."""
        found: dict[str, Path] = {}
        for prefix, root in [(("", self.artifacts_dir))] + self.extra_roots:
            if not root.is_dir():
                continue
            for name in self.CHECKPOINT_NAMES:
                for path in sorted(root.rglob(name)):
                    resolved = path.resolve()
                    if not resolved.is_relative_to(root):
                        continue  # symlink pointing outside the allow-listed root
                    relative = path.parent.relative_to(root)
                    if not relative.parts:
                        continue
                    model_id = f"{prefix}{relative.as_posix()}"
                    found.setdefault(model_id, resolved)
        return found

    def _roots_for(self, path: Path) -> list[Path]:
        """Directories a sibling tokenizer may live in, nearest first."""
        candidates = [path.parent, path.parent.parent]
        for _prefix, root in self.extra_roots:
            candidates.append(root)
            candidates.append(root.parent)
        return candidates

    def _tokenizer_for(self, path: Path, checkpoint: dict):
        """The tokenizer a checkpoint was trained with.

        A checkpoint may carry its tokenizer inline, or the canonical layout
        may keep it in a sibling ``tokenizer.json``.
        """
        if checkpoint.get("tokenizer") is not None:
            return tokenizer_from_checkpoint(checkpoint)
        for directory in self._roots_for(path):
            candidate = directory / "tokenizer.json"
            if not candidate.exists():
                continue
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if "merges" in payload:
                return BPETokenizer(list(payload["vocabulary"]),
                                    [tuple(pair) for pair in payload["merges"]])
            return CharacterTokenizer(payload["vocabulary"])
        if checkpoint.get("vocabulary") is not None:
            return CharacterTokenizer(checkpoint["vocabulary"])
        raise KeyError(
            "no tokenizer: the checkpoint carries none inline and no "
            "tokenizer.json was found beside it")

    def _load(self, model_id: str, path: Path) -> LoadedModel:
        # Only locally produced checkpoints under the allow-listed roots reach
        # this call.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "model_config" not in checkpoint:
            raise KeyError(
                "'model_config': this checkpoint does not carry a model "
                "configuration, so it cannot be served")
        config = GPTConfig(**checkpoint["model_config"])
        model = MiniGPT(config).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        tokenizer = self._tokenizer_for(path, checkpoint)
        kind = "bpe" if hasattr(tokenizer, "merges") else "character"
        info = {
            "id": model_id,
            "tokenizer": kind,
            "vocabulary_size": tokenizer.vocabulary_size,
            "parameters": model.parameter_count(),
            "context_length": config.context_length,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "embedding_dim": config.embedding_dim,
            "normalization": config.normalization,
            "completed_steps": int(checkpoint.get("completed_steps", 0)),
        }
        return LoadedModel(model_id, model, tokenizer, info)

    def get(self, model_id: str) -> LoadedModel:
        """Return a loaded model; raises ``KeyError`` for unknown ids."""
        with self.lock:
            if model_id in self._cache:
                self._cache.move_to_end(model_id)
                return self._cache[model_id]
            found = self.discover()
            if model_id not in found:
                raise KeyError(model_id)
            loaded = self._load(model_id, found[model_id])
            self._cache[model_id] = loaded
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return loaded

    def list_models(self) -> list[dict]:
        """Metadata for every discoverable checkpoint (a broken file reports an error).

        The models that load come first: the training checkpoints in
        ``artifacts/`` are training artefacts rather than servable models, so
        listing eight of them above the four that work would bury the point of
        the dropdown. The failures are still reported, because a checkpoint
        that cannot be loaded is exactly the thing that should stay visible.
        """
        models = []
        with self.lock:
            for model_id, path in self.discover().items():
                key = (model_id, path.stat().st_mtime)
                if key not in self._info:
                    try:
                        self._info[key] = self.get(model_id).info
                    except Exception as error:  # noqa: BLE001 - surface any load failure to the UI
                        self._info[key] = {"id": model_id, "error": f"{type(error).__name__}: {error}"}
                models.append(self._info[key])
        models.sort(key=lambda m: (bool(m.get("error")),
                                   -m.get("parameters", 0),
                                   m["id"]))
        return models
