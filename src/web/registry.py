"""Discover and cache trained checkpoints for the web interface.

Clients only ever send a model *id* (for example ``tick-204/bpe``); ids are
resolved through a server-side discovery map, so no client-supplied path is
ever opened.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

from src.model import GPTConfig, MiniGPT
from src.tokenizer import tokenizer_from_checkpoint


@dataclass
class LoadedModel:
    id: str
    model: MiniGPT
    tokenizer: object
    info: dict


class ModelRegistry:
    """Finds ``checkpoint.pt`` files under one root and keeps a few models in memory."""

    def __init__(self, artifacts_dir: str | Path, *, cache_size: int = 2, device: str = "cpu") -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self.artifacts_dir = Path(artifacts_dir).resolve()
        self.device = device
        self.cache_size = cache_size
        # Re-entrant: handlers hold this lock while generating and call get() inside it.
        self.lock = threading.RLock()
        self._cache: OrderedDict[str, LoadedModel] = OrderedDict()
        self._info: dict[tuple[str, float], dict] = {}

    def discover(self) -> dict[str, Path]:
        """Map model id -> checkpoint path for every checkpoint below the root."""
        found: dict[str, Path] = {}
        if not self.artifacts_dir.is_dir():
            return found
        for path in sorted(self.artifacts_dir.rglob("checkpoint.pt")):
            resolved = path.resolve()
            if not resolved.is_relative_to(self.artifacts_dir):
                continue  # symlink pointing outside the allow-listed root
            relative = path.parent.relative_to(self.artifacts_dir)
            if not relative.parts:
                continue
            found[relative.as_posix()] = resolved
        return found

    def _load(self, model_id: str, path: Path) -> LoadedModel:
        # Only locally produced checkpoints under the allow-listed root reach this call.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = GPTConfig(**checkpoint["model_config"])
        model = MiniGPT(config).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        tokenizer = tokenizer_from_checkpoint(checkpoint)
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
        """Metadata for every discoverable checkpoint (a broken file reports an error)."""
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
        return models
