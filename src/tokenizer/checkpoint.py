"""Rebuild the tokenizer stored inside a training checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .bpe import BPETokenizer
from .character import CharacterTokenizer


def tokenizer_from_checkpoint(checkpoint: Mapping[str, Any]) -> CharacterTokenizer | BPETokenizer:
    """Return the tokenizer a checkpoint was trained with.

    Checkpoints written before Phase 2 only carry a flat ``vocabulary`` list and
    are treated as character tokenizers.
    """
    payload = checkpoint.get("tokenizer")
    if payload is None:
        return CharacterTokenizer(checkpoint["vocabulary"])
    kind = payload.get("type")
    if kind == "character":
        return CharacterTokenizer(payload["vocabulary"])
    if kind == "bpe":
        return BPETokenizer(list(payload["vocabulary"]), [tuple(pair) for pair in payload["merges"]])
    raise ValueError(f"unsupported checkpoint tokenizer type: {kind!r}")
