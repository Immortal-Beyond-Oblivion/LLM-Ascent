"""Small, deterministic datasets for causal language-model training."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


def load_text(path: str | Path) -> str:
    """Read UTF-8 training text and reject empty files."""
    text = Path(path).read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"dataset {path!s} is empty")
    return text


class TextDataset:
    """Tokenized text with reproducible train/validation splits and batches."""

    def __init__(self, token_ids: list[int], *, train_fraction: float = 0.9) -> None:
        if len(token_ids) < 3:
            raise ValueError("at least three tokens are needed for causal training")
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be strictly between zero and one")
        split_index = int(len(token_ids) * train_fraction)
        if split_index < 2 or len(token_ids) - split_index < 2:
            raise ValueError("both splits need at least two tokens")
        tokens = torch.tensor(token_ids, dtype=torch.long)
        self.train = tokens[:split_index]
        self.validation = tokens[split_index:]

    @classmethod
    def from_text(cls, text: str, tokenizer: Encoder, *, train_fraction: float = 0.9) -> "TextDataset":
        return cls(tokenizer.encode(text), train_fraction=train_fraction)

    def batch(
        self,
        split: str,
        *,
        batch_size: int,
        context_length: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return integer input/target batches shaped ``[B, T]``.

        Targets are inputs shifted by one token in the underlying stream.
        """
        if batch_size <= 0 or context_length <= 0:
            raise ValueError("batch_size and context_length must be positive")
        data = {"train": self.train, "validation": self.validation}.get(split)
        if data is None:
            raise ValueError("split must be 'train' or 'validation'")
        if len(data) <= context_length:
            raise ValueError("split is too short for the requested context length")
        starts = torch.randint(
            len(data) - context_length,
            (batch_size,),
            generator=generator,
        )
        inputs = torch.stack([data[start : start + context_length] for start in starts])
        targets = torch.stack([data[start + 1 : start + context_length + 1] for start in starts])
        return inputs.to(device), targets.to(device)
