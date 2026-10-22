"""Deterministic character-level tokenization for the Phase 1 baseline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any


class CharacterTokenizer:
    """Map individual Unicode characters to integer IDs.

    Vocabulary construction is deterministic: ``<unk>`` is always ID zero and
    corpus characters receive IDs in sorted Unicode order. This makes a saved
    training configuration reproducible as long as its corpus is unchanged.
    """

    UNKNOWN_TOKEN = "<unk>"

    def __init__(self, vocabulary: Iterable[str]) -> None:
        """Create a tokenizer from an ordered vocabulary.

        ``vocabulary`` must start with ``<unk>`` and contain unique one-character
        tokens after it. Prefer :meth:`from_text` when constructing a baseline
        tokenizer from a corpus.
        """
        tokens = tuple(vocabulary)
        if not tokens or tokens[0] != self.UNKNOWN_TOKEN:
            raise ValueError("vocabulary must begin with the <unk> token")
        if len(set(tokens)) != len(tokens):
            raise ValueError("vocabulary tokens must be unique")
        if any(len(token) != 1 for token in tokens[1:]):
            raise ValueError("character vocabulary entries must each be one character")

        self._vocabulary = tokens
        self._token_to_id = MappingProxyType(
            {token: token_id for token_id, token in enumerate(tokens)}
        )

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        """Build a tokenizer from the distinct characters in ``text``.

        An empty corpus remains valid and produces a vocabulary containing only
        the unknown token.
        """
        cls._validate_text(text)
        return cls((cls.UNKNOWN_TOKEN, *sorted(set(text))))

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """The ordered immutable vocabulary."""
        return self._vocabulary

    @property
    def token_to_id(self) -> Mapping[str, int]:
        """A read-only mapping from token string to integer ID."""
        return self._token_to_id

    @property
    def vocabulary_size(self) -> int:
        """Number of IDs available to the model embedding layer."""
        return len(self._vocabulary)

    @property
    def unknown_token_id(self) -> int:
        """The ID used for characters absent from the vocabulary."""
        return 0

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs, substituting unknown characters with ID 0."""
        self._validate_text(text)
        return [self._token_to_id.get(character, self.unknown_token_id) for character in text]

    def encode_tensor(self, text: str, *, device: Any | None = None) -> Any:
        """Encode text into a 1-D ``torch.long`` tensor.

        PyTorch is imported only when this model-facing API is called, keeping
        vocabulary inspection usable in lightweight environments. Install the
        project dependencies before using this method.
        """
        try:
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "encode_tensor requires PyTorch; install dependencies with "
                "`python3 -m pip install -r requirements.txt`."
            ) from error

        return torch.tensor(self.encode(text), dtype=torch.long, device=device)

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text.

        Invalid IDs are rejected rather than silently converted, which exposes
        model/vocabulary incompatibilities early. Unknown IDs decode to the
        visible ``<unk>`` marker unless special tokens are being skipped.
        """
        decoded: list[str] = []
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if token_id < 0 or token_id >= self.vocabulary_size:
                raise ValueError(f"token ID {token_id} is outside the vocabulary range")
            if skip_special_tokens and token_id == self.unknown_token_id:
                continue
            decoded.append(self._vocabulary[token_id])
        return "".join(decoded)

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
