"""A small serializable character-initialized byte-pair tokenizer."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    """Deterministic BPE for compact course-scale corpora.

    The base alphabet is corpus characters. Each merge is selected by highest
    frequency then lexicographic order, making training repeatable.
    """

    UNKNOWN_TOKEN = "<unk>"

    def __init__(self, vocabulary: list[str], merges: list[tuple[str, str]]) -> None:
        if not vocabulary or vocabulary[0] != self.UNKNOWN_TOKEN or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("vocabulary must begin with unique <unk>")
        self.vocabulary = tuple(vocabulary)
        self.merges = tuple(merges)
        self.token_to_id = {token: index for index, token in enumerate(vocabulary)}
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    @classmethod
    def train(cls, text: str, *, max_merges: int = 100) -> "BPETokenizer":
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if max_merges < 0:
            raise ValueError("max_merges must be non-negative")
        words = [list(text)]
        alphabet = sorted({symbol for word in words for symbol in word})
        merges: list[tuple[str, str]] = []
        for _ in range(max_merges):
            pairs = Counter(pair for word in words for pair in zip(word, word[1:]))
            if not pairs:
                break
            maximum = max(pairs.values())
            pair = min(item for item, count in pairs.items() if count == maximum)
            merges.append(pair)
            merged = "".join(pair)
            words = [cls._merge_word(word, pair, merged) for word in words]
        vocabulary = [cls.UNKNOWN_TOKEN, *alphabet]
        for left, right in merges:
            token = left + right
            if token not in vocabulary:
                vocabulary.append(token)
        return cls(vocabulary, merges)

    @staticmethod
    def _merge_word(word: list[str], pair: tuple[str, str], merged: str) -> list[str]:
        result: list[str] = []
        index = 0
        while index < len(word):
            if index + 1 < len(word) and (word[index], word[index + 1]) == pair:
                result.append(merged)
                index += 2
            else:
                result.append(word[index])
                index += 1
        return result

    def _tokens(self, text: str) -> list[str]:
        # Replay merges in learned rank order, exactly as `train` applied them,
        # so encoding the training corpus reproduces the trained segmentation.
        pieces = list(text)
        for left, right in self.merges:
            if len(pieces) < 2:
                break
            pieces = self._merge_word(pieces, (left, right), left + right)
        return pieces

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return [self.token_to_id.get(piece, 0) for piece in self._tokens(text)]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        result = []
        for index in ids:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= self.vocabulary_size:
                raise ValueError("token ID is outside the vocabulary range")
            if skip_special_tokens and index == 0:
                continue
            result.append(self.vocabulary[index])
        return "".join(result)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"vocabulary": self.vocabulary, "merges": self.merges}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(list(payload["vocabulary"]), [tuple(pair) for pair in payload["merges"]])
