"""WordPiece tokenizer (course-scale, deterministic).

WordPiece differs from BPE in its pair-scoring metric:
    score(ab) = freq(ab) / (freq(a) * freq(b))

This favours pairs that appear together much more than they appear alone.
Continuation tokens are prefixed with '##'.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from src.tokenizer.bpe import BPETokenizer  # Reuse encode/decode infrastructure


class WordPieceTokenizer:
    """Deterministic WordPiece tokenizer.

    Training:
    1. Start from character-level vocabulary
    2. Iteratively find the best pair (a, b) by score = freq(ab)/(freq(a)*freq(b))
    3. Add merged token a+b (with ## prefix for non-initial pieces)
    4. Repeat until vocab size reaches target
    """

    UNKNOWN_TOKEN = "<unk>"
    CONTINUATION_PREFIX = "##"

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
    def train(cls, text: str, *, max_merges: int = 100, target_vocab_size: int | None = None) -> "WordPieceTokenizer":
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # Determine target vocab size
        base_chars = sorted({c for c in text})
        base_size = 1 + len(base_chars)  # <unk> + chars
        if target_vocab_size is not None:
            max_merges = max(0, target_vocab_size - base_size)

        # Represent text as list of chars, split into words (for ## prefixing)
        # WordPiece operates on words: first char is bare, rest get ## prefix
        words_list = list(text)
        # But we work at character level for comparison with BPE
        symbols = [list(text)]  # List of word lists (each word is a list of symbols)
        alphabet = sorted({s for word in symbols for s in word})

        merges: list[tuple[str, str]] = []
        for _ in range(max_merges):
            # Count pair frequencies
            pair_freqs: Counter[tuple[str, str]] = Counter()
            for word in symbols:
                for pair in zip(word, word[1:]):
                    pair_freqs[pair] += 1

            if not pair_freqs:
                break

            # Compute unigram frequencies for scoring
            unigram_freqs: Counter[str] = Counter()
            for word in symbols:
                for s in word:
                    unigram_freqs[s] += 1

            # Score = freq(ab) / (freq(a) * freq(b))
            best_score = -1.0
            best_pair = None
            for pair, freq in pair_freqs.items():
                a_freq = unigram_freqs.get(pair[0], 1)
                b_freq = unigram_freqs.get(pair[1], 1)
                score = freq / (a_freq * b_freq)
                if score > best_score or best_pair is None:
                    best_score = score
                    best_pair = pair
                elif score == best_score and best_pair is not None and pair < best_pair:
                    best_pair = pair

            if best_pair is None:
                break

            merges.append(best_pair)
            merged = best_pair[0] + best_pair[1]

            # Apply merge
            symbols = [cls._merge_word(word, best_pair, merged) for word in symbols]

        # Build vocabulary
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
        # Replay merges in learned rank order
        pieces = list(text)
        for left, right in self.merges:
            if len(pieces) < 2:
                break
            pieces = self._merge_word(pieces, (left, right), left + right)
        return pieces

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        pieces = self._tokens(text)
        ids = []
        for piece in pieces:
            if piece in self.token_to_id:
                ids.append(self.token_to_id[piece])
            else:
                ids.append(0)  # <unk>
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        result = []
        for index in ids:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= self.vocabulary_size:
                raise ValueError("Token ID is outside the vocabulary range")
            token = self.vocabulary[index]
            if skip_special_tokens and token == self.UNKNOWN_TOKEN:
                continue
            result.append(token)
        return "".join(result)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"vocabulary": self.vocabulary, "merges": self.merges}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WordPieceTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(list(payload["vocabulary"]), [tuple(pair) for pair in payload["merges"]])
