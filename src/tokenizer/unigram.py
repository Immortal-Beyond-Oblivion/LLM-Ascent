"""Unigram language model tokenizer (SentencePiece-style, course-scale).

Unigram LM starts with a large candidate vocabulary and iteratively prunes
tokens that have the smallest impact on the overall log-likelihood. Uses the
Viterbi algorithm for segmentation (finding the most likely tokenization).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path


class UnigramLMTokenizer:
    """Deterministic Unigram LM tokenizer.

    Training (simplified):
    1. Build candidate vocabulary from character n-grams (up to max_ngram)
    2. Initialize token probabilities uniformly
    3. Iteratively: Viterbi segment training text, recompute probabilities,
       prune tokens with lowest loss contribution
    4. Repeat until vocabulary reaches target size
    """

    UNKNOWN_TOKEN = "<unk>"
    UNKNOWN_ID = 0

    def __init__(self, vocabulary: list[str], scores: list[float] | None = None) -> None:
        if not vocabulary or vocabulary[0] != self.UNKNOWN_TOKEN:
            raise ValueError("vocabulary must begin with <unk>")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("vocabulary must be unique")
        self.vocabulary = tuple(vocabulary)
        self.token_to_id = {token: index for index, token in enumerate(vocabulary)}

        # Log-probabilities for each token (used in Viterbi)
        if scores is None:
            scores = [0.0] * len(vocabulary)
            # Default: log uniform probability
            for i in range(1, len(vocabulary)):
                scores[i] = math.log(1.0 / (len(vocabulary) - 1))
        self.scores = list(scores)

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    @classmethod
    def train(cls, text: str, *, target_vocab_size: int = 166, max_ngram: int = 4) -> "UnigramLMTokenizer":
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if target_vocab_size < 2:
            raise ValueError("target_vocab_size must be >= 2")

        # Step 1: Build candidate vocabulary from character n-grams
        candidates: Counter[str] = Counter()
        for n in range(1, max_ngram + 1):
            for i in range(len(text) - n + 1):
                gram = text[i:i + n]
                candidates[gram] += 1

        # Start with all candidates above a frequency threshold
        # Need at least target_vocab_size unique tokens
        vocab = {cls.UNKNOWN_TOKEN}
        for gram, freq in candidates.most_common():
            if len(vocab) >= target_vocab_size:
                break
            if len(gram) > 0:
                vocab.add(gram)

        # Sort vocabulary for determinism
        sorted_vocab = sorted(vocab - {cls.UNKNOWN_TOKEN})
        vocab_list = [cls.UNKNOWN_TOKEN] + sorted_vocab

        # Step 2: Compute initial scores
        # Score = log(prob) where prob = freq(token) / total_freq
        total_freq = sum(candidates.values())
        scores = [0.0] * len(vocab_list)
        for i, token in enumerate(vocab_list):
            if i == 0:
                continue  # <unk> score stays 0
            scores[i] = math.log((candidates.get(token, 1) + 1) / (total_freq + len(vocab_list)))

        # Step 3: Iteratively prune
        # Simple: remove tokens with lowest score until we reach target
        while len(vocab_list) > target_vocab_size:
            # Remove the token (excluding <unk>) with the lowest absolute score
            # (least likely token)
            min_score = float("inf")
            min_idx = -1
            for i in range(1, len(vocab_list)):
                if scores[i] < min_score:
                    min_score = scores[i]
                    min_idx = i
            if min_idx < 0:
                break
            del vocab_list[min_idx]
            del scores[min_idx]
            # Rebuild token_to_id
            # (done in __init__ but we're in train)

        return cls(vocab_list, scores)

    def _viterbi(self, text: str) -> list[str]:
        """Find the most likely segmentation of text into tokens.

        Uses dynamic programming: dp[i] = (best_score, best_token) for prefix i.
        """
        n = len(text)
        # dp[i] = (best_score, best_split_point, best_token)
        dp: list[tuple[float, int, str]] = [(-float("inf"), -1, "") for _ in range(n + 1)]
        dp[0] = (0.0, 0, self.UNKNOWN_TOKEN)

        for i in range(1, n + 1):
            # Try all tokens that could end at position i
            for j in range(max(0, i - 4), i):  # tokens up to 4 chars long
                substr = text[j:i]
                token = substr
                if token in self.token_to_id:
                    token_id = self.token_to_id[token]
                    score = self.scores[token_id]
                    prev_score = dp[j][0]
                    if prev_score + score > dp[i][0]:
                        dp[i] = (prev_score + score, j, token)
                else:
                    # Try <unk> as fallback for single char
                    if i - j == 1:
                        token_id = 0  # <unk>
                        score = self.scores[0] - 2.0  # Penalty for unk
                        prev_score = dp[j][0]
                        if prev_score + score > dp[i][0]:
                            dp[i] = (prev_score + score, j, text[j:i])

        # Backtrack to find segmentation
        if dp[n][1] < 0:
            # No valid segmentation, fall back to character-level
            return list(text)

        tokens = []
        i = n
        while i > 0:
            _, j, token = dp[i]
            tokens.append(token)
            i = j

        tokens.reverse()
        return tokens

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens = self._viterbi(text)
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.UNKNOWN_ID)
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        result = []
        for index in ids:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= self.vocabulary_size:
                raise ValueError("Token ID is outside the vocabulary range")
            token = self.vocabulary[index]
            if skip_special_tokens and index == self.UNKNOWN_ID:
                continue
            # Strip any special markers
            result.append(token)
        return "".join(result)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"vocabulary": self.vocabulary, "scores": self.scores}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "UnigramLMTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(list(payload["vocabulary"]), list(payload["scores"]))
