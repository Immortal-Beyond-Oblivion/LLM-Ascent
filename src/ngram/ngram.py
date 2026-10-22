"""N-gram language models with configurable smoothing.

Supports unigram, bigram, and trigram models with:
- Laplace (add-one) smoothing
- Good-Turing smoothing (c* = c - 0.75)
- Kneser-Ney smoothing (absolute discounting D = 0.75)
- Linear interpolation

All models operate on character-level tokens for comparability with
the transformer tokenizer comparison.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from .smoothing import (
    laplace_probability,
    good_turing_counts,
    good_turing_probability,
    kneser_ney_probability,
    interpolated_probability,
)


@dataclass
class NGramConfig:
    """Configuration for an n-gram language model."""

    order: int = 2  # 1=unigram, 2=bigram, 3=trigram
    smoothing: str = "laplace"  # "laplace", "good_turing", "kneser_ney", "interpolated"
    discount: float = 0.75  # Kneser-Ney absolute discount
    interpolation_weights: tuple[float, float, float] = (0.1, 0.3, 0.6)
    vocabulary: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.order not in (1, 2, 3):
            raise ValueError("order must be 1, 2, or 3")
        if self.smoothing not in ("laplace", "good_turing", "kneser_ney", "interpolated"):
            raise ValueError(f"unknown smoothing: {self.smoothing}")


class NGramModel:
    """Character-level n-gram language model.

    The Markov assumption: P(w_n | w_1...w_{n-1}) ≈ P(w_n | w_{n-k}...w_{n-1})
    where k is the model order.
    """

    def __init__(self, config: NGramConfig) -> None:
        self.config = config
        self.order = config.order
        self.smoothing = config.smoothing

        # Counts
        self.unigram_counts: Counter[str] = Counter()
        self.bigram_counts: Counter[tuple[str, str]] = Counter()
        self.trigram_counts: Counter[tuple[str, str, str]] = Counter()

        # Context counts (denominators)
        self.bigram_context_counts: Counter[str] = Counter()
        self.trigram_context_counts: Counter[tuple[str, str]] = Counter()

        # Vocabulary
        self.vocabulary: set[str] = set(config.vocabulary)
        self.vocabulary_size: int = len(self.vocabulary)

        # Good-Turing adjusted counts
        self._gt_adjusted: dict[int, float] = {}

        # Kneser-Ney continuation counts
        self._kn_continuation_counts: Counter[str] = Counter()
        self._kn_total_continuations: int = 0

    def train(self, text: str) -> None:
        """Train the n-gram model on character-level text."""
        if not text:
            raise ValueError("training text must be non-empty")

        # Build vocabulary
        self.vocabulary = set(text)
        self.vocabulary_size = len(self.vocabulary)

        # Count n-grams
        self.unigram_counts = Counter(text)
        self.bigram_counts = Counter(zip(text, text[1:]))
        self.trigram_counts = Counter(zip(text, text[1:], text[2:]))

        # Context counts
        for (w1, w2), count in self.bigram_counts.items():
            self.bigram_context_counts[w1] += count
        for (w1, w2, w3), count in self.trigram_counts.items():
            self.trigram_context_counts[(w1, w2)] += count

        # Precompute Good-Turing adjusted counts
        if self.smoothing == "good_turing":
            all_counts = Counter()
            all_counts.update(self.unigram_counts)
            all_counts.update(self.bigram_counts)
            all_counts.update(self.trigram_counts)
            self._gt_adjusted = good_turing_counts(all_counts)

        # Precompute Kneser-Ney continuation counts
        if self.smoothing == "kneser_ney":
            self._kn_continuation_counts = Counter()
            for (w1, w2) in self.bigram_counts:
                self._kn_continuation_counts[w2] += 1
            self._kn_total_continuations = sum(self._kn_continuation_counts.values())

    def _unigram_prob(self, char: str) -> float:
        """P(w) with Laplace smoothing."""
        count = self.unigram_counts.get(char, 0)
        total = sum(self.unigram_counts.values())
        return laplace_probability(count, total, self.vocabulary_size)

    def _bigram_prob(self, char: str, context: str) -> float:
        """P(w | prev) with configured smoothing."""
        bigram = (context, char)
        count = self.bigram_counts.get(bigram, 0)
        context_count = self.bigram_context_counts.get(context, 0)

        if self.smoothing == "laplace":
            return laplace_probability(count, context_count, self.vocabulary_size)
        elif self.smoothing == "good_turing":
            return good_turing_probability(count, context_count, self.vocabulary_size, self._gt_adjusted)
        elif self.smoothing == "kneser_ney":
            return kneser_ney_probability(
                count, context_count, self.vocabulary_size,
                self._kn_continuation_counts, self._kn_total_continuations,
                self.config.discount, char=char,
            )
        else:
            return laplace_probability(count, context_count, self.vocabulary_size)

    def _trigram_prob(self, char: str, context: tuple[str, str]) -> float:
        """P(w | prev1, prev2) with configured smoothing."""
        trigram = (context[0], context[1], char)
        count = self.trigram_counts.get(trigram, 0)
        context_count = self.trigram_context_counts.get(context, 0)

        if self.smoothing == "laplace":
            return laplace_probability(count, context_count, self.vocabulary_size)
        elif self.smoothing == "good_turing":
            return good_turing_probability(count, context_count, self.vocabulary_size, self._gt_adjusted)
        elif self.smoothing == "kneser_ney":
            return kneser_ney_probability(
                count, context_count, self.vocabulary_size,
                self._kn_continuation_counts, self._kn_total_continuations,
                self.config.discount, char=char,
            )
        else:
            return laplace_probability(count, context_count, self.vocabulary_size)

    def log_probability(self, text: str) -> float:
        """Compute log P(text) = Σ log P(w_i | w_<i)."""
        if not text:
            return 0.0

        log_prob = 0.0
        for i, char in enumerate(text):
            if self.order == 1:
                prob = self._unigram_prob(char)
            elif self.order == 2:
                if i == 0:
                    prob = self._unigram_prob(char)
                else:
                    prob = self._bigram_prob(char, text[i - 1])
            else:  # trigram
                if i == 0:
                    prob = self._unigram_prob(char)
                elif i == 1:
                    prob = self._bigram_prob(char, text[0])
                else:
                    if self.smoothing == "interpolated":
                        # Interpolated trigram
                        p_uni = self._unigram_prob(char)
                        p_bi = self._bigram_prob(char, text[i - 1])
                        p_tri = self._trigram_prob(char, (text[i - 2], text[i - 1]))
                        prob = interpolated_probability(p_uni, p_bi, p_tri, self.config.interpolation_weights)
                    else:
                        prob = self._trigram_prob(char, (text[i - 2], text[i - 1]))

            # Avoid log(0)
            prob = max(prob, 1e-12)
            log_prob += math.log(prob)

        return log_prob

    def perplexity(self, text: str) -> float:
        """Compute perplexity = exp(-1/N * log P(text))."""
        if not text:
            raise ValueError("text must be non-empty")
        log_prob = self.log_probability(text)
        return math.exp(-log_prob / len(text))

    def bits_per_character(self, text: str) -> float:
        """Compute bits per character = -1/N * log2 P(text)."""
        if not text:
            raise ValueError("text must be non-empty")
        log_prob = self.log_probability(text)
        return -log_prob / len(text) / math.log(2)

    def generate(self, length: int, seed: int = 42) -> str:
        """Generate text by sampling from the model."""
        import random
        random.seed(seed)

        if not self.vocabulary:
            raise ValueError("model must be trained before generation")

        chars = sorted(self.vocabulary)
        result = []

        for i in range(length):
            if self.order == 1 or i == 0:
                # Unigram sampling
                probs = [self._unigram_prob(c) for c in chars]
            elif self.order == 2 or i == 1:
                # Bigram sampling
                context = result[-1] if result else chars[0]
                probs = [self._bigram_prob(c, context) for c in chars]
            else:
                # Trigram sampling
                context = (result[-2], result[-1]) if len(result) >= 2 else (chars[0], result[-1] if result else chars[0])
                probs = [self._trigram_prob(c, context) for c in chars]

            # Normalize
            total = sum(probs)
            if total <= 0:
                probs = [1.0 / len(chars)] * len(chars)
            else:
                probs = [p / total for p in probs]

            # Sample
            r = random.random()
            cumulative = 0.0
            for c, p in zip(chars, probs):
                cumulative += p
                if r <= cumulative:
                    result.append(c)
                    break
            else:
                result.append(chars[-1])

        return "".join(result)
