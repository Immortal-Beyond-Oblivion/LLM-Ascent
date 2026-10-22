"""Smoothing methods for n-gram language models.

Implements:
- Laplace (add-one) smoothing
- Good-Turing smoothing (c* = c - 0.75, Church & Gale 1991)
- Kneser-Ney smoothing (absolute discounting D = 0.75)
- Linear interpolation
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def laplace_probability(count: int, context_count: int, vocabulary_size: int, alpha: float = 1.0) -> float:
    """Laplace (add-alpha) smoothing: P(w|h) = (c(h,w) + α) / (c(h) + αV)."""
    if context_count < 0 or vocabulary_size <= 0:
        raise ValueError("context_count must be >= 0 and vocabulary_size > 0")
    return (count + alpha) / (context_count + alpha * vocabulary_size)


def good_turing_counts(counts: Counter[int]) -> dict[int, float]:
    """Compute Good-Turing adjusted counts c* = c - 0.75.

    Returns a mapping from original count to adjusted count.
    Uses the simple c* = c - 0.75 rule (Church & Gale 1991).
    """
    # Frequency of frequencies
    freq_of_freq: Counter[int] = Counter()
    for count in counts.values():
        freq_of_freq[count] += 1

    adjusted: dict[int, float] = {}
    for c in sorted(freq_of_freq.keys()):
        if c == 0:
            adjusted[c] = 0.0  # Unseen events
        else:
            # Simple Good-Turing: c* = c - 0.75
            adjusted[c] = c - 0.75
    return adjusted


def good_turing_probability(
    count: int,
    context_count: int,
    vocabulary_size: int,
    adjusted_counts: dict[int, float],
) -> float:
    """Good-Turing smoothed probability.

    P*(w|h) = c*(h,w) / N, where c* is the adjusted count.
    Falls back to Laplace for unseen events.
    """
    if context_count <= 0:
        return 1.0 / vocabulary_size

    c_star = adjusted_counts.get(count, 0.0)
    if c_star <= 0:
        # Unseen event: use N1/N as the total probability mass for unseen
        n1 = adjusted_counts.get(1, 0.0)  # Number of events seen once
        if n1 > 0:
            return n1 / (context_count * vocabulary_size)
        return 1.0 / vocabulary_size

    return c_star / context_count


def kneser_ney_probability(
    count: int,
    context_count: int,
    vocabulary_size: int,
    continuation_counts: Counter[str],
    total_continuations: int,
    discount: float = 0.75,
    char: str = "",
) -> float:
    """Kneser-Ney smoothed probability with absolute discounting.

    P_KN(w|h) = max(c(h,w) - D, 0) / c(h) + λ(h) * P_continuation(w)

    where:
    - D = 0.75 (absolute discount)
    - λ(h) = D * |{w : c(h,w) > 0}| / c(h)
    - P_continuation(w) = |{v : c(v,w) > 0}| / total_continuations
    """
    if context_count <= 0:
        return 1.0 / vocabulary_size

    # Discounted probability
    discounted = max(count - discount, 0.0) / context_count

    # Backoff weight
    unique_continuations = sum(1 for c in continuation_counts.values() if c > 0)
    lambda_h = discount * unique_continuations / context_count

    # Continuation probability for the specific character
    if total_continuations > 0:
        p_continuation = continuation_counts.get(char, 0) / total_continuations
    else:
        p_continuation = 1.0 / vocabulary_size

    return discounted + lambda_h * p_continuation


def interpolated_probability(
    unigram_prob: float,
    bigram_prob: float,
    trigram_prob: float | None,
    weights: tuple[float, float, float] = (0.1, 0.3, 0.6),
) -> float:
    """Linear interpolation: P(w|h) = λ1 P(w) + λ2 P(w|h1) + λ3 P(w|h1,h2).

    If trigram_prob is None, renormalizes the remaining weights.
    """
    if trigram_prob is None:
        # Bigram model: interpolate unigram and bigram only
        total = weights[0] + weights[1]
        if total <= 0:
            return unigram_prob
        return (weights[0] * unigram_prob + weights[1] * bigram_prob) / total

    total = weights[0] + weights[1] + weights[2]
    if total <= 0:
        return unigram_prob
    return (
        weights[0] * unigram_prob
        + weights[1] * bigram_prob
        + weights[2] * trigram_prob
    ) / total


def shannon_game_perplexity(log_probabilities: Sequence[float], num_guesses: int) -> float:
    """Compute perplexity in the Shannon Game framework.

    Perplexity = exp(-1/N * Σ log P(w_i | w_<i))

    where N is the number of characters (or tokens) guessed.
    """
    if not log_probabilities:
        raise ValueError("log_probabilities must be non-empty")
    avg_log_prob = sum(log_probabilities) / num_guesses
    return math.exp(-avg_log_prob)
