"""Small evaluation metrics shared by the phase comparison scripts."""

from __future__ import annotations


def repetition_rate(text: str, *, ngram_size: int = 3) -> float:
    """Fraction of n-grams that repeat after their first occurrence."""
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")
    grams = [text[index : index + ngram_size] for index in range(max(0, len(text) - ngram_size + 1))]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)
