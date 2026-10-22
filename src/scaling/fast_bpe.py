"""Fast BPE trainer optimized for large text.

Uses incremental updates: only re-scan words affected by each merge,
instead of re-scanning the entire corpus for every merge.
"""

from __future__ import annotations
from collections import Counter
import math


def train_bpe_fast(text: str, max_merges: int = 4000, min_freq: int = 2) -> tuple[list[str], list[tuple[str, str]]]:
    """Train BPE with incremental updates for speed.

    Returns (vocabulary, merges_list).
    """
    # Pre-segment into words
    words = text.split()
    # Limit to first 1M chars worth of words for speed
    char_count = 0
    selected_words = []
    for w in words:
        if char_count > 2_000_000:
            break
        selected_words.append(w)
        char_count += len(w) + 1

    # Word frequency table
    word_freqs = Counter(selected_words)
    # Each word → list of characters
    word_splits = {w: list(w) for w in word_freqs}

    # Build initial pair frequencies
    pair_freqs = Counter()
    for word, freq in word_freqs.items():
        symbols = word_splits[word]
        for i in range(len(symbols) - 1):
            pair_freqs[(symbols[i], symbols[i+1])] += freq

    # Alphabet should cover every distinct character in the sample, including
    # spaces and typographic glyphs, otherwise they become <unk> and depress
    # chars/token. Training merges remain word-internal.
    alphabet = set(text)

    vocabulary = ["<unk>"] + sorted(alphabet)
    merges = []

    # Special: track which words contain each pair for fast updates
    # pair -> set of word indices
    pair_to_words = {}
    for word in word_freqs:
        symbols = word_splits[word]
        pairs_in_word = set()
        for i in range(len(symbols) - 1):
            pairs_in_word.add((symbols[i], symbols[i+1]))
        for pair in pairs_in_word:
            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word)

    vocab_set = set(vocabulary)

    for merge_num in range(max_merges):
        if not pair_freqs:
            break
        # Find best pair
        max_freq = max(pair_freqs.values())
        if max_freq < min_freq:
            break
        best_pair = min(p for p, f in pair_freqs.items() if f == max_freq)

        # Check termination
        if max_freq <= 0:
            break

        merged = best_pair[0] + best_pair[1]
        if merged not in vocab_set:
            vocabulary.append(merged)
            vocab_set.add(merged)
        merges.append(best_pair)

        # Update pair frequencies incrementally
        affected_words = list(pair_to_words.get(best_pair, set()))
        for word in affected_words:
            freq = word_freqs[word]
            old_symbols = word_splits[word]
            # Remove old pairs
            for i in range(len(old_symbols) - 1):
                pair = (old_symbols[i], old_symbols[i+1])
                pair_freqs[pair] -= freq
                if pair_freqs[pair] <= 0:
                    del pair_freqs[pair]
                if pair in pair_to_words:
                    pair_to_words[pair].discard(word)
                    if not pair_to_words[pair]:
                        del pair_to_words[pair]
            # Apply merge
            new_symbols = []
            i = 0
            while i < len(old_symbols):
                if i + 1 < len(old_symbols) and (old_symbols[i], old_symbols[i+1]) == best_pair:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(old_symbols[i])
                    i += 1
            word_splits[word] = new_symbols
            # Add new pairs
            new_pairs = set()
            for i in range(len(new_symbols) - 1):
                pair = (new_symbols[i], new_symbols[i+1])
                pair_freqs[pair] += freq
                new_pairs.add(pair)
            for pair in new_pairs:
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(word)

        if merge_num % 100 == 0 or merge_num < 10:
            print(f"  merge {merge_num}: {best_pair} (freq={max_freq}), vocab={len(vocabulary)}")

    return vocabulary, merges
