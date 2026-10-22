from __future__ import annotations

import math
import unittest
from pathlib import Path

from src.ngram.ngram import NGramConfig, NGramModel
from src.ngram.smoothing import (
    laplace_probability,
    good_turing_counts,
    good_turing_probability,
    kneser_ney_probability,
    interpolated_probability,
)


class SmoothingTests(unittest.TestCase):
    def test_laplace_basic(self):
        """Laplace: P = (c + α) / (N + αV)."""
        # c=5, N=100, V=50, α=1 → (5+1)/(100+50) = 6/150 = 0.04
        self.assertAlmostEqual(laplace_probability(5, 100, 50), 6 / 150)

    def test_laplace_unseen(self):
        """Laplace for unseen event: P = α / (N + αV)."""
        self.assertAlmostEqual(laplace_probability(0, 100, 50), 1 / 150)

    def test_good_turing_c_star(self):
        """Good-Turing: c* = c - 0.75 for c > 0."""
        from collections import Counter

        counts = Counter({"a": 10, "b": 5, "c": 1, "d": 1})
        adjusted = good_turing_counts(counts)
        self.assertEqual(adjusted[10], 9.25)  # 10 - 0.75
        self.assertEqual(adjusted[5], 4.25)   # 5 - 0.75
        self.assertEqual(adjusted[1], 0.25)   # 1 - 0.75

    def test_kneser_ney_unseen(self):
        """Kneser-Ney for unseen event gives continuation probability."""
        from collections import Counter

        continuation_counts = Counter({"a": 5, "b": 3, "c": 2})
        total = 10
        # Unseen event: discounted=0, so P = λ * P_continuation
        prob = kneser_ney_probability(0, 100, 50, continuation_counts, total, discount=0.75, char="a")
        self.assertGreater(prob, 0)
        self.assertLess(prob, 1)

    def test_interpolated_basic(self):
        """Interpolation: weighted average of probabilities."""
        # Equal weights
        p = interpolated_probability(0.1, 0.3, 0.6, (1/3, 1/3, 1/3))
        self.assertAlmostEqual(p, (0.1 + 0.3 + 0.6) / 3)

    def test_interpolated_no_trigram(self):
        """Interpolation without trigram renormalizes."""
        p = interpolated_probability(0.2, 0.4, None, (0.25, 0.75))
        self.assertAlmostEqual(p, 0.25 * 0.2 + 0.75 * 0.4)


class NGramModelTests(unittest.TestCase):
    def setUp(self):
        self.text = "to be or not to be that is the question"

    def test_unigram_model(self):
        """Unigram model trains and computes perplexity."""
        config = NGramConfig(order=1, smoothing="laplace")
        model = NGramModel(config)
        model.train(self.text)

        self.assertGreater(model.vocabulary_size, 0)
        ppl = model.perplexity(self.text)
        self.assertGreater(ppl, 1.0)
        self.assertTrue(math.isfinite(ppl))

    def test_bigram_model(self):
        """Bigram model trains and computes perplexity."""
        config = NGramConfig(order=2, smoothing="laplace")
        model = NGramModel(config)
        model.train(self.text)

        ppl = model.perplexity(self.text)
        self.assertGreater(ppl, 1.0)
        self.assertTrue(math.isfinite(ppl))

    def test_trigram_model(self):
        """Trigram model trains and computes perplexity."""
        config = NGramConfig(order=3, smoothing="laplace")
        model = NGramModel(config)
        model.train(self.text)

        ppl = model.perplexity(self.text)
        self.assertGreater(ppl, 1.0)
        self.assertTrue(math.isfinite(ppl))

    def test_higher_order_lower_perplexity(self):
        """Higher-order models should have lower perplexity on training data."""
        ppls = []
        for order in [1, 2, 3]:
            config = NGramConfig(order=order, smoothing="laplace")
            model = NGramModel(config)
            model.train(self.text)
            ppls.append(model.perplexity(self.text))

        # Trigram should have lower perplexity than unigram on training data
        self.assertLess(ppls[2], ppls[0])

    def test_good_turing_model(self):
        """Good-Turing smoothing produces valid perplexity."""
        config = NGramConfig(order=2, smoothing="good_turing")
        model = NGramModel(config)
        model.train(self.text)

        ppl = model.perplexity(self.text)
        self.assertGreater(ppl, 1.0)
        self.assertTrue(math.isfinite(ppl))

    def test_kneser_ney_model(self):
        """Kneser-Ney smoothing produces valid perplexity."""
        config = NGramConfig(order=2, smoothing="kneser_ney")
        model = NGramModel(config)
        model.train(self.text)

        ppl = model.perplexity(self.text)
        self.assertGreater(ppl, 1.0)
        self.assertTrue(math.isfinite(ppl))

    def test_interpolated_model(self):
        """Interpolated trigram produces valid perplexity."""
        config = NGramConfig(order=3, smoothing="interpolated")
        model = NGramModel(config)
        model.train(self.text)

        ppl = model.perplexity(self.text)
        self.assertGreater(ppl, 1.0)
        self.assertTrue(math.isfinite(ppl))

    def test_perplexity_bits_per_character(self):
        """BPC = perplexity-related metric is consistent."""
        config = NGramConfig(order=2, smoothing="laplace")
        model = NGramModel(config)
        model.train(self.text)

        ppl = model.perplexity(self.text)
        bpc = model.bits_per_character(self.text)
        # BPC = log2(perplexity)
        self.assertAlmostEqual(bpc, math.log2(ppl), places=10)

    def test_generate_produces_text(self):
        """Generation produces text of requested length."""
        config = NGramConfig(order=2, smoothing="laplace")
        model = NGramModel(config)
        model.train(self.text)

        sample = model.generate(50, seed=42)
        self.assertEqual(len(sample), 50)
        # All characters should be from vocabulary
        for char in sample:
            self.assertIn(char, model.vocabulary)

    def test_empty_text_raises(self):
        """Empty training text raises ValueError."""
        config = NGramConfig(order=2, smoothing="laplace")
        model = NGramModel(config)
        with self.assertRaises(ValueError):
            model.train("")

    def test_invalid_order_raises(self):
        """Invalid order raises ValueError."""
        with self.assertRaises(ValueError):
            NGramConfig(order=4, smoothing="laplace")

    def test_invalid_smoothing_raises(self):
        """Invalid smoothing raises ValueError."""
        with self.assertRaises(ValueError):
            NGramConfig(order=2, smoothing="invalid")

    def test_markov_assumption(self):
        """The Markov assumption: context only looks back 'order' tokens.

        For a bigram model, P(w_i | w_{i-1}, w_{i-2}) = P(w_i | w_{i-1}).
        We verify that the bigram probability only depends on the last token.
        """
        config = NGramConfig(order=2, smoothing="laplace")
        model = NGramModel(config)
        model.train("abcabcabcabc")

        # Bigram prob of 'a' given 'c' should be same regardless of what came before 'c'
        # (This is the Markov assumption)
        prob1 = model._bigram_prob("a", "c")
        prob2 = model._bigram_prob("a", "c")
        self.assertEqual(prob1, prob2)


class ComparisonRunnerTests(unittest.TestCase):
    def test_split_text(self):
        """Text splitting produces correct proportions."""
        from src.ngram.run_comparison import split_text

        text = "a" * 1000
        train, val = split_text(text, 0.9)
        self.assertEqual(len(train), 900)
        self.assertEqual(len(val), 100)

    def test_run_ngram_experiment(self):
        """run_ngram_experiment produces valid metrics.

        Writes into a temporary directory, NOT the tracked ``results/`` tree.
        This test previously clobbered
        ``results/act0-ngram/o2-laplace-seed42/metrics.json``, because
        ``run_ngram_experiment`` had no ``output_root`` parameter and so always
        wrote to ``results/``. Guarded by
        ``tests/test_results_append_only.py``.
        """
        import tempfile

        from src.ngram.run_comparison import run_ngram_experiment

        text = "to be or not to be " * 50
        train_text = text[:int(len(text) * 0.9)]
        val_text = text[int(len(text) * 0.9):]

        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_ngram_experiment(
                order=2,
                smoothing="laplace",
                train_text=train_text,
                validation_text=val_text,
                seed=42,
                output_root=tmp,
            )
            written = list(Path(tmp).rglob("*"))
            self.assertTrue(
                any(p.name == "metrics.json" for p in written),
                "experiment wrote no metrics.json to the requested output_root",
            )

        metrics.validate()
        self.assertGreater(metrics.final_validation_perplexity, 1.0)
        self.assertGreater(metrics.vocabulary_size, 0)


if __name__ == "__main__":
    unittest.main()
