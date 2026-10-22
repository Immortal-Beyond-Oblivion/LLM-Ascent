"""Tests for Act 8b: the hallucination claim format and its classifier.

The claim format has to be stated before any generation is sampled, or the
classification is fitted to whatever the model happened to emit. The format
under test:

    CLAIM    : "the supplied passage states <NUMBER>"
    classes  : supported     -- the generation produced that number
               contradicted  -- it produced a DIFFERENT number
               unsupported   -- it produced no number at all
    RATE     : hallucination = (contradicted + unsupported) / n = 1 - supported

This measures GROUNDEDNESS (does the answer stay faithful to the supplied
source), not factual truth in the world. No fact base is available to check
the latter, and claiming it would be the kind of unverified assertion Act 7
had to retract three times.
"""

from __future__ import annotations

import unittest

import torch


class TestNumberExtraction(unittest.TestCase):
    def test_extracts_plain_number(self):
        from src.eval.act8 import extract_number

        self.assertEqual(extract_number("4"), "4")
        self.assertEqual(extract_number("The answer is 4."), "4")
        self.assertEqual(extract_number("4 people"), "4")

    def test_normalizes_grouping_and_sign(self):
        from src.eval.act8 import extract_number

        self.assertEqual(extract_number("1,000"), "1000")
        self.assertEqual(extract_number("-3"), "-3")
        self.assertEqual(extract_number("3.5"), "3.5")

    def test_returns_none_when_absent(self):
        from src.eval.act8 import extract_number

        for text in ("", "no digits here", "twenty two", "###"):
            self.assertIsNone(extract_number(text), repr(text))

    def test_uses_the_first_number(self):
        """The first number IS the answer commitment; later ones are hedging."""
        from src.eval.act8 import extract_number

        self.assertEqual(extract_number("4 or maybe 9"), "4")

    def test_percent_and_year_are_extracted(self):
        from src.eval.act8 import extract_number

        self.assertEqual(extract_number("50% off"), "50")
        self.assertEqual(extract_number("in 2015 we built it"), "2015")


class TestClaimClassification(unittest.TestCase):
    """The three classes, exactly as the claim format defines them."""

    def test_supported(self):
        from src.eval.act8 import classify

        self.assertEqual(classify("4", "4"), "supported")
        self.assertEqual(classify("The passage says 4.", "4"), "supported")

    def test_contradicted(self):
        from src.eval.act8 import classify

        self.assertEqual(classify("9", "4"), "contradicted")
        self.assertEqual(classify("it is 12 people", "4"), "contradicted")

    def test_unsupported_when_no_number(self):
        from src.eval.act8 import classify

        for g in ("", "I cannot answer that", "!!!!", "unknown", "the passage"):
            self.assertEqual(classify(g, "4"), "unsupported", repr(g))

    def test_unsupported_when_generation_is_degenerate(self):
        from src.eval.act8 import classify

        self.assertEqual(classify("!!!!!!!!!!!!!!!!!!!!!!!!!!!!", "4"), "unsupported")

    def test_classification_is_total(self):
        """Every generation must land in exactly one class -- no silent drops."""
        from src.eval.act8 import classify

        for g in ("4", "7", "", "abc", "4 people", "0"):
            self.assertIn(classify(g, "4"),
                          {"supported", "contradicted", "unsupported"}, repr(g))


class TestRatesAndIntervals(unittest.TestCase):
    def test_hallucination_rate_is_one_minus_supported(self):
        from src.eval.act8 import rate_summary

        labels = ["supported", "supported", "contradicted", "unsupported", "supported"]
        s = rate_summary(labels)
        self.assertEqual(s["n"], 5)
        self.assertEqual(s["supported"], 3)
        self.assertEqual(s["contradicted"], 1)
        self.assertEqual(s["unsupported"], 1)
        self.assertAlmostEqual(s["hallucination_rate"], 2 / 5)
        self.assertAlmostEqual(s["supported_rate"], 3 / 5)
        self.assertAlmostEqual(s["supported_rate"] + s["hallucination_rate"], 1.0)

    def test_rate_summarises_to_one_for_all_supported(self):
        from src.eval.act8 import rate_summary

        s = rate_summary(["supported"] * 10)
        self.assertEqual(s["hallucination_rate"], 0.0)
        lo, hi = s["supported_wilson95"]
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_wilson_interval_is_within_unit_range_for_extremes(self):
        """The naive formula once returned a NEGATIVE lower bound (-0.2852)
        when z^2/(4n) was used where z^2/(4n^2) belongs. Pin it."""
        from src.bench.stats import wilson_interval

        for k, n in ((0, 18), (18, 18), (0, 1), (1, 1), (3, 128), (128, 128)):
            lo, hi = wilson_interval(k, n)
            self.assertGreaterEqual(lo, 0.0, f"k={k} n={n}")
            self.assertLessEqual(hi, 1.0, f"k={k} n={n}")
            self.assertLessEqual(lo, hi, f"k={k} n={n}")

    def test_empty_input_is_rejected_rather_than_dividing_by_zero(self):
        from src.eval.act8 import rate_summary

        with self.assertRaises(ValueError):
            rate_summary([])


class TestCounterfactualConstruction(unittest.TestCase):
    def test_swaps_exactly_the_target_span(self):
        from src.eval.act8 import swap_number

        out = swap_number("The recipe serves 4 people in 2015.", "4", "9")
        self.assertIn("9 people", out)
        self.assertIn("2015", out)
        self.assertNotIn("4 people", out)

    def test_refuses_when_the_number_is_missing(self):
        from src.eval.act8 import swap_number

        with self.assertRaises(ValueError):
            swap_number("no numbers at all here", "4", "9")

    def test_single_number_passages_are_detectable(self):
        from src.eval.act8 import has_single_number, numbers_in

        self.assertTrue(has_single_number("it serves 4 people"))
        self.assertFalse(has_single_number("4 of 9 people"))
        self.assertEqual(numbers_in("4 of 9 people"), ["4", "9"])


class TestSentenceSelection(unittest.TestCase):
    """A scored item must STATE one number, or the rate measures ill-posed
    questions rather than hallucination."""

    def _cands(self, text):
        from src.eval.act8_hallucination import candidate_sentences
        return candidate_sentences(text)

    def test_accepts_a_plain_statement_with_one_number(self):
        got = self._cands("The cost to be in the class is $35 per person.")
        self.assertEqual(len(got), 1)
        self.assertIn("$35", got[0])

    def test_rejects_a_question(self):
        """A question does not STATE its number."""
        # only a question present -> nothing scoreable at all
        self.assertEqual(self._cands("How many days until 3 deadlines fall due?"), [])
        # and a question never survives alongside real statements
        got = self._cands(
            "The cost is $35 per person. How many days until 3 deadlines fall due?")
        self.assertEqual(got, ["The cost is $35 per person."], got)

    def test_rejects_list_markers(self):
        """'1) for a newly created site' states nothing about 1."""
        got = self._cands("Here is advice. The cost is $35 per person. "
                          "1) for a newly created site be safe.")
        self.assertEqual(got, ["The cost is $35 per person."], got)

    def test_rejects_ordinals_a_swap_cannot_inflect(self):
        """'September 22nd' swapped literally becomes '21nd'."""
        got = self._cands(
            "The festival opens on September 22nd near the park entrance.")
        self.assertEqual(got, [], "ordinal passed through; swap would be broken")

    def test_rejects_sentences_with_two_numbers(self):
        """Ambiguous target: which one does the passage 'state'."""
        got = self._cands("We shipped 4 of 9 units to the depot today ok.")
        self.assertEqual(got, [])

    def test_rejects_too_short_fragments(self):
        got = self._cands("4.")
        self.assertEqual(got, [])

    def test_grouped_number_does_not_raise(self):
        """Regression: numbers_in() normalizes '1,000' -> '1000', and calling
        .index('1000') on a sentence containing '1,000' raised ValueError and
        aborted the whole C4 scan."""
        got = self._cands("We sold 1,000 units to the depot yesterday morning.")
        self.assertEqual(len(got), 1, got)

    def test_every_selected_passage_swaps_cleanly(self):
        from src.eval.act8_hallucination import load_passages
        from src.eval.act8 import numbers_in

        items = load_passages("data/v2/c4_en_shard_0.jsonl", limit=60,
                             tokenizer=_RealTok.instance())
        self.assertEqual(len(items), 60, "not enough scoreable C4 passages")
        for it in items:
            self.assertNotEqual(it["counterfactual_passage"], it["passage"],
                                f"failed to swap: {it['passage']!r}")
            self.assertNotIn(it["source_number"],
                             numbers_in(it["counterfactual_passage"]),
                             f"source number survived: {it['passage']!r}")


class TestCounterfactualIsGrammatical(unittest.TestCase):
    def test_swapped_passage_changes_exactly_one_number(self):
        from src.eval.act8_hallucination import load_passages
        from src.eval.act8 import numbers_in

        items = load_passages("data/v2/c4_en_shard_0.jsonl", limit=12,
                             tokenizer=_RealTok.instance())
        self.assertGreaterEqual(len(items), 8, "not enough scoreable C4 passages")
        for it in items:
            self.assertNotIn(it["source_number"],
                             numbers_in(it["counterfactual_passage"]),
                             "source number survived the swap")
            self.assertEqual(len(numbers_in(it["counterfactual_passage"])), 1,
                             "counterfactual introduced or removed a number")

    def test_no_ordinal_suffixes_in_the_counterfactual(self):
        import re

        from src.eval.act8_hallucination import load_passages

        items = load_passages("data/v2/c4_en_shard_0.jsonl", limit=12,
                             tokenizer=_RealTok.instance())
        for it in items:
            self.assertIsNone(
                re.search(r"\d+(?:st|nd|rd|th)\b", it["counterfactual_passage"]),
                f"ungrammatical ordinal in {it['counterfactual_passage']!r}")


class TestCandidateScoring(unittest.TestCase):
    """The answer-copying instrument: score ONLY the candidate answer.

    Regression: an earlier version summed the log-probabilities over the whole
    sequence, so a candidate's score also contained the PROMPT's own
    likelihood. That is harmless when comparing two candidates under one
    prompt (the prompt cancels) but invalidates every across-condition
    contrast, because an oracle prompt is far longer than a closed-book one.
    It surfaced as a passage-lift of -246 nats, which is absurd on its face.
    """

    class _ConstantModel(torch.nn.Module):
        def __init__(self, vocab: int = 100):
            super().__init__()
            self._probe = torch.nn.Parameter(torch.zeros(1))
            self._vocab = vocab

        def forward(self, token_ids, targets=None, cache=None):
            b, L = token_ids.shape
            logits = torch.zeros(b, L, self._vocab)
            logits[..., 7] = 1.0
            return logits, None, None

    class _NumTok:
        def encode(self, text: str) -> list[int]:
            # A candidate (" 35") encodes to the single token 7; a prompt
            # encodes to filler tokens that are NOT 7.
            return [7] if text.startswith(" ") else [3] * len(text)

    def test_score_depends_only_on_the_candidate_not_the_prompt_length(self):
        import torch
        import torch.nn.functional as F

        from src.eval.act8 import score_candidates

        model = self._ConstantModel()
        tok = self._NumTok()

        logits = torch.zeros(1, 10, 100)
        logits[0, 0, 7] = 1.0
        expected = float(F.log_softmax(logits[0], dim=-1)[0, 7])

        short = score_candidates(model, [3, 3, 3], ["35"], tok, "cpu")
        long = score_candidates(model, [3] * 50, ["35"], tok, "cpu")

        self.assertAlmostEqual(short["35"]["total_logprob"], expected, places=5)
        self.assertAlmostEqual(long["35"]["total_logprob"], expected, places=5,
                               msg="candidate score absorbed the prompt's own "
                                   "log-probabilities")

    def test_score_of_a_multi_token_candidate_scales_with_its_own_length(self):
        import torch
        import torch.nn.functional as F

        from src.eval.act8 import score_candidates

        class TwoTokenTok(self._NumTok):
            def encode(self, text: str) -> list[int]:
                return [7, 7] if text.startswith(" ") else [3] * len(text)

        model = self._ConstantModel()
        tok = TwoTokenTok()
        logits = torch.zeros(1, 10, 100)
        logits[0, 0, 7] = 1.0
        expected = 2 * float(F.log_softmax(logits[0], dim=-1)[0, 7])

        got = score_candidates(model, [3, 3], ["35"], tok, "cpu")
        self.assertAlmostEqual(got["35"]["total_logprob"], expected, places=5)
        self.assertEqual(got["35"]["n_tokens"], 2)


class TestCounterfactualMatching(unittest.TestCase):
    """The counterfactual must be comparable to the source it replaces."""

    def test_counterfactual_keeps_the_digit_count(self):
        from src.eval.act8 import match_counterfactual

        for src in ("35", "572", "3", "2017", "45"):
            cf = match_counterfactual(src, _RealTok())
            self.assertEqual(len(cf), len(src), f"{src} -> {cf}")

    def test_counterfactual_keeps_the_token_count(self):
        """'35' is one BPE token but '34' is three; comparing them would hand
        the win to whichever side got the merge."""
        from src.eval.act8 import match_counterfactual, number_token_count

        tok = _RealTok()
        for src in ("35", "572", "3", "2017", "45", "10"):
            cf = match_counterfactual(src, tok)
            self.assertEqual(number_token_count(tok, cf),
                             number_token_count(tok, src),
                             f"{src} -> {cf} changes the token count")

    def test_counterfactual_is_different_from_the_source(self):
        from src.eval.act8 import match_counterfactual

        tok = _RealTok()
        for src in ("35", "572", "3", "2017"):
            self.assertNotEqual(match_counterfactual(src, tok), src)

    def test_unmatched_source_raises(self):
        from src.eval.act8 import match_counterfactual

        with self.assertRaises(ValueError):
            match_counterfactual("not-a-number", _RealTok())


class _RealTok:
    """The canonical BPE, so token counts are the real ones."""

    _cache = None

    @classmethod
    def instance(cls):
        if cls._cache is None:
            from src.tokenizer.bpe import BPETokenizer
            cls._cache = BPETokenizer.load("checkpoints/canonical/tokenizer.json")
        return cls._cache

    def encode(self, text: str) -> list[int]:
        return _RealTok.instance().encode(text)


class TestRetrievalIsNotDegenerate(unittest.TestCase):
    """Regression guard for the degenerate retrieved condition.

    The first version queried the retriever with the instruction text, whose
    terms {what, number, does, the, passage, state} cannot come from a passage
    about a number. Term-overlap ranking degenerated to a constant and the
    SAME chunk came back for all 40 items -- a retrieval success rate of 0/40
    that looked like a finding. These tests fail if that returns.
    """

    def test_anchor_query_actually_retrieves_the_target_document(self):
        from src.eval.act8_hallucination import (build_corpus, load_passages)
        from src.rag.retriever import TextRetriever
        from pathlib import Path

        tok = _RealTok.instance()
        items = load_passages("data/v2/c4_en_shard_0.jsonl", limit=12,
                              tokenizer=tok)
        corpus = build_corpus("data/v2/c4_en_shard_0.jsonl",
                              "artifacts/act8/test_corpus", n_docs=120)
        retriever = TextRetriever.from_files(corpus, chunk_words=40,
                                              overlap_words=0)
        hits, docs = 0, set()
        for item in items:
            query = " ".join(item["anchor_terms"])
            self.assertTrue(query, "item has no lexical anchor at all")
            got = retriever.retrieve(query, top_k=1)
            self.assertTrue(got)
            docs.add(got[0].source)
            if Path(got[0].source).stem == f"doc_{item['doc_index']:05d}":
                hits += 1
        self.assertGreater(len(docs), 1,
                           "every query returned the same chunk -- the ranking "
                           "has degenerated to a constant")
        self.assertGreater(hits, 0,
                           "the anchor never retrieves the target document")

    def test_anchor_never_contains_the_answer_number(self):
        """The anchor is a topical handle, not the answer."""
        from src.eval.act8_hallucination import load_passages
        from src.eval.act8 import NUMBER_RE

        tok = _RealTok.instance()
        items = load_passages("data/v2/c4_en_shard_0.jsonl", limit=40,
                              tokenizer=tok)
        for item in items:
            self.assertNotIn(item["source_number"],
                             " ".join(item["anchor_terms"]),
                             "the retrieval anchor hands over the answer")
            if item["anchor_sentence"]:
                self.assertIsNone(NUMBER_RE.search(item["anchor_sentence"]),
                                  "the anchor sentence contains a number")


class TestChanceBaselineIsPerItem(unittest.TestCase):
    """A flat 'chance = 1/3' would be wrong: the candidate sets vary."""

    def test_candidate_set_sizes_vary(self):
        from src.eval.act8_hallucination import (distractor_pool, load_passages,
                                                 partner_number)

        tok = _RealTok.instance()
        items = load_passages("data/v2/c4_en_shard_0.jsonl", limit=40,
                              tokenizer=tok)
        dist = distractor_pool(items)
        sizes = set()
        for i, it in enumerate(items):
            alt = partner_number("oracle", it)
            sizes.add(len({c for c in [it["source_number"], alt] + dist[i] if c}))
        self.assertGreater(len(sizes), 1,
                           "every item has the same candidate-set size, so the "
                           "per-item chance baseline is untested")
        self.assertTrue(sizes <= {3, 4, 5, 6}, sizes)

    def test_nonnumeric_control_context_has_no_number(self):
        from src.eval.act8_hallucination import assemble, load_passages
        from src.eval.act8 import NUMBER_RE

        tok = _RealTok.instance()
        items = load_passages("data/v2/c4_en_shard_0.jsonl", limit=40,
                              tokenizer=tok)
        checked = 0
        for it in items:
            if not it["anchor_sentence"]:
                continue
            _prompt, fitted = assemble("nonnumeric", it, "", tokenizer=tok,
                                       context_length=128)
            self.assertIsNone(NUMBER_RE.search(fitted),
                              f"the number-free control contains a number: {fitted!r}")
            checked += 1
        self.assertGreater(checked, 0, "no usable number-free control items")


class TestTemplateCompletionEvidence(unittest.TestCase):
    """'CALC(s)' is template completion, and that must be verifiable."""

    def test_argmax_after_calc_is_a_template_fragment_not_an_operator(self):
        from src.eval.act8_tools import template_completion_evidence
        from src.eval.act7_sft import load_canonical
        from src.tokenizer.bpe import BPETokenizer

        tok = BPETokenizer.load("checkpoints/canonical/tokenizer.json")
        model = load_canonical("6.3M")
        ev = template_completion_evidence(model, tok, device="cpu")
        self.assertEqual(ev["template_tokens"], ["C", "AL", "C", "("])
        self.assertEqual(ev["argmax_token"], "s")
        self.assertEqual(ev["runner_up"], "express")
        # The argmax must NOT be an arithmetic symbol or a digit.
        self.assertNotIn(ev["argmax_token"], list("+-*/0123456789"))
        self.assertIn("template", ev["interpretation"].lower())


if __name__ == "__main__":
    unittest.main()
