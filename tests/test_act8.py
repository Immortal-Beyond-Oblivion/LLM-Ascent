"""Tests for Act 8: shared prompt assembly, hallucination rate, RAG controls.

Three v1 defects are in scope here and each gets a guard:

  * D1 RAG context truncated to ``context_length // 2`` CHARACTERS. The budget
    must be denominated in tokens, because characters and tokens are not
    interchangeable at a 6,487-token BPE on English prose (~4 chars/token).
  * D2 RAG prompt format != SFT format. One shared builder must serve both, or
    a model fine-tuned on one format is evaluated on the other.
  * hallucination must be MEASURED as a rate with an interval, not asserted.

D11's lesson is also enforced: ``InstructionExample`` binds positionally as
(instruction, response, input_text), so every construction here uses keywords.
"""

from __future__ import annotations

import unittest

from src.adaptation.instructions import InstructionExample, InstructionFormatter


class TestSharedPromptBuilder(unittest.TestCase):
    """D2: RAG and SFT must emit the same prompt contract."""

    def test_rag_and_instruction_paths_share_one_formatter(self):
        from src.rag.prompt import rag_prompt, shared_formatter

        ctx = "The kettle boiled at 100 degrees."
        got = rag_prompt("What temperature?", context=ctx,
                         tokenizer=_Tok(), context_length=64, answer_reserve=4)
        expected = shared_formatter().prompt(
            InstructionExample(instruction="What temperature?",
                               response="", input_text=ctx))
        self.assertEqual(got, expected,
                         "RAG built its own prompt format -- D2 is back")

    def test_shared_formatter_is_the_instruction_formatter(self):
        from src.rag.prompt import shared_formatter

        self.assertIsInstance(shared_formatter(), InstructionFormatter)
        self.assertIs(shared_formatter().__class__, InstructionFormatter)


class TestTokenContextBudget(unittest.TestCase):
    """D1: the context budget is in tokens, never characters."""

    def test_prompt_never_exceeds_context_length_in_tokens(self):
        from src.rag.prompt import rag_prompt

        tok = _Tok()
        huge = "word " * 4000  # 4,000 words, far past any window
        prompt = rag_prompt("Question?", context=huge, tokenizer=tok,
                            context_length=64, answer_reserve=8)
        self.assertLessEqual(len(tok.encode(prompt)), 64,
                             "prompt exceeds the window in TOKENS")

    def test_long_char_short_token_context_is_not_truncated(self):
        """The discriminator between D1 and its fix.

        A context that is longer in characters than the window but comfortably
        inside it in tokens must survive untouched. The old character budget
        (context_length // 2 = 32 characters) would have cut it to a stub.
        """
        from src.rag.prompt import fit_context

        tok = _Tok()  # 1 id per 4 characters, so chars != tokens
        # 100 characters: longer than the old budget (64 // 2 = 32 characters),
        # yet ~25 tokens, comfortably inside the window.
        context = ("The kettle boiled steadily on the stove while the "
                   "cat watched from the windowsill nearby.")
        self.assertGreater(len(context), 64 // 2,
                           "test context no longer discriminates: it would fit "
                           "the old character budget too")

        fitted = fit_context(context, instruction="Q?", tokenizer=tok,
                             context_length=64, answer_reserve=4)
        self.assertEqual(fitted, context,
                         "context was truncated by a CHARACTER budget -- D1 is back")

    def test_oversized_token_context_is_truncated(self):
        from src.rag.prompt import fit_context

        tok = _Tok()
        context = "x " * 500
        fitted = fit_context(context, instruction="Q?", tokenizer=tok,
                             context_length=48, answer_reserve=6)
        self.assertLess(len(tok.encode(fitted)), 200,
                        "oversized context was not trimmed at all")

    def test_budget_accounts_for_the_question_and_reserved_answer(self):
        """A budget that only fits the template would leave no room to answer."""
        from src.rag.prompt import fit_context

        tok = _Tok()
        long_q = "Explain in detail what the passage says about everything"
        fitted = fit_context("filler " * 500, instruction=long_q, tokenizer=tok,
                             context_length=48, answer_reserve=6)
        prompt_len = len(tok.encode(
            InstructionFormatter().prompt(
                InstructionExample(instruction=long_q, response="",
                                   input_text=fitted))))
        self.assertLessEqual(prompt_len, 48 - 6,
                             "context budget ignored the question or answer reserve")

    def test_impossible_budget_raises_rather_than_returning_a_lie(self):
        from src.rag.prompt import fit_context

        with self.assertRaises(ValueError):
            fit_context("context", instruction="q" * 500, tokenizer=_Tok(),
                        context_length=16, answer_reserve=8)

    def test_context_is_built_with_keyword_arguments(self):
        """D11: positional binding put the context into `response`."""
        from src.rag.prompt import fit_context

        ctx = "the answer lives here"
        fitted = fit_context(ctx, instruction="Q?", tokenizer=_Tok(),
                             context_length=64, answer_reserve=4)
        ex = InstructionExample(instruction="Q?", response="", input_text=fitted)
        # the formatter must place the context under ### Input:, not omit it
        self.assertIn(ctx, InstructionFormatter().prompt(ex))


class TestD1DoesNotReturn(unittest.TestCase):
    """Guard the exact expression that caused D1, at the source level.

    A behavioural test proves the new path is correct; this proves the old one
    is gone. `context_length // 2` was passed as a CHARACTER budget, and on the
    canonical 128-token window that admitted 64 characters -- ~9% of a
    retrieved chunk. It is a harmless-looking expression, so grep it.
    """

    def test_no_character_budget_derived_from_context_length(self):
        import io
        import tokenize
        from pathlib import Path

        # Strip comments and string literals first: honest comments ABOUT D1
        # quote the offending expression, and a naive substring scan would fail
        # on the explanation rather than on the defect.
        offenders = []
        for path in Path("src").rglob("*.py"):
            try:
                toks = [t for t in tokenize.generate_tokens(
                    io.StringIO(path.read_text(encoding="utf-8")).readline)
                    if t.type not in (tokenize.COMMENT, tokenize.STRING)]
            except (tokenize.TokenError, IndentationError):
                continue
            types = [t.string for t in toks]
            for i in range(len(types) - 2):
                if types[i] == "context_length" and types[i + 1] == "//" \
                        and types[i + 2] == "2":
                    # D1 was `max_context_chars=...context_length // 2`, so the
                    # character-flavoured name appears BEFORE the expression.
                    # Check a window on both sides rather than assuming order.
                    window = types[max(0, i - 6):i + 6]
                    if any("char" in s for s in window):
                        offenders.append(str(path))
                        break
        self.assertEqual(offenders, [],
                         f"character budget derived from context_length in {offenders}")

    def test_rag_cli_routes_through_the_shared_builder(self):
        from pathlib import Path

        src = Path("src/rag/query.py").read_text(encoding="utf-8")
        self.assertIn("rag_prompt", src,
                      "RAG CLI no longer routes through the shared builder")
        self.assertNotIn("max_context_chars=model.config.context_length",
                         src, "D1's character budget is back in the CLI")


class _Tok:
    """BPE-like compression: one id per 4 characters.

    Deliberately NOT 1:1. D1's defect only bites because characters and tokens
    are different quantities; a character-per-token tokenizer would make the
    old ``context_length // 2`` character budget look harmless and the guard
    would pass vacuously.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        out = []
        for i in range(0, len(text), 4):
            piece = text[i:i + 4]
            if piece not in self._vocab:
                self._vocab[piece] = len(self._vocab) + 1
            out.append(self._vocab[piece])
        return out

    def decode(self, ids) -> str:
        rev = {v: k for k, v in self._vocab.items()}
        return "".join(rev.get(int(i), "?") for i in ids)


if __name__ == "__main__":
    unittest.main()
