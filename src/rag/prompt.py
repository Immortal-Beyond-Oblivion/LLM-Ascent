"""One prompt builder shared by RAG and instruction tuning.

Fixes two v1 defects:

**D2 -- RAG prompt format != SFT format.** RAG assembled its own template, so a
model fine-tuned on the Alpaca contract in :class:`InstructionFormatter` was
evaluated on a different string. Both paths now go through that same class;
``rag_prompt`` is a thin wrapper that supplies the retrieved context as the
instruction's ``input_text``.

**D1 -- context budgeted in CHARACTERS.** The old call site passed
``model.config.context_length // 2`` as ``max_context_chars``. Characters and
tokens are not interchangeable: the canonical BPE runs ~4 characters per token
on English prose, so a character budget both over- and under-filled the window
depending on the text. The budget here is denominated in tokens and the result
is verified by re-encoding the assembled prompt.

``InstructionExample`` binds positionally as ``(instruction, response,
input_text)`` -- D11 shipped because a payload was passed positionally and
landed in ``response``. Every construction in this module therefore uses
keyword arguments, and ``test_context_is_built_with_keyword_arguments`` pins it.
"""

from __future__ import annotations

from src.adaptation.instructions import InstructionExample, InstructionFormatter

_FORMATTER = InstructionFormatter()


def shared_formatter() -> InstructionFormatter:
    """The single formatter used by SFT, generation, and RAG."""
    return _FORMATTER


def _prompt_length(instruction: str, context_text: str, tokenizer) -> int:
    prompt = _FORMATTER.prompt(
        InstructionExample(instruction=instruction, response="",
                           input_text=context_text))
    return len(tokenizer.encode(prompt))


def fit_context(context: str, *, instruction: str, tokenizer,
                context_length: int, answer_reserve: int) -> str:
    """The longest prefix of ``context`` whose assembled prompt fits the window.

    The budget is in tokens. ``answer_reserve`` tokens are held back for the
    response, so a budget that merely fits the template cannot win.

    Raises:
        ValueError: if the instruction plus an empty context already exceeds
            ``context_length - answer_reserve``. Silently returning a truncated
            context there would produce a prompt the model cannot answer in.
    """
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if answer_reserve < 0:
        raise ValueError("answer_reserve must not be negative")

    if _prompt_length(instruction, "", tokenizer) + answer_reserve > context_length:
        raise ValueError(
            f"instruction plus answer reserve ({answer_reserve} tokens) does not "
            f"fit a {context_length}-token window; no context can be supplied"
        )

    ids = tokenizer.encode(context)
    if not ids:
        return ""

    def fits(k: int) -> bool:
        return (_prompt_length(instruction, tokenizer.decode(ids[:k]), tokenizer)
                + answer_reserve <= context_length)

    # Prefix length is monotone in prompt length, so binary-search the largest
    # prefix that fits rather than eyeballing a margin.
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1

    fitted = tokenizer.decode(ids[:lo])
    if not fits(lo):  # pragma: no cover - defensive; binary search implies it
        raise AssertionError("fit_context returned a prompt that overflows")
    return fitted


def rag_prompt(question: str, *, context: str, tokenizer, context_length: int,
               answer_reserve: int = 8) -> str:
    """RAG prompt in exactly the contract SFT trains on, ending ready to answer.

    Returns a string ending with ``### Response:\\n`` so generation continues
    from the same position a fine-tuned model was trained to start at.
    """
    fitted = fit_context(context, instruction=question, tokenizer=tokenizer,
                         context_length=context_length,
                         answer_reserve=answer_reserve)
    return _FORMATTER.prompt(
        InstructionExample(instruction=question, response="", input_text=fitted))
