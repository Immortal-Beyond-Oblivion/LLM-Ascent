"""Act 8: hallucination measured as a rate over real C4 text.

THE CLAIM FORMAT (declared before any generation is sampled, so the
classification cannot be fitted to whatever the model happened to emit):

    CLAIM    "the supplied passage states <NUMBER>"
    class    supported     -- the generation produced that number
             contradicted  -- it produced a DIFFERENT number
             unsupported   -- it produced no number at all
    RATE     hallucination = (contradicted + unsupported) / n = 1 - supported

Numbers are the claim currency because they are the one atomic fact that can be
extracted and verified mechanically from C4 without an annotator or an external
knowledge base: a scan of 1,500 C4 documents found a number in 70.9%, while
world-fact templates ("the capital of X is Y") appeared in 0%.

WHAT THIS MEASURES. Groundedness -- whether the answer stays faithful to the
supplied source. It does NOT measure factual truth against the world: no fact
base is available here, and a rate that claimed to would be exactly the kind of
unverified assertion Act 7 had to retract.

The first number in a generation is the answer commitment; numbers appearing
after it are hedging. A generation with no number at all is `unsupported`
rather than being dropped, so the denominator is every generation sampled.
"""

from __future__ import annotations

import re
from collections import Counter

import torch
import torch.nn.functional as F

from src.bench.stats import wilson_interval

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
CLASSES = ("supported", "contradicted", "unsupported")
COPY_CLASSES = ("source", "counterfactual", "neither")


def normalize_number(text: str) -> str:
    """Canonical form of a numeric literal: no thousands separators."""
    return text.replace(",", "")


def extract_number(text: str) -> str | None:
    """The FIRST number in ``text``, normalized, or None if there is none."""
    if not text:
        return None
    match = NUMBER_RE.search(text)
    if match is None:
        return None
    return normalize_number(match.group(0))


def numbers_in(text: str) -> list[str]:
    """Every number in ``text``, normalized and in order."""
    return [normalize_number(m) for m in NUMBER_RE.findall(text)]


def has_single_number(text: str) -> bool:
    """True when exactly one number appears -- the passages we can score."""
    return len(numbers_in(text)) == 1


def classify(generation: str, source_number: str) -> str:
    """Label one generation against the number the passage supplies.

    Raises:
        ValueError: if ``source_number`` is not a number, which would make every
            generation `contradicted` and report a fabricated rate.
    """
    source = extract_number(str(source_number))
    if source is None:
        raise ValueError(
            f"source_number {source_number!r} is not a number; refusing to "
            "classify against it (every generation would be 'contradicted')"
        )
    got = extract_number(generation)
    if got is None:
        return "unsupported"
    return "supported" if got == source else "contradicted"


def rate_summary(labels: list[str]) -> dict:
    """Counts, rates, and a Wilson interval for the supported share.

    The hallucination rate is the complement of the supported rate by
    construction, so the two must sum to 1 and the interval is reported on the
    supported share (``wilson_interval`` only accepts a success count).
    """
    if not labels:
        raise ValueError("rate_summary requires at least one label")
    unknown = set(labels) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown label(s): {sorted(unknown)}")
    n = len(labels)
    counts = Counter(labels)
    supported = counts["supported"]
    contradicted = counts["contradicted"]
    unsupported = counts["unsupported"]
    return {
        "n": n,
        "supported": supported,
        "contradicted": contradicted,
        "unsupported": unsupported,
        "supported_rate": supported / n,
        "contradicted_rate": contradicted / n,
        "unsupported_rate": unsupported / n,
        "hallucination_rate": (contradicted + unsupported) / n,
        "supported_wilson95": tuple(round(v, 4) for v in wilson_interval(supported, n)),
    }


def class_rate_summary(labels: list[str], classes: tuple[str, ...]) -> dict:
    """Counts and rates for an arbitrary set of exclusive classes.

    ``rate_summary`` is specific to the label task (supported/contradicted/
    unsupported, where the hallucination rate is the complement of supported).
    The answer-copying instrument has different classes (passage/other/
    neither) and no such complement, so it gets this generic form.
    """
    if not labels:
        raise ValueError("class_rate_summary requires at least one label")
    unknown = set(labels) - set(classes)
    if unknown:
        raise ValueError(f"unknown label(s): {sorted(unknown)}")
    n = len(labels)
    counts = {c: labels.count(c) for c in classes}
    return {
        "n": n,
        "counts": counts,
        "rates": {c: counts[c] / n for c in classes},
        "wilson95": {c: tuple(round(v, 4) for v in wilson_interval(counts[c], n))
                     for c in classes},
    }


def swap_number(passage: str, old: str, new: str) -> str:
    """Replace ``old`` with ``new`` where ``old`` stands alone as a number.

    The guards prevent touching ``2015`` while swapping ``4``, and prevent
    splitting a decimal: a counterfactual that silently altered a second number
    would make the passage ambiguous and the classification meaningless.
    """
    old_n = extract_number(str(old))
    if old_n is None:
        raise ValueError(f"old value {old!r} is not a number")
    # Do not match a digit that is part of a longer number, a decimal, or a
    # grouped literal.
    pattern = re.compile(
        r"(?<![\d.,])" + re.escape(str(old).strip()) + r"(?![\d.,])")
    out, count = pattern.subn(str(new).strip(), passage)
    if count == 0:
        raise ValueError(f"{old!r} not found in passage as a standalone number")
    return out


# --------------------------------------------------------------------------
# Answer-copying measurement
# --------------------------------------------------------------------------
#
# Free generation cannot answer "does the model ground?" at this scale: the
# model rarely volunteers a digit, so a generation-based score confounds
# grounding with label competence. Measured on C4 with the passage in context,
# the model preferred the in-context number on 7 of 8 items while the same
# comparison on free text was 0 of 120 -- the model CAN ground; it just does
# not volunteer an answer.
#
# The instrument below removes the confound: it scores the model's likelihood
# of each candidate answer and asks WHICH one it prefers.


def number_token_count(tokenizer, number: str) -> int:
    """Token length of a number as the model would write it.

    A leading space is included because that is how the answer follows
    ``### Response:\\n``.
    """
    return len(tokenizer.encode(" " + str(number)))


def match_counterfactual(source: str, tokenizer, *, max_shift: int = 6) -> str:
    """A counterfactual number with the SAME digit count and SAME token count.

    Same digit count keeps the swap like-for-like: 9 -> 10 is not a fair
    counterpart to 35.

    Same TOKEN count is the subtler requirement. The BPE has merged tokens for
    some numbers and not others -- '35' is a single token while '34' is
    '3','4' -- so an unmatched pair would compare one cheap token against an
    expensive three-token sequence and hand the win to whichever side
    happened to get the merge. Candidates are searched in a deterministic
    order (+1, -1, +2, -2, ...) so the choice is reproducible.
    """
    src = str(source).strip()
    if not src.isdigit():
        raise ValueError(f"counterfactual matching needs a digit string, got {source!r}")
    n = int(src)
    target_tokens = number_token_count(tokenizer, src)
    for shift in range(1, max_shift + 1):
        for cand in (n + shift, n - shift):
            if cand < 0:
                continue
            text = str(cand)
            if len(text) != len(src):
                continue
            if number_token_count(tokenizer, text) == target_tokens:
                return text
    raise ValueError(
        f"no counterfactual for {source!r} shares its digit and token count")


def score_candidates(model, prompt_ids: list[int], candidates: list[str],
                     tokenizer, device: str) -> dict[str, dict[str, float]]:
    """Teacher-forced likelihood of each candidate as the model's answer.

    Returns, per candidate, the total log-probability and the mean per token.
    Total log-probability is the comparable quantity when candidates have the
    same token length, which :func:`match_counterfactual` guarantees for the
    source/counterfactual pair; the mean is reported alongside so a reader can
    see the comparison is not an artefact of length.
    """
    out: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for cand in candidates:
            cand_ids = tokenizer.encode(" " + str(cand))
            ids = list(prompt_ids) + cand_ids
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            logits, _, _ = model(x)
            logprobs = F.log_softmax(logits[0].float(), dim=-1)
            # Score ONLY the candidate tokens. Summing over ids[1:] would also
            # sum the PROMPT's own log-probabilities, which differ enormously
            # between a long oracle prompt and a short closed-book one -- so
            # the across-condition contrast was measuring the prompt's
            # likelihood, not the answer's. Within one prompt the error
            # cancels, which is why the oracle-vs-counterfactual margin looked
            # sane while the passage-lift contrast came out at -246 nats.
            p = len(prompt_ids)
            total = 0.0
            for k in range(len(cand_ids)):
                j = p - 1 + k  # logits[j] predicts ids[j + 1]
                total += float(logprobs[j, ids[j + 1]])
            out[str(cand)] = {
                "total_logprob": total,
                "mean_logprob": total / max(len(cand_ids), 1),
                "n_tokens": len(cand_ids),
            }
    return out


def classify_copy(chosen: str, source: str | None,
                  counterfactual: str | None) -> str:
    """Which of the tracked answers the model preferred.

    The classes are deliberately CONDITION-AGNOSTIC: 'source' and
    'counterfactual' are the two tracked numbers of the item, and 'neither' is
    a distractor. They are not "the number in the context", because for
    closed_book there IS no number in the context, and labelling by context
    would make the class unreachable by construction -- a tautology that reads
    as a measurement.

    Whether the chosen answer was the one actually in the context is reported
    separately, as ``followed_context``.
    """
    if source is not None and str(chosen) == str(source):
        return "source"
    if counterfactual is not None and str(chosen) == str(counterfactual):
        return "counterfactual"
    return "neither"
