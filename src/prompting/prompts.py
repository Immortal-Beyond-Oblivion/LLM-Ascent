"""Zero-shot, few-shot, and chain-of-thought prompt construction and scoring.

Pure Python (no PyTorch) so templates and scorers are testable anywhere.
Numbers are spelled out in exemplars and tasks because the character corpus
used for the baseline has almost no digits (they would encode as ``<unk>``).
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path

STYLES = ("zero_shot", "few_shot", "cot")
COT_CUE = "Let's think step by step."


@dataclass(frozen=True)
class Exemplar:
    question: str
    reasoning: str
    answer: str


@dataclass(frozen=True)
class PromptTask:
    question: str
    answers: tuple[str, ...]


# Deliberately disjoint from `data/prompt_tasks.jsonl` (questions and answers).
EXEMPLARS = (
    Exemplar("What is one plus one?", "One and one more make two.", "two"),
    Exemplar("What color is grass?", "Grass is a green plant.", "green"),
    Exemplar("How many legs does a spider have?", "A spider is an arachnid with eight legs.", "eight"),
)


def build_prompt(style: str, question: str, *, shots: int = 2) -> str:
    """Build a prompt in one of the three styles.

    ``few_shot`` shows ``shots`` question/answer pairs; ``cot`` shows worked
    reasoning for the same exemplars and ends with the step-by-step cue. With
    ``shots=0``, ``few_shot`` equals ``zero_shot`` and ``cot`` is zero-shot CoT.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    if not 0 <= shots <= len(EXEMPLARS):
        raise ValueError(f"shots must be between 0 and {len(EXEMPLARS)}")
    if not question.strip():
        raise ValueError("question must not be empty")
    if style == "zero_shot":
        return f"Q: {question}\nA:"
    blocks = []
    for exemplar in EXEMPLARS[:shots]:
        if style == "few_shot":
            blocks.append(f"Q: {exemplar.question}\nA: {exemplar.answer}")
        else:
            blocks.append(f"Q: {exemplar.question}\nA: {COT_CUE} {exemplar.reasoning} The answer is {exemplar.answer}.")
    tail = f"Q: {question}\nA:" if style == "few_shot" else f"Q: {question}\nA: {COT_CUE}"
    return "\n\n".join([*blocks, tail])


def clip_generation(text: str) -> str:
    """Cut a continuation at the first blank line or hallucinated next question."""
    for marker in ("\n\n", "\nQ:"):
        index = text.find(marker)
        if index != -1:
            text = text[:index]
    return text.strip()


def extract_answer(style: str, generation: str) -> str:
    """Pull the final answer: the 'The answer is ...' span for CoT, else the first line."""
    clipped = clip_generation(generation)
    if style == "cot":
        match = re.search(r"the answer is\s+([^.\n]+)", clipped, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    return clipped.splitlines()[0].strip() if clipped else ""


def normalize(text: str) -> str:
    return " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def score_answer(extracted: str, generation: str, accepted: tuple[str, ...] | list[str]) -> dict[str, bool]:
    """Strict ``exact`` match of the extracted answer, plus lenient word-boundary ``contains``."""
    accepted_normalized = {normalize(answer) for answer in accepted}
    haystack = f" {normalize(clip_generation(generation))} "
    return {
        "exact": normalize(extracted) in accepted_normalized,
        "contains": any(f" {answer} " in haystack for answer in accepted_normalized),
    }


def load_tasks(path: str | Path) -> list[PromptTask]:
    """Read JSONL rows shaped ``{"question": ..., "answers": [...]}``."""
    tasks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            tasks.append(PromptTask(row["question"], tuple(row["answers"])))
    if not tasks:
        raise ValueError(f"no tasks found in {path}")
    return tasks
