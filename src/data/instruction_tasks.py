"""Synthetic instruction task family for Act 7, with genuinely disjoint splits.

Why synthetic: the canonical models are 1.5M-6.3M C4 language models. Asking them
to do MNLI zero-shot yields chance, which measures nothing about fine-tuning. The
question Act 7 asks is "does supervised adaptation transfer, and how does the
parameter budget shape it", and that needs a task where the base model starts
well BELOW a learnable ceiling but the ceiling is reachable. These tasks are
that: short, compositional, and exactly checkable.

Every task has a deterministic ground truth, so evaluation is exact match with a
Wilson interval rather than a subjective score.

Split integrity is enforced, not assumed: ``assert_disjoint`` checks that no
(input, response) pair appears in more than one split. Generating a held-out
split from the same seed as training is the easiest way to inflate a fine-tuning
result, and it has already happened once in this project (defect D6, BPE fitted
on validation text).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.adaptation.instructions import InstructionExample

TASKS = ("copy", "upper", "first_word", "last_word", "reverse_words", "count_words", "add")

_WORDS = (
    "river stone candle window bridge forest planet silver garden hammer "
    "meadow copper lantern thunder ribbon compass marble kettle harbour"
).split()


def _payload(rng: random.Random, min_words: int = 3, max_words: int = 6) -> list[str]:
    n = rng.randint(min_words, max_words)
    return [rng.choice(_WORDS) for _ in range(n)]


def make_example(task: str, rng: random.Random) -> tuple[InstructionExample, str]:
    """Return (example, expected_response). The response IS the ground truth.

    Keyword arguments throughout. ``InstructionExample``'s field order is
    ``(instruction, response, input_text)``, NOT ``(instruction, input_text)``.
    An earlier version passed the payload positionally, which bound it to
    ``response`` and left ``input_text`` empty -- so the prompt showed no input
    text at all while scoring still compared against it. Every input-dependent
    task was therefore unanswerable BY CONSTRUCTION, and the run produced
    near-zero accuracies that looked like a scientific negative result but were
    a plumbing error. ``test_input_text_is_visible_in_the_prompt`` guards this.
    """
    if task == "copy":
        text = " ".join(_payload(rng))
        return InstructionExample(instruction="Copy the text exactly.",
                                  response=text, input_text=text), text
    if task == "upper":
        text = " ".join(_payload(rng))
        return InstructionExample(instruction="Rewrite the text in capital letters.",
                                  response=text.upper(), input_text=text), text.upper()
    if task == "first_word":
        words = _payload(rng)
        text = " ".join(words)
        return InstructionExample(instruction="Give the first word of the text.",
                                  response=words[0], input_text=text), words[0]
    if task == "last_word":
        words = _payload(rng)
        text = " ".join(words)
        return InstructionExample(instruction="Give the last word of the text.",
                                  response=words[-1], input_text=text), words[-1]
    if task == "reverse_words":
        words = _payload(rng)
        text = " ".join(words)
        return InstructionExample(instruction="Write the words in reverse order.",
                                  response=" ".join(reversed(words)), input_text=text), " ".join(reversed(words))
    if task == "count_words":
        words = _payload(rng)
        text = " ".join(words)
        return InstructionExample(instruction="How many words are in the text?",
                                  response=str(len(words)), input_text=text), str(len(words))
    if task == "add":
        # The operands live in the instruction itself, so no input_text is needed
        # and the answer is derivable from the prompt alone.
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        return InstructionExample(instruction=f"What is {a} plus {b}?",
                                  response=str(a + b), input_text=""), str(a + b)
    raise ValueError(f"unknown task: {task}")


def build_split(tasks: tuple[str, ...], n: int, seed: int,
                add_pool: list[tuple[int, int]] | None = None) -> list[InstructionExample]:
    """Deterministic split. `seed` fully determines the split.

    ``add`` has only 100 distinct instances, so it must be drawn from a pool
    that is PARTITIONED ACROSS SPLITS. Shuffling per split made train and
    held-out share add items, which the disjointness assertion caught — and it
    would have silently inflated the held-out number, the same class of error as
    defect D6.
    """
    rng = random.Random(seed)
    add_i = 0
    out: list[InstructionExample] = []
    for i in range(n):
        task = tasks[i % len(tasks)]
        if task == "add":
            if add_pool is None or add_i >= len(add_pool):
                raise ValueError(
                    "the 'add' task has only 100 distinct instances and this split's "
                    "slice is exhausted; reduce n or drop the task"
                )
            a, b = add_pool[add_i]
            add_i += 1
            # The response must be filled in here; constructing it empty (as an
            # earlier version did) silently trains on blank targets.
            out.append(InstructionExample(instruction=f"What is {a} plus {b}?",
                                          response=str(a + b), input_text=""))
        else:
            example, _ = make_example(task, rng)
            out.append(example)
    return out


def assert_disjoint(splits: dict[str, list[InstructionExample]]) -> None:
    """Raise if any (prompt, response) pair appears in MORE THAN ONE split.

    Duplicates WITHIN a single split are permitted (oversampling a small task is
    legitimate); duplicates ACROSS splits are fatal, because a held-out item
    that also appears in training makes the held-out number meaningless.
    """
    owner: dict[tuple[str, str], str] = {}
    for split_name, examples in splits.items():
        for example in examples:
            key = (example.instruction + "|" + example.input_text, example.response)
            if key in owner and owner[key] != split_name:
                raise AssertionError(
                    f"example {key!r} appears in both {owner[key]!r} and {split_name!r}; "
                    "splits must be disjoint or the held-out number is meaningless"
                )
            owner[key] = split_name


@dataclass(frozen=True)
class TaskSplits:
    train: list[InstructionExample]
    held_out: list[InstructionExample]
    icl: list[InstructionExample]  # few-shot pool, disjoint from both


def make_splits(
    n_train: int = 512,
    n_held_out: int = 128,
    n_icl: int = 8,
    tasks: tuple[str, ...] = TASKS,
) -> TaskSplits:
    """Three disjoint splits.

    The 100 distinct ``add`` instances are shuffled ONCE and partitioned into
    contiguous slices sized to each split's demand, so no add item can appear
    in two splits. Everything else uses per-split seeds 10^6 apart. Disjointness
    is then asserted, not trusted.
    """
    add_tasks = 1 if "add" in tasks else 0
    demands = []
    if add_tasks:
        demands = [
            (n_train * add_tasks + len(tasks) - 1) // len(tasks),
            (n_held_out * add_tasks + len(tasks) - 1) // len(tasks),
            (n_icl * add_tasks + len(tasks) - 1) // len(tasks),
        ]
    pool = [(a, b) for a in range(10) for b in range(10)]
    random.Random(0).shuffle(pool)
    if add_tasks and sum(demands) > len(pool):
        raise ValueError(
            f"splits need {sum(demands)} distinct 'add' instances but only "
            f"{len(pool)} exist; reduce n or drop the task"
        )

    slices, cursor = [], 0
    for want in demands:
        slices.append(pool[cursor : cursor + want])
        cursor += want

    train = build_split(tasks, n_train, seed=1_000_000,
                        add_pool=slices[0] if add_tasks else None)
    held_out = build_split(tasks, n_held_out, seed=2_000_000,
                           add_pool=slices[1] if add_tasks else None)
    icl = build_split(tasks, n_icl, seed=3_000_000,
                      add_pool=slices[2] if add_tasks else None)
    assert_disjoint({"train": train, "held_out": held_out, "icl": icl})
    return TaskSplits(train=train, held_out=held_out, icl=icl)
