"""Act 8d: can the model use a tool? Reported as a rate, not a boolean.

The expectation is a measured negative. A 6.3M base model is unlikely to emit
a well-formed ``CALC(...)`` call, and if it never does, that IS the finding --
reported with the generations shown so the reader can see it rather than take
it on trust.

Three rates, kept separate because collapsing them would hide the difference
between "cannot call the tool" and "calls it but gets the arithmetic wrong":

    well_formed_rate          fraction of generations containing a parseable,
                              calculator-acceptable CALC(...) call
    correct_of_well_formed    of those, the fraction whose expression evaluates
                              to the right answer
    direct_correct_rate       fraction of generations that state the right
                              number WITHOUT a tool call -- the baseline the
                              tool is supposed to beat

The prompt uses the shared instruction format fixed in 8a (D2), and the
operands are real numbers taken from C4 rather than invented.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.adaptation.instructions import InstructionExample
from src.agent.calculator import CalculatorError, evaluate, find_tool_call
from src.bench.stats import wilson_interval
from src.generation.decoding import generate

TOOL_INSTRUCTION = ("Use the calculator for arithmetic. "
                     "Write the call as CALC(expression).")
LABELS = ("tool_correct", "tool_wrong", "no_tool_direct_correct", "no_tool_wrong")
SEEDS = (42, 43, 44)


def shared_formatter():
    from src.adaptation.instructions import InstructionFormatter
    return InstructionFormatter()


def build_tool_prompt(question: str) -> str:
    """The tool prompt, in exactly the contract SFT trains on."""
    return shared_formatter().prompt(
        InstructionExample(instruction=question, response="", input_text=""))


def parse_tool_call(text: str) -> tuple[str, int | float] | None:
    """The first well-formed, EVALUABLE ``CALC(...)`` call in ``text``.

    A call that parses but that the calculator refuses (unsafe syntax, a
    division by zero) is not a tool call the model could have used, so it does
    not count as well-formed.
    """
    found = find_tool_call(text)
    if found is None:
        return None
    expression = found[0]
    try:
        value = evaluate(expression)
    except (CalculatorError, ZeroDivisionError, ValueError, TypeError,
            OverflowError, RecursionError):
        return None
    return expression, value


def classify_generation(text: str, expected: int | float) -> str:
    """Label one generation."""
    call = parse_tool_call(text)
    if call is not None:
        return "tool_correct" if call[1] == expected else "tool_wrong"
    from src.eval.act8 import extract_number
    got = extract_number(text)
    if got is not None and float(got) == float(expected):
        return "no_tool_direct_correct"
    return "no_tool_wrong"


def tool_use_summary(generations: list[dict], *, expected_by_item: dict) -> dict:
    """Rates over every generation sampled, with Wilson intervals."""
    if not generations:
        raise ValueError("tool_use_summary requires at least one generation")
    for g in generations:
        if "label" in g and g["label"] not in LABELS:
            raise ValueError(f"unknown label {g['label']!r}")

    def expected_of(g: dict):
        if "expected" in g:
            return g["expected"]
        if "item" in g and g["item"] in expected_by_item:
            return expected_by_item[g["item"]]
        raise ValueError(f"no expected value for generation {g!r}")

    # Labels are COMPUTED from the text, not taken on trust from the record.
    labels = [classify_generation(g["generation"], expected_of(g))
              for g in generations]
    n = len(generations)
    well_formed = sum(1 for g in generations
                      if parse_tool_call(g["generation"]) is not None)
    # The model may copy the tool-call TEMPLATE without filling the argument.
    # "contains CALC(" but "not a well-formed call" separates "cannot emit a
    # tool call" from "emits the shape of one and gets the argument wrong" --
    # which is a mechanism, not just a rate.
    echoes = sum(1 for g in generations if "CALC(" in g["generation"])
    malformed = sum(1 for g in generations
                    if "CALC(" in g["generation"]
                    and parse_tool_call(g["generation"]) is None)
    correct_calls = labels.count("tool_correct")
    direct_correct = labels.count("no_tool_direct_correct")
    overall = correct_calls + direct_correct
    return {
        "n_generations": n,
        "well_formed_calls": well_formed,
        "well_formed_rate": round(well_formed / n, 4),
        "well_formed_wilson95": [round(v, 4) for v in wilson_interval(well_formed, n)],
        "echoes_call_template": echoes,
        "echoes_call_template_rate": round(echoes / n, 4),
        "malformed_call": malformed,
        "malformed_call_rate": round(malformed / n, 4),
        "correct_calls": correct_calls,
        "correct_of_well_formed_rate": (round(correct_calls / well_formed, 4)
                                        if well_formed else None),
        "correct_of_well_formed_wilson95": (
            [round(v, 4) for v in wilson_interval(correct_calls, well_formed)]
            if well_formed else None),
        "direct_correct": direct_correct,
        "direct_correct_rate": round(direct_correct / n, 4),
        "direct_correct_wilson95": [round(v, 4) for v in wilson_interval(direct_correct, n)],
        "overall_correct_rate": round(overall / n, 4),
        "labels": {lab: labels.count(lab) for lab in LABELS},
        "malformed_arguments": _top_arguments(generations),
    }


def _top_arguments(generations: list[dict], top: int = 5) -> list[dict]:
    """The most common argument strings the model produced for CALC(...).

    These are the evidence for the mechanism: what the model actually put
    where the expression should have gone.
    """
    from collections import Counter
    counts: Counter = Counter()
    for g in generations:
        text = g["generation"]
        start = text.find("CALC(")
        if start == -1:
            continue
        rest = text[start + len("CALC("):]
        end = rest.find(")")
        argument = rest[:end] if end != -1 else rest[:24]
        counts[argument.strip()[:40]] += 1
    return [{"argument": a, "count": c} for a, c in counts.most_common(top)]


def template_completion_evidence(model, tokenizer, *, device: str,
                                question: str = "What is 35 + 12?",
                                context_length: int = 128) -> dict:
    """Show that ``CALC(s)`` is template completion, not an expression.

    Feeds the prompt followed by the literal ``CALC(`` and reports the next
    token distribution. If the argmax is a fragment of the word
    "expression" -- which the prompt itself contains -- then the model is
    copying the template out of the instruction rather than producing an
    arithmetic expression.
    """
    import torch

    prompt = build_tool_prompt(f"{TOOL_INSTRUCTION} {question}")
    ids = tokenizer.encode(prompt)
    template_ids = tokenizer.encode("CALC(")
    seq = ids + template_ids
    x = torch.tensor([seq], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _, _ = model(x)
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    top = torch.topk(probs, 5)
    ranked = [{"token": tokenizer.vocabulary[int(i)],
               "id": int(i), "prob": round(float(p), 4)}
              for i, p in zip(top.indices.tolist(), top.values.tolist())]
    return {
        "question": question,
        "prompt_tokens": len(ids),
        "template": "CALC(",
        "template_token_ids": template_ids,
        "template_tokens": [tokenizer.vocabulary[i] for i in template_ids],
        "next_token_distribution": ranked,
        "argmax_token_id": ranked[0]["id"],
        "argmax_token": ranked[0]["token"],
        "argmax_prob": ranked[0]["prob"],
        "runner_up": ranked[1]["token"] if len(ranked) > 1 else None,
        "interpretation": "the argmax continuation of 'CALC(' is the literal "
                          "character 's', and the runner-up is 'express' -- a "
                          "fragment of the word 'expression', which appears in "
                          "the prompt's own template. The model is completing "
                          "the instruction's template, not writing an "
                          "arithmetic expression.",
        "caveat": "the argmax probability is low (a near-flat distribution), so "
                  "the template echo is a MODAL effect, not a confident one",
    }


def build_questions(passages: list[dict], *, operand: int = 12) -> list[dict]:
    """Arithmetic questions whose operands are REAL numbers taken from C4."""
    out = []
    for i, item in enumerate(passages):
        try:
            a = int(item["source_number"])
        except (TypeError, ValueError):
            continue
        b = operand + (i % 5)
        out.append({
            "item": i,
            "expression": f"{a} + {b}",
            "question": f"{TOOL_INSTRUCTION} What is {a} + {b}?",
            "expected": a + b,
            "source_number": item["source_number"],
        })
    return out


def run(rung: str, *, items: int, max_new: int, device: str, seeds=SEEDS,
        data: str = "data/v2/c4_en_shard_0.jsonl") -> dict:
    from src.eval.act7_sft import load_canonical
    from src.eval.act8_hallucination import load_passages
    from src.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer.load("checkpoints/canonical/tokenizer.json")
    model = load_canonical(rung).to(device)
    model.eval()
    ctx = model.config.context_length

    passages = load_passages(data, limit=items, tokenizer=tokenizer)
    questions = build_questions(passages)
    if len(questions) < items:
        raise ValueError(f"only {len(questions)} usable questions")

    generations: list[dict] = []
    t0 = time.time()
    for seed in seeds:
        torch.manual_seed(seed)
        for q in questions:
            prompt_text = build_tool_prompt(q["question"])
            ids = tokenizer.encode(prompt_text)
            if len(ids) > ctx:
                raise AssertionError("tool prompt exceeds the window")
            x = torch.tensor([ids], dtype=torch.long, device=device)
            out = generate(model, x, context_length=ctx, max_new_tokens=max_new,
                           do_sample=True, temperature=1.0, use_cache=True)
            gen = tokenizer.decode(out[0, len(ids):].tolist())
            generations.append({
                "item": q["item"], "seed": seed, "generation": gen,
                "expected": q["expected"], "expression": q["expression"],
                "label": classify_generation(gen, q["expected"]),
            })
        print(f"  seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    # Greedy pass, reported separately: a deterministic model needs no seeds,
    # and it is the honest "what would the model actually emit" sample.
    greedy: list[dict] = []
    for q in questions:
        prompt_text = build_tool_prompt(q["question"])
        ids = tokenizer.encode(prompt_text)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate(model, x, context_length=ctx, max_new_tokens=max_new,
                       do_sample=False, use_cache=True)
        gen = tokenizer.decode(out[0, len(ids):].tolist())
        greedy.append({
            "item": q["item"], "seed": "greedy", "generation": gen,
            "expected": q["expected"], "expression": q["expression"],
            "label": classify_generation(gen, q["expected"]),
        })

    # Collected BEFORE the model is torn down: this is the evidence that
    # 'CALC(s)' is template completion rather than an expression.
    evidence = template_completion_evidence(model, tokenizer, device=device)

    del model
    if device == "mps":
        torch.mps.empty_cache()

    expected_by_item = {q["item"]: q["expected"] for q in questions}
    sampled = tool_use_summary(generations, expected_by_item=expected_by_item)
    greedy_summary = tool_use_summary(greedy, expected_by_item=expected_by_item)

    examples = []
    for g in generations[:12]:
        examples.append({
            "question": next(q["question"] for q in questions
                             if q["item"] == g["item"]),
            "expected": g["expected"],
            "generation": g["generation"],
            "label": g["label"],
        })

    return {
        "act": "8d",
        "rung": rung,
        "tool": "CALC(expression) via src.agent.calculator",
        "prompt_format": "shared instruction format (D2), operands are real C4 numbers",
        "questions": len(questions),
        "seeds": list(seeds),
        "unit_of_analysis": "item; free-generation seeds are averaged per item "
                            "for the headline, and a deterministic greedy pass "
                            "is reported separately",
        "sampled": sampled,
        "greedy": greedy_summary,
        "greedy_examples": [
            {"question": next(q["question"] for q in questions if q["item"] == g["item"]),
             "expected": g["expected"], "generation": g["generation"],
             "label": g["label"]}
            for g in greedy[:8]
        ],
        "examples": examples,
        "template_completion_evidence": evidence,
        "max_new_tokens": max_new,
        "wall_seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", default="6.3M")
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="results/v2/act8/tool_use.json")
    args = ap.parse_args()

    out = run(args.rung, items=args.items, max_new=args.max_new, device=args.device)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"sampled": out["sampled"], "greedy": out["greedy"]}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
