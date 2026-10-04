"""Reproducible zero-shot / few-shot / CoT evaluation of a trained checkpoint.

Decoding is greedy, so a run is fully determined by the checkpoint, task file,
``--shots`` and ``--max-new-tokens``. Usage (from the repository root)::

    .venv/bin/python -m src.prompting.evaluate --checkpoint artifacts/tick-204/bpe/checkpoint.pt

Prompts that exceed the model's context window are cropped from the left by the
decoder (the instructions and exemplars are lost first). Every item records
``prompt_truncated`` so this limitation is visible rather than silent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.generation import greedy_decode
from src.model import GPTConfig, MiniGPT
from src.tokenizer import tokenizer_from_checkpoint

from .prompts import STYLES, PromptTask, build_prompt, extract_answer, load_tasks, score_answer


def evaluate_prompting(
    model: MiniGPT, tokenizer, tasks: list[PromptTask], *, styles: tuple[str, ...] = STYLES, shots: int = 2, max_new_tokens: int = 48
) -> dict[str, object]:
    """Run every style over every task and return per-item records plus a summary."""
    model.eval()
    device = next(model.parameters()).device
    context_length = model.config.context_length
    items: list[dict[str, object]] = []
    for style in styles:
        for task in tasks:
            prompt = build_prompt(style, task.question, shots=shots)
            ids = tokenizer.encode(prompt)
            output = greedy_decode(
                model, torch.tensor([ids], dtype=torch.long, device=device),
                context_length=context_length, max_new_tokens=max_new_tokens,
            )
            generation = tokenizer.decode(output[0, len(ids):].tolist(), skip_special_tokens=True)
            extracted = extract_answer(style, generation)
            items.append({
                "style": style, "question": task.question, "accepted": list(task.answers),
                "prompt": prompt, "prompt_tokens": len(ids), "context_length": context_length,
                "prompt_truncated": len(ids) > context_length,
                "unknown_prompt_tokens": sum(1 for token in ids if token == 0),
                "generation": generation, "extracted": extracted,
                **score_answer(extracted, generation, task.answers),
            })
    summary: dict[str, dict[str, float]] = {}
    for style in styles:
        rows = [item for item in items if item["style"] == style]
        count = len(rows)
        summary[style] = {
            "n": count,
            "exact_accuracy": sum(bool(row["exact"]) for row in rows) / count,
            "contains_accuracy": sum(bool(row["contains"]) for row in rows) / count,
            "empty_extraction_fraction": sum(not row["extracted"] for row in rows) / count,
            "truncated_fraction": sum(bool(row["prompt_truncated"]) for row in rows) / count,
            "mean_prompt_tokens": sum(row["prompt_tokens"] for row in rows) / count,
            "mean_unknown_prompt_tokens": sum(row["unknown_prompt_tokens"] for row in rows) / count,
        }
    return {"summary": summary, "items": items}


def _markdown(summary: dict[str, dict[str, float]]) -> str:
    header = "| Style | n | Exact | Contains | Empty extraction | Prompt truncated | Mean prompt tokens | Mean <unk> in prompt |"
    lines = [header, "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for style, row in summary.items():
        lines.append(
            f"| {style} | {row['n']} | {row['exact_accuracy']:.2f} | {row['contains_accuracy']:.2f} | "
            f"{row['empty_extraction_fraction']:.2f} | {row['truncated_fraction']:.2f} | "
            f"{row['mean_prompt_tokens']:.1f} | {row['mean_unknown_prompt_tokens']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="artifacts/tick-204/bpe/checkpoint.pt")
    parser.add_argument("--tasks", default="data/prompt_tasks.jsonl")
    parser.add_argument("--output-dir", default="artifacts/tick-401")
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    model = MiniGPT(GPTConfig(**checkpoint["model_config"])).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    result = evaluate_prompting(model, tokenizer, load_tasks(args.tasks), shots=args.shots, max_new_tokens=args.max_new_tokens)

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    record = {"checkpoint": args.checkpoint, "tasks": args.tasks, "shots": args.shots, "max_new_tokens": args.max_new_tokens, **result}
    (root / "prompting.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    table = _markdown(result["summary"])
    samples = "\n\n".join(
        f"**{item['style']} — {item['question']}** (truncated={item['prompt_truncated']}, extracted={item['extracted']!r})\n```\n{item['generation'][:200]}\n```"
        for item in result["items"]
    )
    (root / "prompting.md").write_text(table + "\n\n" + samples + "\n", encoding="utf-8")
    print(table)
    print(f"\nWrote {root / 'prompting.json'} and {root / 'prompting.md'}")


if __name__ == "__main__":
    main()
