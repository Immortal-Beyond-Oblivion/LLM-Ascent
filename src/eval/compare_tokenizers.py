"""Matched character-vs-BPE experiment runner for ``[TICK-204]``.

Both runs share the corpus, 90/10 split rule, model dimensions, optimizer,
seed, batch size, step count, and evaluation batches. Only ``tokenizer`` (and
therefore vocabulary-dependent embedding size and sequence length) changes.

Per-token loss/perplexity is *not* comparable across tokenizers because a BPE
token covers several characters. This runner therefore also reports
bits-per-character (BPC) and character-equivalent perplexity, derived from the
validation loss and the measured characters-per-token ratio of the validation
split.

Usage (from the repository root)::

    .venv/bin/python -m src.eval.compare_tokenizers
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from src.data import TextDataset, load_text
from src.generation.decoding import generate
from src.training import TrainConfig, train

from .metrics import repetition_rate

TOKENIZERS = ("character", "bpe")


def _sample(model, tokenizer, prompt: str, *, context_length: int, chars: int, sampled: bool, seed: int) -> dict[str, object]:
    """Generate ``chars`` tokens, time it, and trim the text to ``chars`` characters."""
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    if sampled:
        torch.manual_seed(seed)
    start = time.perf_counter()
    output = generate(
        model, prompt_tensor, context_length=context_length, max_new_tokens=chars,
        temperature=0.8 if sampled else 1.0, top_k=20 if sampled else None, do_sample=sampled,
    )
    elapsed = time.perf_counter() - start
    new_ids = output[0, len(prompt_ids):].tolist()
    full_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return {
        "mode": "top-k sampling (k=20, T=0.8)" if sampled else "greedy",
        "text": full_text[:chars],
        "repetition_3gram": repetition_rate(full_text[:chars], ngram_size=3),
        "generated_tokens": len(new_ids),
        "generated_characters": len(full_text),
        "seconds": elapsed,
        "tokens_per_second": len(new_ids) / elapsed if elapsed > 0 else float("inf"),
        "characters_per_second": len(full_text) / elapsed if elapsed > 0 else float("inf"),
    }


def run_experiment(kind: str, args: argparse.Namespace, root: Path, text: str) -> dict[str, object]:
    config = TrainConfig(
        dataset_path=args.dataset_path, output_dir=str(root / kind), steps=args.steps,
        batch_size=args.batch_size, context_length=args.context_length, embedding_dim=args.embedding_dim,
        num_heads=args.num_heads, num_layers=args.num_layers, learning_rate=args.learning_rate,
        eval_interval=args.eval_interval, eval_batches=args.eval_batches, seed=args.seed,
        device=args.device, tokenizer=kind, bpe_merges=args.bpe_merges,
    )
    start = time.perf_counter()
    model, tokenizer, history = train(config)
    train_seconds = time.perf_counter() - start
    model.eval()

    # Re-derive the split exactly as `train` did to measure characters per token.
    dataset = TextDataset.from_text(text, tokenizer)
    validation_tokens = dataset.validation.tolist()
    validation_characters = len(tokenizer.decode(validation_tokens))
    characters_per_token = validation_characters / len(validation_tokens)

    final = history[-1]
    validation_loss = final["validation_loss"]
    bits_per_character = validation_loss / math.log(2) / characters_per_token
    sample_args = dict(context_length=config.context_length, chars=args.sample_chars, seed=args.seed)
    return {
        "tokenizer": kind,
        "vocabulary_size": tokenizer.vocabulary_size,
        "merges": len(getattr(tokenizer, "merges", ())),
        "parameters": model.parameter_count(),
        "total_tokens": len(dataset.train) + len(dataset.validation),
        "validation_tokens": len(validation_tokens),
        "characters_per_token": characters_per_token,
        "train_seconds": train_seconds,
        "final_step": final["step"],
        "train_loss": final["train_loss"],
        "validation_loss": validation_loss,
        "validation_perplexity_per_token": final["validation_perplexity"],
        "validation_bits_per_character": bits_per_character,
        "validation_perplexity_per_character": 2.0**bits_per_character,
        "loss_history": history,
        "samples": [
            _sample(model, tokenizer, args.prompt, sampled=False, **sample_args),
            _sample(model, tokenizer, args.prompt, sampled=True, **sample_args),
        ],
        "config": config.__dict__,
    }


def _markdown(results: list[dict[str, object]]) -> str:
    def row(label: str, key: str, fmt: str = "{}") -> str:
        return f"| {label} | " + " | ".join(fmt.format(result[key]) for result in results) + " |"

    def sample_row(label: str, index: int, key: str, fmt: str) -> str:
        return f"| {label} | " + " | ".join(fmt.format(result["samples"][index][key]) for result in results) + " |"

    lines = [
        "| Metric | " + " | ".join(str(result["tokenizer"]) for result in results) + " |",
        "|---|" + "---:|" * len(results),
        row("Vocabulary size", "vocabulary_size"),
        row("BPE merges", "merges"),
        row("Parameters", "parameters", "{:,}"),
        row("Characters per token (validation)", "characters_per_token", "{:.3f}"),
        row("Train loss (per token)", "train_loss", "{:.4f}"),
        row("Validation loss (per token)", "validation_loss", "{:.4f}"),
        row("Validation perplexity (per token, NOT cross-comparable)", "validation_perplexity_per_token", "{:.2f}"),
        row("Validation bits per character", "validation_bits_per_character", "{:.4f}"),
        row("Validation perplexity (per character)", "validation_perplexity_per_character", "{:.3f}"),
        row("Training wall-clock (s)", "train_seconds", "{:.1f}"),
        sample_row("Greedy repetition (char 3-gram)", 0, "repetition_3gram", "{:.3f}"),
        sample_row("Sampled repetition (char 3-gram)", 1, "repetition_3gram", "{:.3f}"),
        sample_row("Greedy tokens/s", 0, "tokens_per_second", "{:.1f}"),
        sample_row("Greedy characters/s", 0, "characters_per_second", "{:.1f}"),
        sample_row("Sampled characters/s", 1, "characters_per_second", "{:.1f}"),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", default="data/tinyshakespeare.txt")
    parser.add_argument("--output-dir", default="artifacts/tick-204")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--bpe-merges", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--sample-chars", type=int, default=200)
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    text = load_text(args.dataset_path)
    results = []
    for kind in TOKENIZERS:
        print(f"[TICK-204] running {kind} experiment ...", flush=True)
        results.append(run_experiment(kind, args, root, text))

    (root / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    table = _markdown(results)
    samples = "\n\n".join(
        f"**{result['tokenizer']} / {sample['mode']}**\n```\n{args.prompt}{sample['text']}\n```"
        for result in results
        for sample in result["samples"]
    )
    (root / "comparison.md").write_text(table + "\n\n" + samples + "\n", encoding="utf-8")
    print(table)
    print(f"\nWrote {root / 'comparison.json'} and {root / 'comparison.md'}")


if __name__ == "__main__":
    main()
