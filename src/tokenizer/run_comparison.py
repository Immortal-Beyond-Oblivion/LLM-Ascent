"""Act 3 comparison: char, BPE, WordPiece, Unigram LM on the same model.

Includes the equal-character-span control (D9): matches the number of
characters seen per step across tokenizers.

Usage:
    .venv/bin/python -m src.tokenizer.run_comparison
    .venv/bin/python -m src.tokenizer.run_comparison --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from src.bench.experiment import MetricsSchema
from src.bench.seeds import set_seed
from src.bench.stats import mean, std
from src.data import TextDataset, load_text
from src.model.gpt import GPTConfig, MiniGPT
from src.tokenizer import CharacterTokenizer, BPETokenizer, WordPieceTokenizer, UnigramLMTokenizer
from src.training.train import resolve_device, _lr_at_step


TOKENIZER_NAMES = ("character", "bpe", "wordpiece", "unigram")


def build_tokenizer(kind: str, text: str, max_merges: int = 100, train_fraction: float = 0.9):
    """Build a tokenizer of the given kind on the train split (D6 fix)."""
    split_idx = int(len(text) * train_fraction)
    train_text = text[:split_idx]

    if kind == "character":
        return CharacterTokenizer.from_text(train_text)
    elif kind == "bpe":
        return BPETokenizer.train(train_text, max_merges=max_merges)
    elif kind == "wordpiece":
        return WordPieceTokenizer.train(train_text, max_merges=max_merges)
    elif kind == "unigram":
        return UnigramLMTokenizer.train(train_text, target_vocab_size=1 + len(sorted(set(train_text))) + max_merges)
    else:
        raise ValueError(f"unknown tokenizer: {kind}")


def train_tokenizer_model(
    tokenizer,
    text: str,
    *,
    tokenizer_kind: str = "unknown",
    steps: int = 500,
    batch_size: int = 16,
    context_length: int = 64,
    learning_rate: float = 1e-3,
    warmup_steps: int = 0,
    eval_interval: int = 100,
    eval_batches: int = 5,
    seed: int = 42,
    device: str = "cpu",
    equal_char_span: bool = False,
    chars_per_token: float | None = None,
) -> MetricsSchema:
    set_seed(seed)
    device = resolve_device(device)

    dataset = TextDataset.from_text(text, tokenizer, train_fraction=0.9)

    # Compute characters per token for equal-char-span control
    if chars_per_token is None:
        val_tokens = dataset.validation.tolist()
        val_chars = len(tokenizer.decode(val_tokens))
        chars_per_token = val_chars / len(val_tokens)

    # Equal-char-span: adjust context length to match ~64 characters
    if equal_char_span:
        target_chars = 64
        context_length = max(4, int(target_chars / chars_per_token))

    model_config = None
    # Need to compute these after determining context_length
    from src.model.gpt import GPTConfig, MiniGPT

    model_config = GPTConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        context_length=context_length,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        normalization="layernorm",
        optimized_attention=True,
    )
    model = MiniGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loss_history = []

    for step in range(1, steps + 1):
        lr = _lr_at_step_simple(step, learning_rate, warmup_steps, steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = dataset.batch(
            "train", batch_size=batch_size, context_length=context_length, device=device
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % eval_interval == 0 or step == steps:
            model.eval()
            val_losses, train_losses = [], []
            with torch.no_grad():
                for _ in range(eval_batches):
                    vi, vt = dataset.batch("validation", batch_size=batch_size, context_length=context_length, device=device)
                    _, vloss = model(vi, vt)
                    assert vloss is not None
                    val_losses.append(float(vloss.detach()))
                for _ in range(eval_batches):
                    ti, tt = dataset.batch("train", batch_size=batch_size, context_length=context_length, device=device)
                    _, tloss = model(ti, tt)
                    assert tloss is not None
                    train_losses.append(float(tloss.detach()))
            model.train()

            loss_history.append({
                "step": float(step),
                "train_loss": sum(train_losses) / len(train_losses),
                "validation_loss": sum(val_losses) / len(val_losses),
                "validation_perplexity": math.exp(sum(val_losses) / len(val_losses)),
            })

    final = loss_history[-1]
    experiment_name = f"{tokenizer_kind}-seed{seed}-ec{'s' if equal_char_span else 'n'}"
    metrics = MetricsSchema(
        act="act3-tokenizers",
        experiment=experiment_name,
        seed=seed,
        device=device,
        config={"tokenizer": tokenizer_kind, "equal_char_span": equal_char_span, "context_length": context_length, "steps": steps},
        parameters=model.parameter_count(),
        trainable_parameters=model.parameter_count(),
        total_tokens=len(dataset.train) + len(dataset.validation),
        train_tokens=len(dataset.train),
        validation_tokens=len(dataset.validation),
        vocabulary_size=tokenizer.vocabulary_size,
        loss_history=loss_history,
        wall_clock_seconds=0.0,
        steps_per_second=0.0,
        final_step=float(final["step"]),
        final_train_loss=final["train_loss"],
        final_validation_loss=final["validation_loss"],
        final_validation_perplexity=final["validation_perplexity"],
    )
    metrics.validate()

    # Save sidecar with BPC
    result_dir = Path("results/act3-tokenizers") / "act3-tokenizers" / experiment_name
    result_dir.mkdir(parents=True, exist_ok=True)
    bpc = final["validation_loss"] / math.log(2) / chars_per_token
    sidecar = {
        "tokenizer": tokenizer_kind,
        "vocabulary_size": tokenizer.vocabulary_size,
        "chars_per_token": chars_per_token,
        "validation_bits_per_character": bpc,
        "validation_perplexity_per_character": 2.0 ** bpc,
        "equal_char_span": equal_char_span,
        "context_length_used": context_length,
    }
    (result_dir / "tokenizer_stats.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    (result_dir / "metrics.json").write_text(metrics.to_json(), encoding="utf-8")

    print(f"[act3] {tokenizer_kind} (ec={equal_char_span}) seed={seed}: val_loss={final['validation_loss']:.4f} bpc={bpc:.4f} ctx={context_length}")
    return metrics


def _lr_at_step_simple(step, peak_lr, warmup_steps, total_steps):
    if warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="data/tinyshakespeare.txt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--max-merges", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="results/act3-tokenizers")
    args = parser.parse_args()

    text = load_text(args.dataset_path)
    device = resolve_device(args.device)
    print(f"[act3] device={device} seeds={args.seeds}")

    all_results = {}
    for kind in TOKENIZER_NAMES:
        all_results[kind] = {"standard": [], "equal_char_span": []}
        for seed in args.seeds:
            tokenizer = build_tokenizer(kind, text, max_merges=args.max_merges)

            # Standard comparison
            m1 = train_tokenizer_model(
                tokenizer, text, tokenizer_kind=kind, steps=args.steps, batch_size=args.batch_size,
                context_length=args.context_length, seed=seed, device=device,
                equal_char_span=False,
            )
            all_results[kind]["standard"].append(m1)

            # Equal-char-span control (D9)
            tok2 = build_tokenizer(kind, text, max_merges=args.max_merges)
            m2 = train_tokenizer_model(
                tok2, text, tokenizer_kind=kind, steps=args.steps, batch_size=args.batch_size,
                context_length=args.context_length, seed=seed, device=device,
                equal_char_span=True,
            )
            all_results[kind]["equal_char_span"].append(m2)

    # Aggregate
    summary = {}
    for kind in TOKENIZER_NAMES:
        for control in ("standard", "equal_char_span"):
            runs = all_results[kind][control]
            bpcs = []
            for m in runs:
                name = m.experiment
                result_dir = Path("results/act3-tokenizers") / "act3-tokenizers" / name
                if (result_dir / "tokenizer_stats.json").exists():
                    sidecar = json.loads((result_dir / "tokenizer_stats.json").read_text())
                    bpcs.append(sidecar["validation_bits_per_character"])
            if bpcs:
                key = f"{kind}-{control}"
                summary[key] = {
                    "bpc_mean": mean(bpcs),
                    "bpc_std": std(bpcs) if len(bpcs) > 1 else 0.0,
                    "n_seeds": len(bpcs),
                    "per_seed_bpc": bpcs,
                }

    summary_path = Path(args.output_root) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Act 3: Tokenizer Comparison (4 tokenizers) ===")
    for control in ("standard", "equal_char_span"):
        print(f"\n--- {control} control ---")
        print(f"{'Tokenizer':<12} {'BPC mean':>10} {'BPC std':>10} {'n':>5}")
        print("-" * 32)
        for kind in TOKENIZER_NAMES:
            key = f"{kind}-{control}"
            if key in summary:
                s = summary[key]
                print(f"{kind:<12} {s['bpc_mean']:>10.4f} {s['bpc_std']:>10.4f} {s['n_seeds']:>5}")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
