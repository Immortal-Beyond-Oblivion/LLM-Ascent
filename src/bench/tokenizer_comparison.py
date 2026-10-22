"""Tokenizer comparison on the A2 experiment framework.

Fixes D6 (BPE fitted on train split only) and D7 (seeded full-sweep validation).
Produces uniform metrics.json under results/<act>/<experiment>/.

Usage:
    .venv/bin/python -m src.bench.tokenizer_comparison
    .venv/bin/python -m src.bench.tokenizer_comparison --seeds 42 43 44 --steps 500
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from src.bench.experiment import ExperimentConfig, MetricsSchema, run_experiment, load_metrics
from src.bench.seeds import set_seed
from src.bench.stats import mean, std, wilson_interval
from src.data import TextDataset, load_text
from src.generation.decoding import generate
from src.model import GPTConfig, MiniGPT
from src.tokenizer import BPETokenizer, CharacterTokenizer
from src.training.train import resolve_device

from src.eval.metrics import repetition_rate


def _build_tokenizer_train_only(
    text: str, config: ExperimentConfig, train_fraction: float = 0.9, tokenizer_kind: str | None = None
) -> CharacterTokenizer | BPETokenizer:
    """Fit tokenizer on train split only (D6 fix)."""
    kind = tokenizer_kind or config.tokenizer
    split_index = int(len(text) * train_fraction)
    train_text = text[:split_index]
    if kind == "character":
        return CharacterTokenizer.from_text(train_text)
    if kind == "bpe":
        return BPETokenizer.train(train_text, max_merges=config.bpe_merges)
    raise ValueError(f"unknown tokenizer: {kind}")


def _evaluate_full_sweep(
    model: MiniGPT,
    dataset: TextDataset,
    config: ExperimentConfig,
    device: str,
    split: str = "validation",
) -> float:
    """Seeded full-sweep validation (D7 fix).

    Iterates over the entire validation set in fixed-order batches with a
    seeded generator, rather than sampling random batches. Pre-builds all
    batches as tensors for efficiency.
    """
    set_seed(config.seed)  # seeded for reproducibility
    data = dataset.validation if split == "validation" else dataset.train
    # Use larger batches for validation to reduce Python loop overhead
    batch_size = max(config.batch_size, 512)
    context_length = config.context_length

    # Pre-build all batches as tensors (efficient)
    num_sequences = len(data) - context_length - 1
    indices = torch.arange(num_sequences)
    # Build inputs and targets in one shot using unfold
    unfolded = data.unfold(0, context_length + 1, 1)  # [num_seq, context+1]
    all_inputs = unfolded[:, :context_length]  # [num_seq, context]
    all_targets = unfolded[:, 1:]  # [num_seq, context]

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, num_sequences, batch_size):
            end = min(start + batch_size, num_sequences)
            inputs = all_inputs[start:end].to(device)
            targets = all_targets[start:end].to(device)
            _, loss, _ = model(inputs, targets)
            assert loss is not None
            num_targets = targets.numel()
            total_loss += float(loss.detach()) * num_targets
            total_tokens += num_targets

    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def run_tokenizer_comparison(
    tokenizer_kind: str,
    config: ExperimentConfig,
) -> MetricsSchema:
    """Run one tokenizer configuration under the A2 framework with D6/D7 fixed."""
    set_seed(config.seed)
    device = resolve_device(config.device)
    print(f"[tokenizer-comparison] tokenizer={tokenizer_kind} device={device} seed={config.seed}")

    text = load_text(config.dataset_path)

    # D6 fix: fit tokenizer on train split only
    tokenizer = _build_tokenizer_train_only(
        text, config, train_fraction=config.train_fraction, tokenizer_kind=tokenizer_kind
    )

    # Encode full text with the train-fitted tokenizer
    dataset = TextDataset.from_text(text, tokenizer, train_fraction=config.train_fraction)

    # Build model
    model_config = GPTConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        normalization=config.normalization,
        optimized_attention=config.optimized_attention,
    )
    model = MiniGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    loss_history: list[dict[str, float]] = []
    start_time = time.perf_counter()

    for step in range(1, config.steps + 1):
        lr = _lr_at_step(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = dataset.batch(
            "train",
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=device,
        )
        _, loss, _ = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % config.eval_interval == 0 or step == config.steps:
            # D7 fix: seeded full-sweep validation
            val_loss = _evaluate_full_sweep(model, dataset, config, device, "validation")
            train_loss = _evaluate_full_sweep(model, dataset, config, device, "train")
            loss_history.append({
                "step": float(step),
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_perplexity": math.exp(val_loss),
            })

    elapsed = time.perf_counter() - start_time

    # Compute bits-per-character
    validation_tokens = dataset.validation.tolist()
    validation_characters = len(tokenizer.decode(validation_tokens))
    characters_per_token = validation_characters / len(validation_tokens)
    final = loss_history[-1]
    bits_per_character = final["validation_loss"] / math.log(2) / characters_per_token

    # Build metrics
    metrics = MetricsSchema(
        act=config.act,
        experiment=f"{tokenizer_kind}-seed{config.seed}",
        seed=config.seed,
        device=device,
        config={**config.to_dict(), "tokenizer_kind": tokenizer_kind},
        parameters=model.parameter_count(),
        trainable_parameters=model.parameter_count(trainable_only=True),
        total_tokens=len(dataset.train) + len(dataset.validation),
        train_tokens=len(dataset.train),
        validation_tokens=len(dataset.validation),
        vocabulary_size=tokenizer.vocabulary_size,
        loss_history=loss_history,
        wall_clock_seconds=round(elapsed, 4),
        steps_per_second=round(config.steps / elapsed, 4),
        final_step=float(final["step"]),
        final_train_loss=final["train_loss"],
        final_validation_loss=final["validation_loss"],
        final_validation_perplexity=final["validation_perplexity"],
    )

    # Write outputs
    result_dir = Path(config.output_root) / config.act / f"{tokenizer_kind}-seed{config.seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics.validate()
    (result_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (result_dir / "metrics.json").write_text(metrics.to_json(), encoding="utf-8")

    # Also save a sidecar with BPC and characters-per-token
    sidecar = {
        "tokenizer": tokenizer_kind,
        "characters_per_token": characters_per_token,
        "validation_bits_per_character": bits_per_character,
        "validation_perplexity_per_character": 2.0 ** bits_per_character,
        "merges": len(getattr(tokenizer, "merges", ())),
    }
    (result_dir / "tokenizer_stats.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print(f"[tokenizer-comparison] {tokenizer_kind} done. val_loss={final['validation_loss']:.4f} bpc={bits_per_character:.4f}")
    return metrics


def run_multi_seed_comparison(
    act: str,
    base_config: ExperimentConfig,
    seeds: list[int],
) -> dict[str, object]:
    """Run char-vs-BPE across multiple seeds and aggregate."""
    results = {"character": [], "bpe": []}
    for seed in seeds:
        for kind in ("character", "bpe"):
            config = ExperimentConfig.from_dict(base_config.to_dict())
            config.act = act
            config.seed = seed
            metrics = run_tokenizer_comparison(kind, config)
            results[kind].append(metrics)

    # Aggregate
    summary = {}
    for kind in ("character", "bpe"):
        bpcs = []
        for m in results[kind]:
            sidecar_path = Path(m.config["output_root"]) / act / f"{kind}-seed{m.seed}" / "tokenizer_stats.json"
            if sidecar_path.exists():
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                bpcs.append(sidecar["validation_bits_per_character"])
        if bpcs:
            summary[kind] = {
                "bits_per_character_mean": mean(bpcs),
                "bits_per_character_std": std(bpcs) if len(bpcs) > 1 else 0.0,
                "n_seeds": len(bpcs),
                "per_seed_bpc": bpcs,
            }

    return summary


def _lr_at_step(step: int, config: ExperimentConfig) -> float:
    """Linear warmup then cosine decay to 10% of peak LR."""
    if config.warmup_steps > 0 and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="data/tinyshakespeare.txt")
    parser.add_argument("--output-root", default="results/tokenizer-comparison")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--bpe-merges", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--normalization", default="layernorm")
    args = parser.parse_args()

    base_config = ExperimentConfig(
        act="tokenizer-comparison",
        experiment="main",
        seed=args.seeds[0],
        dataset_path=args.dataset_path,
        train_fraction=args.train_fraction,
        tokenizer="character",
        bpe_merges=args.bpe_merges,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        normalization=args.normalization,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        device=args.device,
        output_root=args.output_root,
    )

    summary = run_multi_seed_comparison("tokenizer-comparison", base_config, args.seeds)

    # Write summary
    summary_path = Path(args.output_root) / "tokenizer-comparison" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print table
    print("\n=== Tokenizer Comparison (D6/D7 fixed) ===")
    print(f"{'Tokenizer':<12} {'BPC mean':>10} {'BPC std':>10} {'n':>5}")
    print("-" * 40)
    for kind in ("character", "bpe"):
        if kind in summary:
            s = summary[kind]
            print(f"{kind:<12} {s['bits_per_character_mean']:>10.4f} {s['bits_per_character_std']:>10.4f} {s['n_seeds']:>5}")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
