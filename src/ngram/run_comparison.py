"""Act 0 comparison runner: n-gram LMs with different smoothing methods.

Runs unigram/bigram/trigram models with Laplace, Good-Turing, Kneser-Ney,
and interpolation on TinyShakespeare, emitting uniform metrics.json under
the A2 framework.

Usage:
    .venv/bin/python -m src.ngram.run_comparison
    .venv/bin/python -m src.ngram.run_comparison --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from src.bench.experiment import MetricsSchema
from src.bench.seeds import set_seed
from src.bench.stats import mean, std
from src.data import load_text
from src.ngram.ngram import NGramConfig, NGramModel


def split_text(text: str, train_fraction: float = 0.9) -> tuple[str, str]:
    """Split text into train and validation sets."""
    split_index = int(len(text) * train_fraction)
    return text[:split_index], text[split_index:]


def run_ngram_experiment(
    order: int,
    smoothing: str,
    train_text: str,
    validation_text: str,
    seed: int,
    act: str = "act0-ngram",
    discount: float = 0.75,
    interpolation_weights: tuple[float, float, float] = (0.1, 0.3, 0.6),
    output_root: str | Path = "results",
) -> MetricsSchema:
    """Run one n-gram configuration and return metrics.

    ``output_root`` is the directory results are written under. It is a
    parameter rather than a hard-coded ``results`` so that callers which are
    not the real experiment sweep — notably the test suite — can send their
    output somewhere disposable. ``results/`` holds tracked evidence, and a test
    that overwrites it silently invalidates that evidence with no diff in git
    status until the damage is already committed.
    """
    set_seed(seed)
    config = NGramConfig(
        order=order,
        smoothing=smoothing,
        discount=discount,
        interpolation_weights=interpolation_weights,
    )
    model = NGramModel(config)

    start = time.perf_counter()
    model.train(train_text)
    train_time = time.perf_counter() - start

    # Compute metrics on validation
    val_perplexity = model.perplexity(validation_text)
    val_bpc = model.bits_per_character(validation_text)

    # Also compute train metrics
    train_perplexity = model.perplexity(train_text[:10000])  # Sample for speed
    train_bpc = model.bits_per_character(train_text[:10000])

    # Generate samples
    greedy_sample = model.generate(200, seed=seed)

    experiment_name = f"o{order}-{smoothing}-seed{seed}"
    metrics = MetricsSchema(
        act=act,
        experiment=experiment_name,
        seed=seed,
        device="cpu",
        config={
            "order": order,
            "smoothing": smoothing,
            "discount": discount,
            "interpolation_weights": interpolation_weights,
            "train_chars": len(train_text),
            "validation_chars": len(validation_text),
            "vocabulary_size": model.vocabulary_size,
        },
        parameters=sum(model.unigram_counts.values()) + sum(model.bigram_counts.values()) + sum(model.trigram_counts.values()),
        trainable_parameters=sum(model.unigram_counts.values()) + sum(model.bigram_counts.values()) + sum(model.trigram_counts.values()),
        total_tokens=len(train_text) + len(validation_text),
        train_tokens=len(train_text),
        validation_tokens=len(validation_text),
        vocabulary_size=model.vocabulary_size,
        loss_history=[{
            "step": 1.0,
            "train_loss": -math.log(train_perplexity) / math.log(math.e),
            "validation_loss": -math.log(val_perplexity) / math.log(math.e),
            "validation_perplexity": val_perplexity,
        }],
        wall_clock_seconds=round(train_time, 4),
        steps_per_second=0.0,
        final_step=1.0,
        final_train_loss=-math.log(train_perplexity),
        final_validation_loss=-math.log(val_perplexity),
        final_validation_perplexity=val_perplexity,
    )

    # Write outputs
    result_dir = Path(output_root) / act / experiment_name
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics.validate()
    (result_dir / "metrics.json").write_text(metrics.to_json(), encoding="utf-8")

    # Save sample
    (result_dir / "sample.txt").write_text(greedy_sample, encoding="utf-8")

    print(f"[act0] {experiment_name}: val_ppl={val_perplexity:.2f} bpc={val_bpc:.4f} train_time={train_time:.2f}s")
    return metrics


def run_act0_comparison(
    seeds: list[int] | None = None,
    dataset_path: str = "data/tinyshakespeare.txt",
    train_fraction: float = 0.9,
) -> dict:
    """Run the full Act 0 comparison across orders, smoothings, and seeds."""
    if seeds is None:
        seeds = [42, 43, 44]

    text = load_text(dataset_path)
    train_text, validation_text = split_text(text, train_fraction)

    print(f"[act0] dataset={dataset_path} train={len(train_text)} chars validation={len(validation_text)} chars")
    print(f"[act0] seeds={seeds}")

    results = {}
    for order in [1, 2, 3]:
        for smoothing in ["laplace", "good_turing", "kneser_ney", "interpolated"]:
            key = f"o{order}-{smoothing}"
            results[key] = []
            for seed in seeds:
                metrics = run_ngram_experiment(
                    order=order,
                    smoothing=smoothing,
                    train_text=train_text,
                    validation_text=validation_text,
                    seed=seed,
                )
                results[key].append({
                    "seed": seed,
                    "val_perplexity": metrics.final_validation_perplexity,
                    "val_bpc": -metrics.final_validation_loss / math.log(2),
                    "train_time": metrics.wall_clock_seconds,
                })

    # Aggregate
    summary = {}
    for key, runs in results.items():
        bpcs = [r["val_bpc"] for r in runs]
        ppls = [r["val_perplexity"] for r in runs]
        summary[key] = {
            "bpc_mean": mean(bpcs),
            "bpc_std": std(bpcs) if len(bpcs) > 1 else 0.0,
            "perplexity_mean": mean(ppls),
            "perplexity_std": std(ppls) if len(ppls) > 1 else 0.0,
            "n_seeds": len(bpcs),
            "per_seed_bpc": bpcs,
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="data/tinyshakespeare.txt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--output-root", default="results/act0-ngram")
    args = parser.parse_args()

    summary = run_act0_comparison(
        seeds=args.seeds,
        dataset_path=args.dataset_path,
        train_fraction=args.train_fraction,
    )

    # Write summary
    output_path = Path(args.output_root) / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print table
    print("\n=== Act 0: N-gram LM Comparison ===")
    print(f"{'Model':<25} {'BPC mean':>10} {'BPC std':>10} {'PPL mean':>10} {'n':>5}")
    print("-" * 65)
    for key in sorted(summary.keys()):
        s = summary[key]
        print(f"{key:<25} {s['bpc_mean']:>10.4f} {s['bpc_std']:>10.4f} {s['perplexity_mean']:>10.2f} {s['n_seeds']:>5}")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
