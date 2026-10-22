"""Act 2 ablation runner: Transformer component ablations on the A2 framework.

Runs normalization, positional encoding, FFN, and head-count ablations.
Emits uniform metrics.json under results/act2-ablations/.

Usage:
    .venv/bin/python -m src.model.ablation_runner
    .venv/bin/python -m src.model.ablation_runner --seeds 42 43 44
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
from src.model.ablations import AblationConfig, AblationTransformer, AblationBlock
from src.training.train import resolve_device


def train_ablation(
    model: AblationTransformer,
    dataset: TextDataset,
    *,
    steps: int = 500,
    batch_size: int = 16,
    context_length: int = 64,
    learning_rate: float = 3e-4,
    warmup_steps: int = 0,
    eval_interval: int = 100,
    seed: int = 42,
    device: str = "cpu",
) -> MetricsSchema:
    set_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loss_history: list[dict] = []

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
            # Eval
            model.eval()
            val_losses = []
            train_losses = []
            with torch.no_grad():
                for _ in range(5):
                    vi, vt = dataset.batch("validation", batch_size=batch_size, context_length=context_length, device=device)
                    _, vloss = model(vi, vt)
                    assert vloss is not None
                    val_losses.append(float(vloss.detach()))
                for _ in range(5):
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
    config_dict = model.config.__dict__ if hasattr(model, "config") else {}
    metrics = MetricsSchema(
        act="act2-ablations",
        experiment="ablation",
        seed=seed,
        device=device,
        config=config_dict,
        parameters=model.parameter_count(),
        trainable_parameters=model.parameter_count(),
        total_tokens=len(dataset.train) + len(dataset.validation),
        train_tokens=len(dataset.train),
        validation_tokens=len(dataset.validation),
        vocabulary_size=dataset.train.max().item() + 1,
        loss_history=loss_history,
        wall_clock_seconds=0.0,
        steps_per_second=0.0,
        final_step=float(final["step"]),
        final_train_loss=final["train_loss"],
        final_validation_loss=final["validation_loss"],
        final_validation_perplexity=final["validation_perplexity"],
    )
    metrics.validate()
    return metrics


def _lr_at_step_simple(step: int, peak_lr: float, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# ──────────────────────────────
# Ablation configurations to run
# ──────────────────────────────

# Normalization ablation: LayerNorm, RMSNorm, BatchNorm, none × pre vs post
NORM_ABLATIONS = []
for norm in ["layernorm", "rmsnorm", "batchnorm", "none"]:
    for pos in ["pre", "post"]:
        if norm == "none" and pos == "post":
            continue  # Skip meaningless combination
        NORM_ABLATIONS.append((norm, pos))

# Positional encoding ablation
PE_ABLATIONS = ["learned", "sinusoidal", "rope", "alibi", "none"]

# FFN activation ablation
FFN_ABLATIONS = ["gelu", "relu", "swiglu"]

# Head count ablation
HEAD_ABLATIONS = [1, 2, 4, 8]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="data/tinyshakespeare.txt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ablation", default="all", choices=["all", "norm", "pe", "ffn", "heads"])
    parser.add_argument("--output-root", default="results/act2-ablations")
    args = parser.parse_args()

    text = load_text(args.dataset_path)
    tokenizer = None
    # Use character tokenizer (simplest)
    from src.tokenizer import CharacterTokenizer
    tokenizer = CharacterTokenizer.from_text(text)
    dataset = TextDataset.from_text(text, tokenizer, train_fraction=0.9)
    vocabulary_size = tokenizer.vocabulary_size

    device = resolve_device(args.device)
    print(f"[act2] device={device} vocab={vocabulary_size} seeds={args.seeds}")

    # Base config
    base_config = {
        "vocabulary_size": vocabulary_size,
        "context_length": args.context_length,
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": 0.0,
        "normalization": "layernorm",
        "norm_position": "pre",
        "positional_encoding": "learned",
        "ffn_activation": "gelu",
    }

    # Collect ablation variants to run
    variants = []
    if args.ablation in ("all", "norm"):
        for norm, pos in NORM_ABLATIONS:
            variants.append(("norm", f"{norm}-{pos}", {**base_config, "normalization": norm, "norm_position": pos}))
    if args.ablation in ("all", "pe"):
        for pe in PE_ABLATIONS:
            variants.append(("pe", f"pe-{pe}", {**base_config, "positional_encoding": pe}))
    if args.ablation in ("all", "ffn"):
        for act in FFN_ABLATIONS:
            variants.append(("ffn", f"ffn-{act}", {**base_config, "ffn_activation": act}))
    if args.ablation in ("all", "heads") and args.num_heads >= 4:
        for h in HEAD_ABLATIONS:
            if h <= args.num_heads:
                variants.append(("heads", f"heads-{h}", {**base_config, "num_heads_ablation": h, "positional_encoding": "learned"}))

    # Run ablations
    all_results = {}
    for group, name, config in variants:
        all_results[name] = []
        for seed in args.seeds:
            print(f"[act2] {name} seed={seed} ...", flush=True)
            ablation_config = AblationConfig(**config)
            model = AblationTransformer(ablation_config)
            metrics = train_ablation(
                model, dataset,
                steps=args.steps,
                batch_size=args.batch_size,
                context_length=args.context_length,
                learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                eval_interval=args.eval_interval,
                seed=seed,
                device=device,
            )
            all_results[name].append(metrics)

            # Write metrics
            result_dir = Path(args.output_root) / "act2-ablations" / f"{name}-seed{seed}"
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "metrics.json").write_text(metrics.to_json(), encoding="utf-8")

    # Aggregate
    summary = {}
    for name, runs in all_results.items():
        val_losses = [m.final_validation_loss for m in runs]
        val_ppls = [m.final_validation_perplexity for m in runs]
        summary[name] = {
            "validation_loss_mean": mean(val_losses),
            "validation_loss_std": std(val_losses) if len(val_losses) > 1 else 0.0,
            "validation_perplexity_mean": mean(val_ppls),
            "parameters": runs[0].parameters,
            "n_seeds": len(runs),
        }

    summary_path = Path(args.output_root) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print table
    print("\n=== Act 2: Transformer Ablations ===")
    print(f"{'Ablation':<30} {'Params':>8} {'Val Loss':>10} {'Val PPL':>10} {'Std':>8} {'n':>3}")
    print("-" * 75)
    for name in sorted(summary.keys()):
        s = summary[name]
        print(f"{name:<30} {s['parameters']:>8} {s['validation_loss_mean']:>10.4f} {s['validation_perplexity_mean']:>10.2f} {s['validation_loss_std']:>8.4f} {s['n_seeds']:>3}")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
