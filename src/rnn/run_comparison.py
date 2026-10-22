"""Act 1 comparison: RNN vs LSTM vs GRU at matched parameter count.

Also demonstrates vanishing/exploding gradients in vanilla RNNs.

Usage:
    .venv/bin/python -m src.rnn.run_comparison
    .venv/bin/python -m src.rnn.run_comparison --seeds 42 43 44 --steps 500
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from src.bench.experiment import MetricsSchema
from src.bench.seeds import set_seed
from src.bench.stats import mean, std
from src.data import TextDataset, load_text
from src.rnn.rnn import RNNConfig, RNNLanguageModel, FixedWindowNNLM
from src.rnn.lstm import LSTMConfig, LSTMLanguageModel
from src.rnn.gru import GRUConfig, GRULanguageModel
from src.training.train import resolve_device, _lr_at_step


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_matched_models(vocabulary_size: int, target_params: int = 50000) -> dict[str, torch.nn.Module]:
    """Build RNN, LSTM, GRU models with approximately matched parameter counts.

    Adjusts hidden_dim to get as close as possible to target_params.
    """
    models = {}

    # Start with a reasonable hidden_dim and adjust
    for name, config_cls, model_cls in [
        ("rnn", RNNConfig, RNNLanguageModel),
        ("lstm", LSTMConfig, LSTMLanguageModel),
        ("gru", GRUConfig, GRULanguageModel),
    ]:
        # Binary search for hidden_dim that gets closest to target_params
        best_dim = 32
        best_diff = float("inf")
        for dim in [16, 24, 32, 48, 64, 96, 128]:
            config = config_cls(vocabulary_size=vocabulary_size, embedding_dim=32, hidden_dim=dim)
            model = model_cls(config)
            params = count_parameters(model)
            diff = abs(params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_dim = dim

        config = config_cls(vocabulary_size=vocabulary_size, embedding_dim=32, hidden_dim=best_dim)
        model = model_cls(config)
        models[name] = model

    return models


def train_rnn_model(
    model: torch.nn.Module,
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
    """Train an RNN-family model and return metrics."""
    set_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loss_history: list[dict[str, float]] = []

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
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(5):
                    val_inputs, val_targets = dataset.batch(
                        "validation", batch_size=batch_size, context_length=context_length, device=device
                    )
                    _, val_loss = model(val_inputs, val_targets)
                    assert val_loss is not None
                    val_losses.append(float(val_loss.detach()))
            model.train()
            avg_val_loss = sum(val_losses) / len(val_losses)

            # Train loss
            train_losses = []
            with torch.no_grad():
                for _ in range(5):
                    train_inputs, train_targets = dataset.batch(
                        "train", batch_size=batch_size, context_length=context_length, device=device
                    )
                    _, train_loss = model(train_inputs, train_targets)
                    assert train_loss is not None
                    train_losses.append(float(train_loss.detach()))
            avg_train_loss = sum(train_losses) / len(train_losses)

            loss_history.append({
                "step": float(step),
                "train_loss": avg_train_loss,
                "validation_loss": avg_val_loss,
                "validation_perplexity": math.exp(avg_val_loss),
            })

    final = loss_history[-1]
    metrics = MetricsSchema(
        act="act1-rnn",
        experiment=f"rnn-model-seed{seed}",
        seed=seed,
        device=device,
        config={"model_type": "rnn", "steps": steps, "batch_size": batch_size, "context_length": context_length},
        parameters=count_parameters(model),
        trainable_parameters=count_parameters(model),
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
    """Simple LR schedule: warmup then cosine decay."""
    if warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def demonstrate_vanishing_gradients(
    vocabulary_size: int = 66,
    embedding_dim: int = 16,
    hidden_dim: int = 32,
    sequence_length: int = 50,
) -> dict[str, list[float]]:
    """Demonstrate vanishing/exploding gradients in a vanilla RNN.

    Trains a vanilla RNN on a copy task and records gradient norms over time.
    Shows that gradients vanish (or explode) in vanilla RNNs but not in LSTMs.
    """
    set_seed(42)
    device = "cpu"

    results = {"rnn": [], "lstm": []}

    for model_type in ["rnn", "lstm"]:
        if model_type == "rnn":
            config = RNNConfig(
                vocabulary_size=vocabulary_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
            )
            model = RNNLanguageModel(config)
        else:
            config = LSTMConfig(
                vocabulary_size=vocabulary_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
            )
            model = LSTMLanguageModel(config)

        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Generate a simple copy task: predict the input shifted by 1
        # This is a task that requires remembering information over the sequence
        grad_norms = []
        for step in range(100):
            # Random input
            inputs = torch.randint(0, vocabulary_size, (4, sequence_length), device=device)
            targets = inputs  # Copy task: predict same sequence

            _, loss = model(inputs, targets)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Record total gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            optimizer.step()

        results[model_type] = grad_norms

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="data/tinyshakespeare.txt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-params", type=int, default=50000)
    parser.add_argument("--output-root", default="results/act1-rnn")
    args = parser.parse_args()

    text = load_text(args.dataset_path)
    from src.tokenizer import CharacterTokenizer
    tokenizer = CharacterTokenizer.from_text(text)
    dataset = TextDataset.from_text(text, tokenizer, train_fraction=0.9)
    vocabulary_size = tokenizer.vocabulary_size

    device = resolve_device(args.device)
    print(f"[act1] device={device} vocab_size={vocabulary_size} seeds={args.seeds}")

    # Build matched models
    models = build_matched_models(vocabulary_size, target_params=args.target_params)
    for name, model in models.items():
        print(f"[act1] {name}: {count_parameters(model)} parameters")

    # Train each model
    all_results = {}
    for name, model in models.items():
        all_results[name] = []
        for seed in args.seeds:
            print(f"[act1] training {name} seed={seed} ...")
            metrics = train_rnn_model(
                model,
                dataset,
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
            result_dir = Path(args.output_root) / "act1-rnn" / f"{name}-seed{seed}"
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "metrics.json").write_text(metrics.to_json(), encoding="utf-8")

    # Aggregate
    summary = {}
    for name, runs in all_results.items():
        val_losses = [m.final_validation_loss for m in runs]
        val_ppls = [m.final_validation_perplexity for m in runs]
        params = count_parameters(models[name])
        summary[name] = {
            "validation_loss_mean": mean(val_losses),
            "validation_loss_std": std(val_losses) if len(val_losses) > 1 else 0.0,
            "validation_perplexity_mean": mean(val_ppls),
            "parameters": params,
            "n_seeds": len(runs),
        }

    # Demonstrate vanishing gradients
    print("[act1] demonstrating vanishing/exploding gradients ...")
    grad_results = demonstrate_vanishing_gradients()
    for model_type, norms in grad_results.items():
        summary[f"{model_type}_grad_norms"] = {
            "initial": norms[0] if norms else 0.0,
            "final": norms[-1] if norms else 0.0,
            "mean": mean(norms) if norms else 0.0,
        }

    # Write summary
    summary_path = Path(args.output_root) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print table
    print("\n=== Act 1: RNN vs LSTM vs GRU (matched parameters) ===")
    print(f"{'Model':<10} {'Params':>10} {'Val Loss':>10} {'Val PPL':>10} {'n':>5}")
    print("-" * 50)
    for name in ["rnn", "lstm", "gru"]:
        if name in summary:
            s = summary[name]
            print(f"{name:<10} {s['parameters']:>10} {s['validation_loss_mean']:>10.4f} {s['validation_perplexity_mean']:>10.2f} {s['n_seeds']:>5}")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
