"""Unified experiment framework: one config schema, one metrics.json shape.

Every act declares a config and emits a uniform metrics.json so results/
aggregates into the report automatically.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.data import TextDataset, load_text
from src.model import GPTConfig, MiniGPT
from src.tokenizer import BPETokenizer, CharacterTokenizer
from src.training.train import TrainConfig, _build_tokenizer, resolve_device


@dataclass
class ExperimentConfig:
    """Unified config schema for all acts."""

    # Identity
    act: str
    experiment: str
    seed: int = 42

    # Data
    dataset_path: str = "data/tinyshakespeare.txt"
    train_fraction: float = 0.9

    # Tokenizer
    tokenizer: str = "character"  # "character" | "bpe"
    bpe_merges: int = 100

    # Model
    context_length: int = 64
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.0
    normalization: str = "layernorm"
    optimized_attention: bool = True

    # Training
    steps: int = 500
    batch_size: int = 16
    learning_rate: float = 3e-4
    warmup_steps: int = 0
    eval_interval: int = 100
    eval_batches: int = 10
    device: str = "auto"

    # Output
    output_root: str = "results"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(**data)

    def to_train_config(self, output_dir: str) -> TrainConfig:
        return TrainConfig(
            dataset_path=self.dataset_path,
            output_dir=output_dir,
            steps=self.steps,
            batch_size=self.batch_size,
            context_length=self.context_length,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            dropout=self.dropout,
            eval_interval=self.eval_interval,
            eval_batches=self.eval_batches,
            seed=self.seed,
            device=self.device,
            normalization=self.normalization,
            optimized_attention=self.optimized_attention,
            tokenizer=self.tokenizer,
            bpe_merges=self.bpe_merges,
        )


@dataclass
class LossRecord:
    step: int
    train_loss: float
    validation_loss: float
    validation_perplexity: float

    def to_dict(self) -> dict[str, float]:
        return {
            "step": float(self.step),
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "validation_perplexity": self.validation_perplexity,
        }


@dataclass
class MetricsSchema:
    """The uniform metrics.json shape. Every act emits this."""

    # Identity
    act: str
    experiment: str
    seed: int
    device: str
    config: dict[str, Any]

    # Model
    parameters: int
    trainable_parameters: int

    # Data
    total_tokens: int
    train_tokens: int
    validation_tokens: int
    vocabulary_size: int

    # Training
    loss_history: list[dict[str, float]] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    steps_per_second: float = 0.0

    # Final metrics (convenience — mirrors last loss_history entry)
    final_step: float = 0.0
    final_train_loss: float = 0.0
    final_validation_loss: float = 0.0
    final_validation_perplexity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "MetricsSchema":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricsSchema":
        return cls(**data)

    def validate(self) -> None:
        """Assert the schema is self-consistent."""
        if not self.act:
            raise ValueError("act must be non-empty")
        if not self.experiment:
            raise ValueError("experiment must be non-empty")
        if self.parameters <= 0:
            raise ValueError("parameters must be positive")
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if self.validation_tokens <= 0:
            raise ValueError("validation_tokens must be positive")
        if not self.loss_history:
            raise ValueError("loss_history must be non-empty")
        for record in self.loss_history:
            if "step" not in record or "train_loss" not in record or "validation_loss" not in record:
                raise ValueError(f"loss_history record missing required keys: {record}")
        # Final metrics must match last history entry
        if self.loss_history:
            last = self.loss_history[-1]
            if self.final_step != last["step"]:
                raise ValueError(f"final_step {self.final_step} != last history step {last['step']}")
            if abs(self.final_train_loss - last["train_loss"]) > 1e-9:
                raise ValueError(f"final_train_loss {final_train_loss} != last history train_loss {last['train_loss']}")
            if abs(self.final_validation_loss - last["validation_loss"]) > 1e-9:
                raise ValueError(f"final_validation_loss {self.final_validation_loss} != last history validation_loss {last['validation_loss']}")


def run_experiment(config: ExperimentConfig) -> MetricsSchema:
    """Run a training experiment and emit uniform metrics.json.

    Writes results/<act>/<experiment>/{config.json, metrics.json}.
    """
    result_dir = Path(config.output_root) / config.act / config.experiment
    result_dir.mkdir(parents=True, exist_ok=True)

    # Seed all RNGs for reproducibility
    from src.bench.seeds import set_seed
    set_seed(config.seed)

    # Resolve device
    device = resolve_device(config.device)
    print(f"[experiment] act={config.act} experiment={config.experiment} device={device}")

    # Load data and build tokenizer
    text = load_text(config.dataset_path)
    tokenizer = _build_tokenizer_from_config(text, config)
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
    loss_history: list[LossRecord] = []
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
            train_loss = _evaluate(model, dataset, config, "train", device)
            val_loss = _evaluate(model, dataset, config, "validation", device)
            loss_history.append(LossRecord(
                step=step,
                train_loss=train_loss,
                validation_loss=val_loss,
                validation_perplexity=math.exp(val_loss),
            ))

    elapsed = time.perf_counter() - start_time

    # Build metrics
    final = loss_history[-1]
    metrics = MetricsSchema(
        act=config.act,
        experiment=config.experiment,
        seed=config.seed,
        device=device,
        config=config.to_dict(),
        parameters=model.parameter_count(),
        trainable_parameters=model.parameter_count(trainable_only=True),
        total_tokens=len(dataset.train) + len(dataset.validation),
        train_tokens=len(dataset.train),
        validation_tokens=len(dataset.validation),
        vocabulary_size=tokenizer.vocabulary_size,
        loss_history=[r.to_dict() for r in loss_history],
        wall_clock_seconds=round(elapsed, 4),
        steps_per_second=round(config.steps / elapsed, 4),
        final_step=float(final.step),
        final_train_loss=final.train_loss,
        final_validation_loss=final.validation_loss,
        final_validation_perplexity=final.validation_perplexity,
    )

    # Write outputs
    metrics.validate()
    (result_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (result_dir / "metrics.json").write_text(metrics.to_json(), encoding="utf-8")

    print(f"[experiment] done. validation_loss={final.validation_loss:.4f} steps/s={metrics.steps_per_second:.2f}")
    return metrics


def load_metrics(path: str | Path) -> MetricsSchema:
    """Load and validate a metrics.json file."""
    text = Path(path).read_text(encoding="utf-8")
    metrics = MetricsSchema.from_json(text)
    metrics.validate()
    return metrics


def plot_metrics(metrics: MetricsSchema, output_path: str | Path) -> None:
    """Plot a loss curve from a metrics file and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in metrics.loss_history]
    train_losses = [r["train_loss"] for r in metrics.loss_history]
    val_losses = [r["validation_loss"] for r in metrics.loss_history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_losses, label="train", alpha=0.8)
    ax.plot(steps, val_losses, label="validation", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title(f"{metrics.act}/{metrics.experiment} (seed={metrics.seed}, device={metrics.device})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_tokenizer_from_config(
    text: str, config: ExperimentConfig
) -> CharacterTokenizer | BPETokenizer:
    if config.tokenizer == "character":
        return CharacterTokenizer.from_text(text)
    if config.tokenizer == "bpe":
        return BPETokenizer.train(text, max_merges=config.bpe_merges)
    raise ValueError(f"unknown tokenizer: {config.tokenizer}")


def _evaluate(
    model: MiniGPT,
    dataset: TextDataset,
    config: ExperimentConfig,
    split: str,
    device: str,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(config.eval_batches):
        inputs, targets = dataset.batch(
            split,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=device,
        )
        _, loss, _ = model(inputs, targets)
        assert loss is not None
        losses.append(float(loss.detach()))
    model.train(was_training)
    return sum(losses) / len(losses)


def _lr_at_step(step: int, config: ExperimentConfig) -> float:
    """Linear warmup then cosine decay to 10% of peak LR."""
    if config.warmup_steps > 0 and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
