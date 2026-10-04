"""Configurable CPU/Colab-ready training entry point for MiniGPT."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.data import TextDataset, load_text
from src.model import GPTConfig, MiniGPT
from src.tokenizer import CharacterTokenizer


@dataclass
class TrainConfig:
    dataset_path: str
    output_dir: str = "artifacts/baseline"
    steps: int = 500
    batch_size: int = 16
    context_length: int = 64
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    learning_rate: float = 3e-4
    eval_interval: int = 100
    eval_batches: int = 10
    seed: int = 42
    device: str = "auto"
    normalization: str = "layernorm"
    optimized_attention: bool = True


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


@torch.no_grad()
def evaluate_loss(model: MiniGPT, dataset: TextDataset, config: TrainConfig, *, split: str) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(config.eval_batches):
        inputs, targets = dataset.batch(split, batch_size=config.batch_size, context_length=config.context_length, device=next(model.parameters()).device)
        _, loss = model(inputs, targets)
        assert loss is not None
        losses.append(float(loss))
    model.train(was_training)
    return sum(losses) / len(losses)


def train(config: TrainConfig) -> tuple[MiniGPT, CharacterTokenizer, list[dict[str, float]]]:
    """Train a small causal LM and write config, metrics, and checkpoint artifacts."""
    if config.steps <= 0 or config.eval_interval <= 0:
        raise ValueError("steps and eval_interval must be positive")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    text = load_text(config.dataset_path)
    tokenizer = CharacterTokenizer.from_text(text)
    dataset = TextDataset.from_text(text, tokenizer)
    model = MiniGPT(GPTConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        normalization=config.normalization,
        optimized_attention=config.optimized_attention,
    )).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    metrics: list[dict[str, float]] = []
    for step in range(1, config.steps + 1):
        inputs, targets = dataset.batch("train", batch_size=config.batch_size, context_length=config.context_length, device=device)
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.eval_interval == 0 or step == config.steps:
            train_loss = evaluate_loss(model, dataset, config, split="train")
            validation_loss = evaluate_loss(model, dataset, config, split="validation")
            metrics.append({"step": float(step), "train_loss": train_loss, "validation_loss": validation_loss, "validation_perplexity": math.exp(validation_loss)})
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": model.config.to_dict(), "model_state": model.state_dict(), "vocabulary": tokenizer.vocabulary}, output / "checkpoint.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return model, tokenizer, metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional JSON or YAML configuration file")
    for name, field in TrainConfig.__dataclass_fields__.items():
        argument = "--" + name.replace("_", "-")
        field_type = TrainConfig.__annotations__[name]
        if field_type is bool:
            parser.add_argument(argument, dest=name, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(argument, dest=name, type=field_type, default=None)
    return parser


def _config_from_args() -> TrainConfig:
    values = vars(_parser().parse_args())
    config_path = values.pop("config")
    file_values: dict[str, object] = {}
    if config_path:
        content = Path(config_path).read_text(encoding="utf-8")
        if config_path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ModuleNotFoundError as error:
                raise RuntimeError("YAML configs require `pip install pyyaml`.") from error
            file_values = yaml.safe_load(content) or {}
        else:
            file_values = json.loads(content)
    file_values.update({key: value for key, value in values.items() if value is not None})
    if "dataset_path" not in file_values:
        raise ValueError("dataset_path must be supplied by --dataset-path or configuration file")
    return TrainConfig(**file_values)


if __name__ == "__main__":
    configuration = _config_from_args()
    _, _, history = train(configuration)
    print(json.dumps(history[-1], indent=2))
