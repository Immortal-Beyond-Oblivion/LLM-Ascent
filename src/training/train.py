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
from src.tokenizer import BPETokenizer, CharacterTokenizer


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
    tokenizer: str = "character"
    bpe_merges: int = 100
    resume_from: str | None = None


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


def _build_tokenizer(text: str, config: TrainConfig) -> CharacterTokenizer | BPETokenizer:
    if config.tokenizer == "character":
        return CharacterTokenizer.from_text(text)
    if config.tokenizer == "bpe":
        return BPETokenizer.train(text, max_merges=config.bpe_merges)
    raise ValueError("tokenizer must be 'character' or 'bpe'")


def _tokenizer_payload(tokenizer: CharacterTokenizer | BPETokenizer) -> dict[str, object]:
    if isinstance(tokenizer, CharacterTokenizer):
        return {"type": "character", "vocabulary": list(tokenizer.vocabulary)}
    return {
        "type": "bpe",
        "vocabulary": list(tokenizer.vocabulary),
        "merges": [list(pair) for pair in tokenizer.merges],
    }


def train(config: TrainConfig) -> tuple[MiniGPT, CharacterTokenizer | BPETokenizer, list[dict[str, float]]]:
    """Train a small causal LM and write config, metrics, and checkpoint artifacts."""
    if config.steps <= 0 or config.eval_interval <= 0:
        raise ValueError("steps and eval_interval must be positive")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    text = load_text(config.dataset_path)
    tokenizer = _build_tokenizer(text, config)
    dataset = TextDataset.from_text(text, tokenizer)
    model_config = GPTConfig(
        vocabulary_size=tokenizer.vocabulary_size, context_length=config.context_length,
        embedding_dim=config.embedding_dim, num_heads=config.num_heads, num_layers=config.num_layers,
        normalization=config.normalization, optimized_attention=config.optimized_attention,
    )
    model = MiniGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    metrics: list[dict[str, float]] = []
    start_step = 0
    if config.resume_from:
        checkpoint = torch.load(config.resume_from, map_location=device, weights_only=False)
        if checkpoint["model_config"] != model_config.to_dict():
            raise ValueError("resume checkpoint model configuration does not match the requested configuration")
        if checkpoint.get("tokenizer") != _tokenizer_payload(tokenizer):
            raise ValueError("resume checkpoint tokenizer does not match the requested configuration")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["completed_steps"])
        if config.steps <= start_step:
            raise ValueError("steps must exceed the checkpoint's completed_steps")
    for step in range(start_step + 1, config.steps + 1):
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
    torch.save({
        "model_config": model.config.to_dict(), "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "completed_steps": config.steps,
        "tokenizer": _tokenizer_payload(tokenizer),
        # Retained for backward compatibility with the Phase 3 smoke CLIs.
        "vocabulary": tokenizer.vocabulary,
    }, output / "checkpoint.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return model, tokenizer, metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional JSON or YAML configuration file")
    for name, field in TrainConfig.__dataclass_fields__.items():
        argument = "--" + name.replace("_", "-")
        if isinstance(field.default, bool):
            parser.add_argument(argument, dest=name, action=argparse.BooleanOptionalAction, default=None)
        else:
            argument_type = str if name == "dataset_path" or field.default is None else type(field.default)
            parser.add_argument(argument, dest=name, type=argument_type, default=None)
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
