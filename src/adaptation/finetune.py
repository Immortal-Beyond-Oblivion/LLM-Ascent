"""A small LoRA instruction-tuning loop for in-memory micro datasets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.model import MiniGPT

from .instructions import InstructionExample, collate_instruction_batch
from .lora import inject_lora


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    learning_rate: float = 1e-3
    steps: int = 100
    max_length: int = 128


def finetune_lora(model: MiniGPT, examples: list[InstructionExample], tokenizer: object, config: LoRAConfig = LoRAConfig()) -> list[float]:
    """Adapt attention projections and return the per-step supervised losses."""
    if not examples or config.steps <= 0:
        raise ValueError("examples must be non-empty and steps positive")
    inject_lora(model, rank=config.rank, alpha=config.alpha, dropout=config.dropout)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    device = next(model.parameters()).device
    inputs, labels = collate_instruction_batch(examples, tokenizer, max_length=min(config.max_length, model.config.context_length))
    inputs, labels = inputs.to(device), labels.to(device)
    losses: list[float] = []
    model.train()
    for _ in range(config.steps):
        _, loss = model(inputs, labels)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses
