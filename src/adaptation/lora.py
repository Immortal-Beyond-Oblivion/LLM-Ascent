"""Minimal LoRA adapters for the project's linear transformer modules."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen linear projection plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, *, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank, self.scaling = rank, alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b) * self.scaling
        return self.base(inputs) + update


def inject_lora(model: nn.Module, *, target_suffixes: tuple[str, ...] = ("qkv", "projection"), rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> list[str]:
    """Freeze a model and replace named linear projections with LoRA wrappers."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    replaced: list[str] = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            qualified = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and child_name in target_suffixes:
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced.append(qualified)
    if not replaced:
        raise ValueError("no target linear layers were found for LoRA")
    return replaced
