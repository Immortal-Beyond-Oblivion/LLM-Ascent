"""Normalization layers used by the Phase 2 transformer."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned per-channel scale."""

    def __init__(self, dimension: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.epsilon).to(inputs.dtype)
        return normalized * self.weight
