"""GRU (Gated Recurrent Unit) language model.

The GRU cell has reset and update gates:
    r_t = σ(W_r @ [h_{t-1}, x_t] + b_r)    # reset gate
    z_t = σ(W_z @ [h_{t-1}, x_t] + b_z)    # update gate
    n_t = tanh(W_n @ [r_t * h_{t-1}, x_t] + b_n)  # candidate
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}  # hidden state update

The GRU is a simplified LSTM with 2 gates instead of 3, and no separate
cell state. It has fewer parameters than an LSTM at the same hidden size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GRUConfig:
    vocabulary_size: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0


class GRULanguageModel(nn.Module):
    """GRU language model.

    Architecture:
        Embedding → GRU → Linear → logits
    """

    def __init__(self, config: GRUConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary_size, config.embedding_dim)
        self.gru = nn.GRU(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_dim, config.vocabulary_size)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.embedding(token_ids)
        output, _ = self.gru(hidden)
        logits = self.output(output)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def init_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim, device=device)

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)
