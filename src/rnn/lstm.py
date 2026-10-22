"""LSTM language model with forget/input/output gates.

The LSTM cell maintains a cell state c_t and hidden state h_t:
    f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)    # forget gate
    i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)    # input gate
    g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g) # candidate
    c_t = f_t * c_{t-1} + i_t * g_t       # cell state update
    o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)    # output gate
    h_t = o_t * tanh(c_t)                  # hidden state

The gating mechanism mitigates vanishing gradients by allowing the network
to learn when to forget and when to update the cell state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LSTMConfig:
    vocabulary_size: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0


class LSTMLanguageModel(nn.Module):
    """LSTM language model.

    Architecture:
        Embedding → LSTM → Linear → logits
    """

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary_size, config.embedding_dim)
        self.lstm = nn.LSTM(
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
        output, _ = self.lstm(hidden)
        logits = self.output(output)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def init_hidden(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states to zeros."""
        h = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim, device=device)
        c = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim, device=device)
        return h, c

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)
