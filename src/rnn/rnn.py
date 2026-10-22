"""Vanilla RNN language model with BPTT.

The vanilla RNN has a single hidden state update:
    h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)

This is the simplest recurrent architecture and suffers from vanishing/exploding
gradients over long sequences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RNNConfig:
    vocabulary_size: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0


class RNNLanguageModel(nn.Module):
    """Vanilla RNN language model.

    Architecture:
        Embedding → RNN → Linear → logits
    """

    def __init__(self, config: RNNConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary_size, config.embedding_dim)
        self.rnn = nn.RNN(
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
        output, _ = self.rnn(hidden)
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


class FixedWindowNNLM(nn.Module):
    """Fixed-window neural language model (baseline).

    Uses a fixed context window of `window_size` tokens to predict the next token.
    This is the non-recurrent baseline that RNNs should beat at matched parameters.
    """

    def __init__(self, vocabulary_size: int, window_size: int = 3, embedding_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.window_size = window_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.fc1 = nn.Linear(window_size * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocabulary_size)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq = token_ids.shape
        if seq < self.window_size:
            raise ValueError(f"sequence length {seq} < window size {self.window_size}")

        # For each position i >= window_size, predict token[i] from token[i-window_size:i]
        inputs = []
        for i in range(self.window_size, seq):
            window = token_ids[:, i - self.window_size : i]  # [B, window_size]
            embedded = self.embedding(window)  # [B, window_size, E]
            flat = embedded.reshape(batch, -1)  # [B, window_size * E]
            hidden = torch.tanh(self.fc1(flat))
            inputs.append(hidden)

        # Stack predictions for positions window_size..seq-1
        # We need to align with targets[window_size-1..seq-2] (shifted by 1)
        # Actually, for position i, target is token[i+1]
        # So we predict positions window_size..seq-1, targets are window_size..seq-1 shifted
        logits = self.fc2(torch.stack(inputs, dim=1))  # [B, seq-window_size, V]

        loss = None
        if targets is not None:
            # Targets for positions window_size..seq-1 are token[window_size+1..seq]
            # But targets[window_size-1..seq-2] = token[window_size..seq-1]
            # We predict token[i] from token[i-window_size:i], so target is token[i]
            # i ranges from window_size to seq-1
            # targets[:, window_size-1:seq-1] = token[window_size:seq]... no
            # Actually: input token[i-window_size:i] predicts token[i]
            # targets[:, i] = token[i+1]
            # So for input ending at i-1, target is token[i] = targets[:, i-1]
            # i ranges window_size..seq-1, so target index is i-1 = window_size-1..seq-2
            aligned_targets = targets[:, self.window_size - 1 : -1]  # [B, seq-window_size]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), aligned_targets.reshape(-1)
            )
        return logits, loss

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)
