"""Transformer component ablations: normalization, positional encoding, FFN.

Ablation variants for Act 2:
- Normalization: LayerNorm, RMSNorm, BatchNorm, pre-norm vs post-norm
- Positional encoding: learned absolute, sinusoidal, RoPE, ALiBi, none
- FFN activation: GELU, ReLU, SwiGLU
- Attention: causal mask toggle, head count sweep
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


# ──────────────────────────────
# Normalization variants
# ──────────────────────────────


class RMSNorm(nn.Module):
    """RMSNorm: re-center invariant normalization using only root mean square."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class BatchNorm1dCausal(nn.Module):
    """BatchNorm along the feature dimension for variable-length sequences.

    Uses running statistics. Not recommended for variable-length sequences
    but included as an ablation baseline to demonstrate why it's unsuitable.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] → [B, D, T] for BatchNorm1d → [B, T, D]
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


# ──────────────────────────────
# Positional encoding variants
# ──────────────────────────────


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding: sin(pos / 10000^(2i/d))."""

    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class RoPEPE(nn.Module):
    """Rotary Position Embedding (RoPE).

    Rotates the query/key vectors by angle proportional to position.
    Applied inside attention, but we simulate it by adding positional info
    to the input for comparability with other PE methods.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RoPE is typically applied to q/k, not x. For ablation comparability,
        # we apply it as an additive mask to the embedding.
        # This is a simplified version that captures the position-dependence.
        B, T, D = x.shape
        position = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(1)
        angle = position / (10000.0 ** (torch.arange(0, D, 2, device=x.device).float() / D))
        sin = torch.sin(angle).unsqueeze(0)
        cos = torch.cos(angle).unsqueeze(0)
        # Interleave sin/cos
        x_rot = x.clone()
        x_rot[:, :, 0::2] = x[:, :, 0::2] * cos - x[:, :, 1::2] * sin
        x_rot[:, :, 1::2] = x[:, :, 0::2] * sin + x[:, :, 1::2] * cos
        return x_rot


class ALiBiPE(nn.Module):
    """ALiBi: Attention with Linear Biases.

    No positional encoding added to embeddings. Instead, a linear bias is
    added to attention scores: bias = -m * (query_pos - key_pos) where m
    is a head-specific slope.
    For ablation comparability, we add the bias to the input.
    """

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        slopes = torch.pow(2, -torch.arange(1, heads + 1) * 8 / heads)
        self.register_buffer("slopes", slopes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ALiBi doesn't modify embeddings; the bias is applied in attention.
        # For ablation comparability, we leave embeddings unchanged.
        return x


class NoPE(nn.Module):
    """No positional encoding baseline (ablation)."""

    def __init__(self, dim: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ──────────────────────────────
# FFN activation variants
# ──────────────────────────────


class FFN(nn.Module):
    """Position-wise FFN with configurable activation."""

    def __init__(self, dim: int, activation: str = "gelu", expansion: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.activation = activation

        if activation == "swiglu":
            # SwiGLU: silu(W1x) * W3x → W2
            self.w1 = nn.Linear(dim, hidden, bias=False)
            self.w3 = nn.Linear(dim, hidden, bias=False)
            self.w2 = nn.Linear(hidden, dim, bias=False)
        else:
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        elif self.activation == "gelu":
            return self.fc2(F.gelu(self.fc1(x)))
        elif self.activation == "relu":
            return self.fc2(F.relu(self.fc1(x)))
        else:
            return self.fc2(F.gelu(self.fc1(x)))


# ──────────────────────────────
# MiniGPT with ablation config
# ──────────────────────────────


@dataclass(frozen=True)
class AblationConfig:
    """Configuration for a transformer ablation variant."""
    vocabulary_size: int
    context_length: int = 64
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.0
    # Ablation choices
    normalization: str = "layernorm"  # "layernorm", "rmsnorm", "batchnorm", "none"
    norm_position: str = "pre"  # "pre", "post"
    positional_encoding: str = "learned"  # "learned", "sinusoidal", "rope", "alibi", "none"
    ffn_activation: str = "gelu"  # "gelu", "relu", "swiglu"
    num_heads_ablation: int | None = None  # override num_heads for head ablation


class AblationTransformer(nn.Module):
    """MiniGPT with configurable ablation components."""

    def __init__(self, config: AblationConfig) -> None:
        super().__init__()
        self.config = config
        heads = config.num_heads_ablation or config.num_heads

        self.token_embedding = nn.Embedding(config.vocabulary_size, config.embedding_dim)
        self.pos_embedding = self._build_pe(config.positional_encoding, config.embedding_dim, config.context_length)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            block = AblationBlock(
                embedding_dim=config.embedding_dim,
                num_heads=heads,
                dropout=config.dropout,
                normalization=config.normalization,
                norm_position=config.norm_position,
                ffn_activation=config.ffn_activation,
            )
            self.blocks.append(block)

        self.final_norm = self._build_norm(config.normalization, config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocabulary_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def _build_pe(self, pe_type: str, dim: int, max_len: int) -> nn.Module | None:
        if pe_type == "learned":
            return nn.Embedding(max_len, dim)
        elif pe_type == "sinusoidal":
            return SinusoidalPE(dim, max_len)
        elif pe_type == "rope":
            return RoPEPE(dim)
        elif pe_type == "alibi":
            return ALiBiPE(dim, max_len)
        elif pe_type == "none":
            return None
        else:
            return nn.Embedding(max_len, dim)

    def _build_norm(self, norm_type: str, dim: int) -> nn.Module:
        if norm_type == "layernorm":
            return nn.LayerNorm(dim)
        elif norm_type == "rmsnorm":
            return RMSNorm(dim)
        elif norm_type == "batchnorm":
            return BatchNorm1dCausal(dim)
        elif norm_type == "none":
            return nn.Identity()
        else:
            return nn.LayerNorm(dim)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq = token_ids.shape
        positions = torch.arange(seq, device=token_ids.device)

        # Token embedding
        x = self.token_embedding(token_ids)

        # Positional encoding
        if isinstance(self.pos_embedding, nn.Embedding):
            x = x + self.pos_embedding(positions)
        elif self.pos_embedding is not None:
            x = self.pos_embedding(x)

        x = self.dropout(x)

        # ALiBi: apply attention bias (not implemented, skip)
        # Causal mask is handled inside attention

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)


class AblationBlock(nn.Module):
    """Transformer block with configurable normalization and FFN."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        normalization: str,
        norm_position: str,
        ffn_activation: str,
    ) -> None:
        super().__init__()
        self.norm_position = norm_position
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = self._build_norm(normalization, embedding_dim)
        self.norm2 = self._build_norm(normalization, embedding_dim)
        self.ffn = FFN(embedding_dim, activation=ffn_activation)
        self.dropout = nn.Dropout(dropout)

    def _build_norm(self, norm_type: str, dim: int) -> nn.Module:
        if norm_type == "layernorm":
            return nn.LayerNorm(dim)
        elif norm_type == "rmsnorm":
            return RMSNorm(dim)
        elif norm_type == "batchnorm":
            return BatchNorm1dCausal(dim)
        elif norm_type == "none":
            return nn.Identity()
        else:
            return nn.LayerNorm(dim)

    def _causal_mask(self, seq: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq, seq, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.size(1)
        causal_mask = self._causal_mask(seq, x.device)

        if self.norm_position == "pre":
            # Pre-norm
            normed = self.norm1(x)
            attn_out, _ = self.attention(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            # Post-norm
            attn_out, _ = self.attention(x, x, x, attn_mask=causal_mask, need_weights=False)
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.dropout(self.ffn(x)))

        return x
