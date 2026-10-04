"""A compact decoder-only Transformer for local language-model experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .normalization import RMSNorm


@dataclass(frozen=True)
class GPTConfig:
    vocabulary_size: int
    context_length: int = 128
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    normalization: str = "layernorm"
    optimized_attention: bool = True

    def __post_init__(self) -> None:
        if self.vocabulary_size <= 0 or self.context_length <= 0:
            raise ValueError("vocabulary_size and context_length must be positive")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must divide evenly by num_heads")
        if self.normalization not in {"layernorm", "rmsnorm"}:
            raise ValueError("normalization must be 'layernorm' or 'rmsnorm'")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _norm(config: GPTConfig) -> nn.Module:
    if config.normalization == "rmsnorm":
        return RMSNorm(config.embedding_dim)
    return nn.LayerNorm(config.embedding_dim)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.optimized_attention = config.optimized_attention
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels = inputs.shape
        qkv = self.qkv(inputs).view(batch, sequence, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        if self.optimized_attention and hasattr(F, "scaled_dot_product_attention"):
            output = F.scaled_dot_product_attention(
                query, key, value, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True
            )
        else:
            scores = (query @ key.transpose(-2, -1)) * (self.head_dim**-0.5)
            causal = torch.ones(sequence, sequence, dtype=torch.bool, device=inputs.device).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
            output = self.dropout(F.softmax(scores, dim=-1)) @ value
        output = output.transpose(1, 2).contiguous().view(batch, sequence, channels)
        return self.dropout(self.projection(output))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm_attention = _norm(config)
        self.attention = CausalSelfAttention(config)
        self.norm_mlp = _norm(config)
        self.mlp = MLP(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.norm_attention(inputs))
        return inputs + self.mlp(self.norm_mlp(inputs))


class MiniGPT(nn.Module):
    """Decoder-only model returning logits ``[B, T, V]`` and optional loss."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = _norm(config)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocabulary_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a torch.long tensor shaped [batch, time]")
        _, sequence = token_ids.shape
        if sequence > self.config.context_length:
            raise ValueError("sequence length exceeds configured context_length")
        positions = torch.arange(sequence, device=token_ids.device)
        hidden = self.dropout(self.token_embedding(token_ids) + self.position_embedding(positions))
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must match token_ids shape")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if not trainable_only or parameter.requires_grad)
