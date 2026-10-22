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
    num_kv_heads: int = 0
    window_size: int = 0
    dropout: float = 0.1
    normalization: str = "layernorm"
    optimized_attention: bool = True

    def __post_init__(self) -> None:
        if self.vocabulary_size <= 0 or self.context_length <= 0:
            raise ValueError("vocabulary_size and context_length must be positive")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must divide evenly by num_heads")
        if self.num_kv_heads == 0:
            self.__dict__["num_kv_heads"] = self.num_heads
        if self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads:
            raise ValueError("num_kv_heads must divide num_heads")
        if self.window_size < 0:
            raise ValueError("window_size must be non-negative")
        if self.normalization not in {"layernorm", "rmsnorm"}:
            raise ValueError("normalization must be 'layernorm' or 'rmsnorm'")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _norm(config: GPTConfig) -> nn.Module:
    if config.normalization == "rmsnorm":
        return RMSNorm(config.embedding_dim)
    return nn.LayerNorm(config.embedding_dim)


class KVCache:
    """Incremental KV cache for fast autoregressive decoding."""

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, num_kv_heads: int | None = None) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        if layer_idx < len(self.keys):
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], key], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], value], dim=2)
        else:
            self.keys.append(key)
            self.values.append(value)

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Cache stores the raw num_kv_heads heads; expansion for GQA happens in
        # CausalSelfAttention.forward so the cache stays compact.
        return self.keys[layer_idx], self.values[layer_idx]

    @property
    def sequence_length(self) -> int:
        return self.keys[0].size(2) if self.keys else 0

    def truncate(self, length: int) -> None:
        """Roll the cache back to `length` positions (for speculative rollback)."""
        if self.keys:
            self.keys = [k[:, :, :length, :] for k in self.keys]
            self.values = [v[:, :, :length, :] for v in self.values]

    def clone(self) -> "KVCache":
        """Deep copy, for beam search where each beam needs its own cache.

        A shallow copy would share the per-layer tensors, so one beam appending
        a token would corrupt every other beam's history.
        """
        new = KVCache(self.num_layers, self.num_heads, self.head_dim, self.num_kv_heads)
        new.keys = [k.clone() for k in self.keys]
        new.values = [v.clone() for v in self.values]
        return new


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.window_size = config.window_size
        self.optimized_attention = config.optimized_attention
        # Act 9 hooks. `ablate_heads` is the set of head indices whose
        # attention output is zeroed (a causal intervention); `record_attention`
        # stashes the attention matrix for rollout and entropy measurements.
        # Both default to inert, so no existing behaviour changes.
        self.ablate_heads: set[int] = set()
        self.record_attention: bool = False
        self.last_attn: torch.Tensor | None = None
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.q = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.kv = nn.Linear(config.embedding_dim, 2 * self.kv_dim)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, inputs: torch.Tensor, cache: KVCache | None = None, layer_idx: int = 0
    ) -> torch.Tensor:
        batch, sequence, channels = inputs.shape
        q = self.q(inputs).view(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(inputs).view(batch, sequence, 2, self.num_kv_heads, self.head_dim)
        new_k, new_v = kv.unbind(dim=2)
        new_k = new_k.transpose(1, 2)
        new_v = new_v.transpose(1, 2)

        if cache is not None:
            cache.append(layer_idx, new_k, new_v)
            k, v = cache.get(layer_idx)
        else:
            k, v = new_k, new_v

        # Prefix tuning injects learned key/value pairs ahead of the real ones.
        # Attaching them to the attention module keeps this contained: the
        # injected pairs are part of the causal prefix, so no mask change is
        # needed (they precede every real position and are visible to all).
        prefix = getattr(self, "_prefix_kv", None)
        if prefix is not None:
            pk, pv = prefix
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)

        # GQA: expand shared KV heads to all query heads
        if self.num_kv_heads < self.num_heads:
            gsz = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(gsz, dim=1)
            v = v.repeat_interleave(gsz, dim=1)

        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        tq = q.size(2); tk = k.size(2)
        if cache is None or sequence > 1 or sequence == tk:
            # Real query i must attend to the injected prefix AND real positions
            # 0..i. With no prefix, tk - tq == 0 and this reduces to a plain
            # lower-triangular mask, which is the original behaviour.
            offset = tk - tq
            causal = torch.ones(tq, tk, dtype=torch.bool, device=inputs.device).tril(
                diagonal=offset)
            scores = scores.masked_fill(~causal, float("-inf"))

        if self.window_size > 0:
            # positions relative to full sequence
            pos_q = (torch.arange(tq, device=inputs.device) + (tk - tq)).unsqueeze(-1)
            pos_k = torch.arange(tk, device=inputs.device).unsqueeze(0)
            window_mask = (pos_q - pos_k).abs() <= self.window_size
            scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = attn @ v
        # Head ablation (Act 9). Zeroing a head's slice of the per-head
        # attention output, before it is mixed back into the residual stream,
        # is the causal intervention: the head still computes, but its
        # contribution to the block's output is removed.
        if self.ablate_heads:
            for head in self.ablate_heads:
                output[:, head] = 0.0
        if self.record_attention:
            self.last_attn = attn.detach()
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

    def forward(
        self, inputs: torch.Tensor, cache: KVCache | None = None, layer_idx: int = 0
    ) -> torch.Tensor:
        inputs = inputs + self.attention(self.norm_attention(inputs), cache=cache, layer_idx=layer_idx)
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
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, KVCache | None]:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a torch.long tensor shaped [batch, time]")
        _, sequence = token_ids.shape
        if sequence > self.config.context_length:
            raise ValueError("sequence length exceeds configured context_length")
        if cache is not None and sequence == 1:
            # Incremental decode: the new token is at absolute position = cache length
            positions = torch.full((1,), cache.sequence_length, device=token_ids.device)
        else:
            positions = torch.arange(sequence, device=token_ids.device)
        hidden = self.dropout(self.token_embedding(token_ids) + self.position_embedding(positions))
        for layer_idx, block in enumerate(self.blocks):
            hidden = block(hidden, cache=cache, layer_idx=layer_idx)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must match token_ids shape")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss, cache

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if not trainable_only or parameter.requires_grad)
    def non_embedding_params(self) -> int:
        """Non-embedding parameter count (Chinchilla convention), excluding embedding matrix."""
        return self.parameter_count() - self.token_embedding.weight.numel()

    def total_params(self) -> int:
        return self.parameter_count()
