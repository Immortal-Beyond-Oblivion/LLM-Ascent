"""PEFT family for Act 7, sized so the comparison is controlled.

The point of a PEFT head-to-head is the **matched trainable-parameter budget**.
Without it, "LoRA r=8 beats prompt tuning" can just mean "LoRA r=8 has more
trainable parameters". So every method here exposes a knob that moves its
trainable count, and ``size_to_budget`` picks the configuration closest to a
target budget for each method — that is what makes the comparison mean anything.

Methods:
  * ``lora``          — existing ``LoRALinear``, rank r
  * ``prompt_tuning`` — a learned prefix of k virtual token embeddings
  * ``prefix_tuning`` — k learned key/value pairs injected per layer
  * ``adapters``      — bottleneck MLP after each attention block
  * ``full``          — every parameter

All are implemented as thin wrappers that freeze the base model and expose
``parameters()``, so the SFT loop is identical across methods and only the
trainable set differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .lora import inject_lora

# MiniGPT keeps SEPARATE q and kv projections, so the v1 target list
# ("qkv", "projection") matched only "projection" -- LoRA was silently
# adapting one matrix instead of three.
LORA_TARGETS = ("q", "kv", "projection")

# Prompt tuning prepends virtual tokens to the sequence, so its size is bounded
# by the context window, not by the parameter budget alone.
MAX_PROMPT_TOKENS = 64


@dataclass(frozen=True)
class PEFTSpec:
    method: str            # lora | prompt_tuning | prefix_tuning | adapters | full
    size: int              # rank, prefix length, or adapter bottleneck
    target_suffixes: tuple[str, ...] = LORA_TARGETS

    def label(self) -> str:
        return f"{self.method}{self.size}" if self.method != "full" else "full"


def _freeze(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


# ── prompt tuning ───────────────────────────────────────────────────────────

class PromptTuning(nn.Module):
    """Learned virtual tokens prepended to the embedding sequence."""

    def __init__(self, model, n_virtual: int) -> None:
        super().__init__()
        _freeze(model)
        self.model = model
        self.n_virtual = n_virtual
        emb = model.token_embedding.weight.shape[1]
        self.prompt = nn.Parameter(
            torch.randn(n_virtual, emb, device=model.token_embedding.weight.device) * 0.02
        )

    def parameters(self) -> list[nn.Parameter]:
        return [self.prompt]

    def n_trainable(self) -> int:
        return self.prompt.numel()

    def forward(self, token_ids, targets=None, cache=None):
        """Inject the prompt after the embedding lookup.

        Only the uncached path is supported, which is all SFT needs. Attention
        already masks causally over whatever sequence it is handed, so
        prepending positions simply shifts the indices consistently.
        """
        base_emb = self.model.token_embedding(token_ids)
        prefix = self.prompt.unsqueeze(0).expand(base_emb.size(0), -1, -1)
        hidden = torch.cat([prefix, base_emb], dim=1)
        hidden = hidden + self.model.position_embedding(
            torch.arange(hidden.size(1), device=hidden.device)
        )
        for layer_idx, block in enumerate(self.model.blocks):
            hidden = block(hidden, cache=cache, layer_idx=layer_idx)
        logits = self.model.lm_head(self.model.final_norm(hidden))
        loss = None
        if targets is not None:
            shifted = torch.cat(
                [torch.full((targets.size(0), self.n_virtual), -100, device=targets.device,
                            dtype=targets.dtype), targets], dim=1)
            # `targets` is ALREADY aligned to the input sequence (the SFT loop
            # passes labels shifted by one, and MiniGPT's own loss applies no
            # further shift). The only adjustment needed is for the virtual
            # prefix, which has no target of its own.
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                shifted.reshape(-1), ignore_index=-100)
        return logits, loss, cache


# ── prefix tuning ───────────────────────────────────────────────────────────

class PrefixTuning(nn.Module):
    """Learned key/value pairs injected at every layer, concatenated with the
    real keys and values. Distinct from prompt tuning: this adds to the KV
    cache directly rather than to the input embeddings."""

    def __init__(self, model, n_virtual: int) -> None:
        super().__init__()
        _freeze(model)
        self.model = model
        self.n_virtual = n_virtual
        cfg = model.config
        self.kv_dim = cfg.num_kv_heads * (cfg.embedding_dim // cfg.num_heads)
        dev = model.token_embedding.weight.device
        head = cfg.embedding_dim // cfg.num_heads
        # Shape must be (batch, num_kv_heads, n_virtual, head_dim) to concatenate
        # with the real keys/values inside attention.
        self.prefix_k = nn.ParameterList(
            [nn.Parameter(torch.randn(1, cfg.num_kv_heads, n_virtual, head, device=dev) * 0.02)
             for _ in range(cfg.num_layers)])
        self.prefix_v = nn.ParameterList(
            [nn.Parameter(torch.randn(1, cfg.num_kv_heads, n_virtual, head, device=dev) * 0.02)
             for _ in range(cfg.num_layers)])

    def parameters(self) -> list[nn.Parameter]:
        return list(self.prefix_k) + list(self.prefix_v)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, token_ids, targets=None, cache=None):
        """Attach the learned K/V to each attention module, run, then detach.

        The causal mask shift that makes injected prefix pairs visible to real
        queries lives in ``CausalSelfAttention.forward`` (a ``tril`` diagonal
        offset), so this stays a thin wrapper.
        """
        blocks = self.model.blocks
        batch = token_ids.size(0)
        for i, block in enumerate(blocks):
            # Expand the single shared prefix across the batch.
            block.attention._prefix_kv = (self.prefix_k[i].expand(batch, -1, -1, -1),
                                          self.prefix_v[i].expand(batch, -1, -1, -1))
        try:
            emb = self.model.token_embedding(token_ids)
            hidden = emb + self.model.position_embedding(
                torch.arange(token_ids.size(1), device=token_ids.device))
            for layer_idx, block in enumerate(blocks):
                hidden = hidden + block.attention(block.norm_attention(hidden),
                                                  cache=cache, layer_idx=layer_idx)
                hidden = hidden + block.mlp(block.norm_mlp(hidden))
            logits = self.model.lm_head(self.model.final_norm(hidden))
            loss = None
            if targets is not None:
                # No shift: `targets` is already aligned to the input, exactly
                # as MiniGPT's own loss expects. Shifting a second time taught
                # these wrappers to predict the wrong position, which is why
                # they scored exactly 0.000 while actually training.
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1), ignore_index=-100)
            return logits, loss, cache
        finally:
            for block in blocks:
                block.attention._prefix_kv = None


# ── adapters ────────────────────────────────────────────────────────────────

class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int, device=None) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, device=device)
        self.up = nn.Linear(bottleneck, dim, device=device)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(torch.relu(self.down(x)))


class Adapters(nn.Module):
    """Bottleneck residual adapters after each block's attention."""

    def __init__(self, model, bottleneck: int) -> None:
        super().__init__()
        _freeze(model)
        self.model = model
        dim = model.config.embedding_dim
        # Same device concern as LoRA: build on the base model's device so a
        # model already moved to MPS does not get CPU adapters bolted onto it.
        device = model.token_embedding.weight.device
        self.adapters = nn.ModuleList(
            [BottleneckAdapter(dim, bottleneck, device=device)
             for _ in range(model.config.num_layers)])

    def parameters(self) -> list[nn.Parameter]:
        return list(self.adapters.parameters())

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, token_ids, targets=None, cache=None):
        emb = self.model.token_embedding(token_ids)
        hidden = emb + self.model.position_embedding(
            torch.arange(token_ids.size(1), device=token_ids.device))
        for layer_idx, block in enumerate(self.model.blocks):
            hidden = hidden + block.attention(block.norm_attention(hidden), cache=cache,
                                              layer_idx=layer_idx)
            hidden = hidden + block.mlp(block.norm_mlp(hidden))
            hidden = self.adapters[layer_idx](hidden)
        logits = self.model.lm_head(self.model.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1), ignore_index=-100)
        return logits, loss, cache


# ── uniform interface ───────────────────────────────────────────────────────

class PEFTModel(nn.Module):
    """Wraps a method so the SFT loop always calls ``model(inputs, labels)``."""

    def __init__(self, inner, kind: str) -> None:
        super().__init__()
        self.inner = inner
        self.kind = kind

    @property
    def base(self):
        """The underlying frozen MiniGPT, however deeply it is nested."""
        obj = self.inner
        for _ in range(4):
            if hasattr(obj, "config"):
                return obj
            if hasattr(obj, "inner"):
                obj = obj.inner
            elif hasattr(obj, "model"):
                obj = obj.model
            else:
                break
        raise AttributeError("could not locate the base model config")

    @property
    def config(self):
        """The SFT loop reads ``model.config.context_length``; delegate it."""
        return self.base.config

    def trainable_parameters(self) -> list[nn.Parameter]:
        if self.kind == "full":
            return [p for p in self.inner.parameters() if p.requires_grad]
        return self.inner.parameters()

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def forward(self, token_ids, targets=None, cache=None):
        return self.inner(token_ids, targets, cache)


def apply_peft(model, spec: PEFTSpec) -> PEFTModel:
    """Apply a PEFT method to a fresh model instance."""
    if spec.method == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return PEFTModel(model, "full")
    if spec.method == "lora":
        _freeze(model)
        inject_lora(model, target_suffixes=spec.target_suffixes, rank=spec.size, alpha=2 * spec.size)
        model.lora_params = [p for p in model.parameters() if p.requires_grad]
        return _LoRAWrapper(model)
    if spec.method == "prompt_tuning":
        return PEFTModel(PromptTuning(model, spec.size), spec.method)
    if spec.method == "prefix_tuning":
        return PEFTModel(PrefixTuning(model, spec.size), spec.method)
    if spec.method == "adapters":
        return PEFTModel(Adapters(model, spec.size), spec.method)
    raise ValueError(f"unknown PEFT method: {spec.method}")


class _LoRAWrapper(PEFTModel):
    def __init__(self, model) -> None:
        super().__init__(model, "lora")

    def trainable_parameters(self) -> list[nn.Parameter]:
        return self.inner.lora_params


# ── budget matching ─────────────────────────────────────────────────────────

def n_trainable_for(model, spec: PEFTSpec) -> int:
    """Trainable-parameter count for a spec, without keeping the wrapper.

    LoRA injects new modules, so this is measured on a throwaway copy by the
    caller when an exact count matters; the approximation below is used only to
    choose candidate sizes.
    """
    if spec.method == "full":
        return sum(p.numel() for p in model.parameters())
    if spec.method == "lora":
        # LoRALinear adds r*(in + out) parameters per wrapped Linear. The loop
        # already visits every block, so this must NOT be scaled again.
        total = 0
        for block in model.blocks:
            for suffix in spec.target_suffixes:
                base = getattr(block.attention, suffix, None)
                if isinstance(base, nn.Linear):
                    total += spec.size * (base.in_features + base.out_features)
        return total
    if spec.method in ("prompt_tuning", "prefix_tuning"):
        cfg = model.config
        head = cfg.embedding_dim // cfg.num_heads
        if spec.method == "prompt_tuning":
            return spec.size * cfg.embedding_dim
        return spec.size * cfg.num_kv_heads * head * cfg.num_layers * 2
    if spec.method == "adapters":
        dim = model.config.embedding_dim
        return len(model.blocks) * (2 * dim * spec.size + spec.size + dim)
    raise ValueError(spec.method)


def size_to_budget(model, method: str, budget: int,
                   target_suffixes: tuple[str, ...] = LORA_TARGETS) -> PEFTSpec:
    """Pick the size whose trainable count is closest to `budget`."""
    if method == "lora":
        candidates = list(range(1, 65))
    elif method == "prompt_tuning":
        candidates = list(range(1, MAX_PROMPT_TOKENS + 1))
    else:
        candidates = list(range(1, 129))
    scored = []
    for size in candidates:
        spec = PEFTSpec(method=method, size=size, target_suffixes=target_suffixes)
        scored.append((abs(n_trainable_for(model, spec) - budget), size))
    _, best = min(scored)
    return PEFTSpec(method=method, size=best, target_suffixes=target_suffixes)


METHODS = ("lora", "prompt_tuning", "prefix_tuning", "adapters", "full")


def clamp_to_context(spec: PEFTSpec, context_length: int) -> PEFTSpec:
    """Keep any sequence-extending method inside the context window."""
    if spec.method == "prompt_tuning" and spec.size > MAX_PROMPT_TOKENS:
        return PEFTSpec(method=spec.method, size=MAX_PROMPT_TOKENS,
                        target_suffixes=spec.target_suffixes)
    return spec
