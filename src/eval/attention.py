"""Extract per-head causal attention weights from a MiniGPT forward pass.

``F.scaled_dot_product_attention`` never exposes its weights, so they are
recomputed from the same ``qkv`` projection the model uses. A forward pre-hook
on each attention module captures the (already normalised) block input, which
keeps this independent of the ``optimized_attention`` setting.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.model import MiniGPT


@torch.no_grad()
def attention_weights(model: MiniGPT, token_ids: torch.Tensor) -> torch.Tensor:
    """Return softmax attention weights shaped ``[layers, heads, T, T]``.

    ``token_ids`` is a single ``torch.long`` sequence shaped ``[1, T]`` with
    ``T <= context_length``. Row ``i`` of each ``[T, T]`` map sums to 1 and has
    zero weight on positions after ``i`` (causality).
    """
    if token_ids.ndim != 2 or token_ids.size(0) != 1 or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a torch.long tensor shaped [1, time]")

    inputs: list[torch.Tensor] = []
    hooks = [
        block.attention.register_forward_pre_hook(lambda _module, args: inputs.append(args[0].detach()))
        for block in model.blocks
    ]
    was_training = model.training
    model.eval()
    try:
        model(token_ids)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    maps = []
    for block, hidden in zip(model.blocks, inputs):
        attention = block.attention
        batch, sequence, _ = hidden.shape
        qkv = attention.qkv(hidden).view(batch, sequence, 3, attention.num_heads, attention.head_dim)
        query, key, _ = qkv.unbind(dim=2)
        query, key = query.transpose(1, 2), key.transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) * (attention.head_dim**-0.5)
        causal = torch.ones(sequence, sequence, dtype=torch.bool, device=hidden.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        maps.append(F.softmax(scores, dim=-1)[0])
    return torch.stack(maps)
