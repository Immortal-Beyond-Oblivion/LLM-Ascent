"""Greedy, beam, top-k, and nucleus decoding for causal models."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import functional as F

Model = Callable[[torch.Tensor], tuple[torch.Tensor, object]]


def _last_logits(model: Model, tokens: torch.Tensor, context_length: int) -> torch.Tensor:
    logits, _ = model(tokens[:, -context_length:])
    return logits[:, -1, :]


def _filter(logits: torch.Tensor, *, top_k: int | None, top_p: float | None) -> torch.Tensor:
    filtered = logits.clone()
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        threshold = torch.topk(filtered, min(top_k, filtered.size(-1))).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, indices = torch.sort(filtered, descending=True)
        remove = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        filtered.scatter_(dim=-1, index=indices, src=sorted_logits.masked_fill(remove, float("-inf")))
    return filtered


@torch.no_grad()
def generate(
    model: Model,
    prompt: torch.Tensor,
    *,
    context_length: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    do_sample: bool = False,
) -> torch.Tensor:
    """Generate a continuation, preserving prompt and returning ``[B, T+N]``."""
    if prompt.ndim != 2 or prompt.dtype != torch.long:
        raise ValueError("prompt must be a torch.long tensor shaped [batch, time]")
    if max_new_tokens < 0 or temperature <= 0:
        raise ValueError("max_new_tokens must be non-negative and temperature positive")
    result = prompt
    for _ in range(max_new_tokens):
        logits = _last_logits(model, result, context_length) / temperature
        if do_sample:
            next_token = torch.multinomial(F.softmax(_filter(logits, top_k=top_k, top_p=top_p), dim=-1), 1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
        result = torch.cat((result, next_token), dim=1)
    return result


def greedy_decode(model: Model, prompt: torch.Tensor, *, context_length: int, max_new_tokens: int) -> torch.Tensor:
    return generate(model, prompt, context_length=context_length, max_new_tokens=max_new_tokens)


@torch.no_grad()
def beam_search(
    model: Model,
    prompt: torch.Tensor,
    *,
    context_length: int,
    max_new_tokens: int,
    beam_width: int = 3,
) -> torch.Tensor:
    """Return the highest log-probability continuation for a batch of one."""
    if prompt.size(0) != 1 or beam_width <= 0:
        raise ValueError("beam search supports one prompt and a positive beam_width")
    beams: list[tuple[torch.Tensor, float]] = [(prompt, 0.0)]
    for _ in range(max_new_tokens):
        candidates: list[tuple[torch.Tensor, float]] = []
        for tokens, score in beams:
            log_probs = F.log_softmax(_last_logits(model, tokens, context_length), dim=-1)
            values, ids = torch.topk(log_probs, beam_width, dim=-1)
            for value, token_id in zip(values[0], ids[0]):
                candidates.append((torch.cat((tokens, token_id.reshape(1, 1)), dim=1), score + float(value)))
        beams = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)[:beam_width]
    return beams[0][0]
