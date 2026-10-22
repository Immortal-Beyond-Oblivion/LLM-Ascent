"""Greedy, beam, top-k, and nucleus decoding for causal models."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import functional as F

Model = Callable[[torch.Tensor], tuple[torch.Tensor, object]]


def _last_logits(model: Model, tokens: torch.Tensor, context_length: int) -> torch.Tensor:
    result = model(tokens[:, -context_length:])
    logits = result[0] if isinstance(result, tuple) else result
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
    use_cache: bool = False,
) -> torch.Tensor:
    """Generate a continuation, preserving prompt and returning ``[B, T+N]``."""
    if prompt.ndim != 2 or prompt.dtype != torch.long:
        raise ValueError("prompt must be a torch.long tensor shaped [batch, time]")
    if max_new_tokens < 0 or temperature <= 0:
        raise ValueError("max_new_tokens must be non-negative and temperature positive")

    if not use_cache:
        result = prompt
        for _ in range(max_new_tokens):
            logits = _last_logits(model, result, context_length) / temperature
            if do_sample:
                next_token = torch.multinomial(F.softmax(_filter(logits, top_k=top_k, top_p=top_p), dim=-1), 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            result = torch.cat((result, next_token), dim=1)
        return result

    # Cached path: prefill then incremental decode
    from src.model.gpt import KVCache

    # Infer model dimensions for cache
    num_layers = len(model.blocks)
    num_heads = model.config.num_heads
    head_dim = model.config.embedding_dim // num_heads
    cache = KVCache(num_layers, num_heads, head_dim)

    # Prefill: process full prompt
    logits, _, cache = model(prompt, cache=cache)
    logits = logits[:, -1, :] / temperature

    result = prompt
    for _ in range(max_new_tokens):
        if do_sample:
            next_token = torch.multinomial(F.softmax(_filter(logits, top_k=top_k, top_p=top_p), dim=-1), 1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
        result = torch.cat((result, next_token), dim=1)
        if _ < max_new_tokens - 1:
            logits, _, cache = model(next_token, cache=cache)
            logits = logits[:, -1, :] / temperature

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
    use_cache: bool = False,
) -> torch.Tensor:
    """Return the highest log-probability continuation for a batch of one.

    With ``use_cache=True`` each beam keeps its OWN KV cache and every
    expansion feeds a SINGLE new token, instead of re-running the whole prefix
    for every candidate at every step. The two paths are numerically
    equivalent; ``tests/test_kv_cache.py`` asserts that, because the uncached
    path is the reference and an equivalence test alone cannot tell us the
    cached one is not merely differently wrong.
    """
    if prompt.size(0) != 1 or beam_width <= 0:
        raise ValueError("beam search supports one prompt and a positive beam_width")
    if not use_cache:
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

    if prompt.shape[1] + max_new_tokens > context_length:
        raise ValueError(
            f"prompt {prompt.shape[1]} + {max_new_tokens} new exceeds "
            f"context_length {context_length}"
        )

    from src.model.gpt import KVCache

    cfg = model.config
    base = KVCache(len(model.blocks), cfg.num_heads,
                   cfg.embedding_dim // cfg.num_heads, cfg.num_kv_heads)
    logits, _, _ = model(prompt, cache=base)

    # A beam holds (tokens, cache covering exactly those tokens, score,
    # last) where `last` is the distribution over the next token, already
    # computed. Because `last` is available, expansion to scores needs NO
    # forward pass at all; forwards are spent only on the beam_width
    # SURVIVORS, whose caches must be advanced by their chosen token.
    beams: list[list] = [[prompt, base, 0.0, logits[:, -1, :]]]

    for step in range(max_new_tokens):
        scored: list[tuple[float, int, int, torch.Tensor]] = []
        for idx, (tokens, cache, score, last) in enumerate(beams):
            log_probs = F.log_softmax(last, dim=-1)
            values, ids = torch.topk(log_probs, beam_width, dim=-1)
            for value, token_id in zip(values[0], ids[0]):
                scored.append((score + float(value), idx, int(token_id), tokens))

        scored.sort(key=lambda c: c[0], reverse=True)
        winners = scored[:beam_width]

        best_score, best_idx, best_tok, best_tokens = winners[0]
        best_full = torch.cat(
            (best_tokens,
             torch.tensor([[best_tok]], dtype=torch.long, device=prompt.device)), dim=1)

        if step == max_new_tokens - 1:
            return best_full  # last step: no cache work needed at all

        # One clone and one forward per survivor, never per candidate.
        # Cloning per candidate cost beam_width^2 deep copies of the whole
        # prefix per step and made a naive cached beam 2.5x SLOWER.
        new_beams = []
        for score, idx, tok_id, tokens in winners:
            tok = torch.tensor([[tok_id]], dtype=torch.long, device=prompt.device)
            out, _, child = model(tok, cache=beams[idx][1].clone())
            new_beams.append([torch.cat((tokens, tok), dim=1), child, score, out[:, -1, :]])
        beams = new_beams

    return beams[0][0]
