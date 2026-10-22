"""Speculative decoding (Lecture 16, Leviathan et al. 2023) with KV cache.

A small draft model proposes K tokens autoregressively; the large target model
verifies all K in ONE forward pass; accepted tokens are kept and the first
rejection is resampled from the corrected residual distribution.

Correctness: the emitted token distribution is EXACTLY the target model's own
distribution. On acceptance the draft token was drawn from q and verified against
p, so it is kept with probability min(1, p/q). On rejection we resample from
normalised max(0, p - q). This is verified statistically in
tests/test_speculative.py and tests/test_speculative_real.py, not assumed.

Cache invariant (this is the part that is easy to get wrong):
    At the top of every round, each cache holds EXACTLY the tokens of the
    accepted sequence `target`, and the held "base" logits are the distribution
    for the position immediately AFTER `target`.

Every forward pass inside a round therefore feeds only NEW tokens, never the
whole prefix. An earlier version of this file re-fed the full sequence each
round on top of an already-prefilled cache, silently double-counting the
context: the target's p then differed from the draft's q even when draft and
target were the same weights, and the acceptance rate collapsed to ~40% instead
of ~100%. tests/test_speculative_cache.py pins the invariant.

On rejection both caches are rolled back to the accepted prefix via
KVCache.truncate, then the resampled token is fed to advance them.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import torch
from torch.nn import functional as F

from src.model.gpt import KVCache


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Normalised max(0, p - q): the distribution to resample from on rejection."""
    residual = torch.clamp(p - q, min=0.0)
    total = residual.sum()
    if float(total) <= 0.0:
        return p
    return residual / total


def _make_cache(model) -> KVCache:
    cfg = model.config
    return KVCache(
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        head_dim=cfg.embedding_dim // cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
    )


def _pick(probs: torch.Tensor, do_sample: bool) -> int:
    if do_sample:
        return int(torch.multinomial(probs, 1).item())
    return int(probs.argmax().item())


@torch.no_grad()
def _cached_probs(model, tokens: torch.Tensor, cache: KVCache, temperature: float) -> torch.Tensor:
    """Feed ONLY `tokens` (new ones) into the cache; return next-token distribution."""
    logits, _, _ = model(tokens, cache=cache)
    return F.softmax(logits[:, -1, :] / temperature, dim=-1)[0]


@torch.no_grad()
def _full_probs(model, tokens: torch.Tensor, temperature: float) -> torch.Tensor:
    """No-cache reference: run the whole sequence, return next-token distribution."""
    logits, _, _ = model(tokens)
    return F.softmax(logits[:, -1, :] / temperature, dim=-1)[0]


@torch.no_grad()
def speculative_decode(
    draft_model,
    target_model,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    K: int = 3,
    do_sample: bool = True,
    temperature: float = 1.0,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Decode with speculative sampling. Returns tokens plus telemetry.

    Telemetry: acceptance_rate, forward_calls (counted separately for the
    cached and uncached paths so the two can be compared fairly), seconds,
    tokens_per_second.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    start = time.perf_counter()
    target = prompt
    plen = prompt.shape[1]
    emitted: list[int] = []
    proposed_total = 0
    accepted_total = 0
    calls = {"draft": 0, "target": 0}

    if use_cache:
        draft_cache = _make_cache(draft_model)
        target_cache = _make_cache(target_model)
        # Prefill: caches now hold the prompt, bases predict position `plen`.
        draft_base = _cached_probs(draft_model, prompt, draft_cache, temperature)
        target_base = _cached_probs(target_model, prompt, target_cache, temperature)
        calls["draft"] += 1
        calls["target"] += 1

    while len(emitted) < max_new_tokens:
        L0 = target.shape[1]
        n_cand = min(K, max_new_tokens - len(emitted))

        # ── 1. draft n_cand tokens; dprobs[j] predicts position L0 + j ──────
        cands: list[int] = []
        dprobs: list[torch.Tensor] = []
        for j in range(n_cand):
            if use_cache:
                probs = draft_base if j == 0 else _cached_probs(
                    draft_model, torch.tensor([[cands[j - 1]]], dtype=torch.long),
                    draft_cache, temperature)
                if j > 0:
                    calls["draft"] += 1
            else:
                probs = _full_probs(
                    draft_model,
                    torch.cat([target, torch.tensor([cands[:j]], dtype=torch.long)], dim=1)
                    if j else target,
                    temperature)
                calls["draft"] += 1
            dprobs.append(probs)
            tok = _pick(probs, do_sample)
            cands.append(tok)
        proposed_total += n_cand

        # One extra draft forward so dprobs[n_cand] exists and the draft cache
        # ends the round at the same length as the target cache. Without it the
        # next round's base logits would have to be recomputed from scratch.
        if use_cache:
            dprobs.append(_cached_probs(
                draft_model, torch.tensor([[cands[-1]]], dtype=torch.long),
                draft_cache, temperature))
            calls["draft"] += 1

        # ── 2. target verifies; p_seq[i] predicts position L0 + i ──────────
        p_seq: list[torch.Tensor] = []
        for i in range(n_cand + 1):
            if use_cache:
                if i == 0:
                    p_seq.append(target_base)
                else:
                    p_seq.append(_cached_probs(
                        target_model, torch.tensor([[cands[i - 1]]], dtype=torch.long),
                        target_cache, temperature))
                    calls["target"] += 1
            else:
                seq = torch.cat([target, torch.tensor([cands[:i]], dtype=torch.long)], dim=1) \
                    if i else target
                p_seq.append(_full_probs(target_model, seq, temperature))
                calls["target"] += 1

        # ── 3. accept prefix, resample at the first rejection ───────────────
        round_start = len(emitted)
        accepted_here = 0
        new_token: int | None = None
        broke = False
        for i, cand in enumerate(cands):
            p, q_i = p_seq[i], dprobs[i]
            if do_sample:
                q_cand, p_cand = float(q_i[cand]), float(p[cand])
                take = q_cand > 0 and torch.rand(1).item() <= min(1.0, p_cand / q_cand)
            else:
                take = int(p.argmax().item()) == cand
            if take:
                emitted.append(cand)
                accepted_here += 1
                accepted_total += 1
                continue
            new_token = _pick(residual_distribution(p, q_i) if do_sample else p, do_sample)
            emitted.append(new_token)
            broke = True
            break

        if not broke and len(emitted) < max_new_tokens:
            # Everything accepted: take one bonus token from the target's own
            # distribution at p_seq[n_cand] (the position after the last candidate).
            new_token = _pick(p_seq[n_cand], do_sample)
            emitted.append(new_token)

        # ── 4. realign caches to the new accepted sequence ─────────────────
        if use_cache:
            target_cache.truncate(L0 + accepted_here)
            draft_cache.truncate(L0 + accepted_here)
            if new_token is not None:
                tok_t = torch.tensor([[new_token]], dtype=torch.long)
                target_base = _cached_probs(target_model, tok_t, target_cache, temperature)
                draft_base = _cached_probs(draft_model, tok_t, draft_cache, temperature)
                calls["target"] += 1
                calls["draft"] += 1
            else:
                # Token budget hit exactly on an accepted candidate; both p_seq
                # and dprobs already hold the post-feed logits for this length.
                target_base = p_seq[accepted_here]
                draft_base = dprobs[accepted_here]

        target = torch.cat([target, torch.tensor([emitted[round_start:]], dtype=torch.long)], dim=1)

    elapsed = time.perf_counter() - start
    tokens = torch.tensor(emitted, dtype=torch.long).unsqueeze(0)
    return {
        "tokens": tokens,
        "sequence": target,
        "n_emitted": len(emitted),
        "n_proposed": proposed_total,
        "n_accepted": accepted_total,
        "acceptance_rate": accepted_total / proposed_total if proposed_total else 0.0,
        "forward_calls": calls["draft"] + calls["target"],
        "draft_forward_calls": calls["draft"],
        "target_forward_calls": calls["target"],
        "seconds": elapsed,
        "tokens_per_second": len(emitted) / elapsed if elapsed > 0 else float("inf"),
    }


@torch.no_grad()
def speculative_sample_tokens(
    draft_model,
    target_model,
    prompt: torch.Tensor,
    *,
    K: int = 3,
    n_samples: int = 1000,
    vocab: int = 32,
) -> dict[int, int]:
    """Empirical first-token distribution over many independent samples."""
    counts: Counter[int] = Counter()
    for i in range(n_samples):
        torch.manual_seed(10_000 + i)
        out = speculative_decode(draft_model, target_model, prompt,
                                 max_new_tokens=1, K=K, do_sample=True)
        counts[int(out["tokens"][0, 0].item())] += 1
    return dict(counts)
