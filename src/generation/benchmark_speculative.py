"""Wall-clock benchmark: speculative decoding vs plain generation.

Measures total wall clock including draft-model time, against plain generation
with the KV cache enabled. That is the honest baseline — plain generation with
no cache would flatter speculative decoding.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from src.generation.speculative import speculative_decode
from src.generation.decoding import generate


@torch.no_grad()
def benchmark_speculative(
    draft_model,
    target_model,
    prompt: torch.Tensor,
    *,
    K_values: tuple[int, ...] = (1, 2, 3, 5, 7),
    max_new_tokens: int = 32,
    n_repeats: int = 3,
    do_sample: bool = True,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Benchmark speculative decoding at each K against plain target generation.

    The baseline is plain generation from the TARGET with the KV cache enabled.
    That is the honest comparison: plain generation without a cache would
    re-compute the whole prefix every step and flatter speculative decoding.

    All timings include draft-model time, since the draft model is part of the
    cost of this method and omitting it would overstate the speedup.

    `context_length` is taken from each model's own config, NOT from the prompt
    length — passing the prompt length would silently truncate the context
    window to the prompt and make both arms look far cheaper than they are.
    """
    ctx = target_model.config.context_length
    if prompt.shape[1] + max_new_tokens > ctx:
        raise ValueError(
            f"prompt {prompt.shape[1]} + {max_new_tokens} new exceeds context_length {ctx}; "
            "shorten the prompt or reduce max_new_tokens"
        )

    plain_times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        generate(target_model, prompt, context_length=ctx,
                 max_new_tokens=max_new_tokens, do_sample=do_sample,
                 temperature=temperature, use_cache=True)
        plain_times.append(time.perf_counter() - t0)
    plain_seconds = min(plain_times)  # best-of, to reduce scheduler noise

    results = {}
    for K in K_values:
        times = []
        acceptance_rates = []
        token_counts = []
        for _ in range(n_repeats):
            out = speculative_decode(
                draft_model, target_model, prompt,
                max_new_tokens=max_new_tokens, K=K,
                do_sample=do_sample, temperature=temperature,
                use_cache=True)
            times.append(out["seconds"])
            acceptance_rates.append(out["acceptance_rate"])
            token_counts.append(out["n_emitted"])
        spec_seconds = min(times)
        n_emitted = min(token_counts)
        results[K] = {
            "K": K,
            "spec_seconds": spec_seconds,
            "plain_seconds": plain_seconds,
            "speedup": plain_seconds / spec_seconds if spec_seconds > 0 else 0.0,
            "acceptance_rate": sum(acceptance_rates) / len(acceptance_rates),
            "tokens_per_second": n_emitted / spec_seconds if spec_seconds > 0 else 0.0,
        }
    return results


def format_benchmark(results: dict[int, dict[str, Any]]) -> str:
    lines = [
        f"{'K':>3} {'spec_s':>8} {'plain_s':>8} {'speedup':>8} {'accept':>8} {'tok/s':>8}",
        "-" * 50,
    ]
    for K in sorted(results):
        r = results[K]
        lines.append(
            f"{K:>3} {r['spec_seconds']:>8.2f} {r['plain_seconds']:>8.2f} "
            f"{r['speedup']:>7.2f}x {r['acceptance_rate']:>7.1%} {r['tokens_per_second']:>8.1f}"
        )
    return "\n".join(lines)