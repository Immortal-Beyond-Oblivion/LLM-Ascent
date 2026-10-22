"""Compute per-rung (non_embedding, embedding, total) parameter breakdown.

Run BEFORE the probe so we can decide: (a) is embedding share acceptable?
(b) should we re-specify the ladder in terms of total params vs non-embedding params?
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from src.scaling.ladder import LADDER, build_model_for_rung

# Simulated vocab size (will be updated from manifest after pre-tokenization)
VOCAB_SIZE = 4164
CONTEXT_LENGTH = 128




def main() -> None:
    print(f"\n{'='*80}")
    print(f"PER-RUNG PARAMETER BREAKDOWN (vocab={VOCAB_SIZE:,}, context={CONTEXT_LENGTH})")
    print(f"{'='*80}")
    print(f"{'Rung':<8} {'Non-emb':>12} {'Embed':>12} {'Total':>12} {'Emb%':>8} {'Layers':>8} {'Dim':>6}")
    print("-" * 75)

    total_params = 0
    for rung in LADDER:
        model = build_model_for_rung(rung, VOCAB_SIZE, CONTEXT_LENGTH)
        non_emb = model.non_embedding_params()
        emb = model.token_embedding.weight.numel()
        total = model.total_params()
        emb_pct = emb / total * 100

        # Need to recover dim/layers from the model
        dim = model.blocks[0].norm1.weight.shape[0]
        layers = len(model.blocks)
        total_params += total

        print(f"{rung.name:<8} {non_emb:>12,} {emb:>12,} {total:>12,} {emb_pct:>7.1f}% {layers:>8} {dim:>6}")

    print("-" * 75)
    print(f"{'TOTAL':<8} {'':>12} {'':>12} {total_params:>12,}")
    print(f"\nC4 corpus: 344,146,089 tokens usable at 2.235 chars/token")
    print(f"Max N: {VOCAB_SIZE:,} vocab requires D <= {8_000_000 * 20:,} tokens for D=20N")

    print(f"\n{'='*80}")
    print("ANALYSIS:")
    print("-" * 80)
    # Check if embedding share is acceptable
    print("Embedding share per rung:")
    for rung in LADDER:
        model = build_model_for_rung(rung, VOCAB_SIZE, CONTEXT_LENGTH)
        emb_pct = model.token_embedding.weight.numel() / model.total_params() * 100
        if emb_pct > 50:
            print(f"  WARNING: {rung.name} has {emb_pct:.1f}% embedding params — may distort scaling")
        else:
            print(f"  {rung.name}: {emb_pct:.1f}% embedding params — OK")

    print(f"\nRECOMMENDATION:")
    if VOCAB_SIZE >= 4000:
        print("  Vocabulary is still large enough that embedding dominates at small rungs.")
        print("  Options:")
        print("  (a) Reduce vocab further (2000-3000) — smaller embedding, cleaner non-emb scaling")
        print("  (b) Report total params instead of non-embedding params")
        print("  (c) Keep 4000 vocab but add an explicit embedding share caveat to the report")
    else:
        print("  Vocabulary is small enough that embedding share is acceptable.")


if __name__ == "__main__":
    main()
