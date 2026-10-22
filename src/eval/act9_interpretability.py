"""Act 9: interpretability, measured rather than illustrated.

Three experiments, each with an external ground truth so the result is a
behavioural claim rather than a heatmap:

9a  CAUSAL HEAD ABLATION. Every (layer, head) of the canonical 6.3M is
    disabled and the held-out loss is re-measured. The null distribution comes
    from ablating RANDOM heads, so "this head matters" is a statement about a
    null rather than an eyeball, and every head gets a z-score. The important
    question is not which head has the largest delta but whether the heads that
    matter TOGETHER matter more than the same number of random heads.

9b  BEHAVIOURAL ATTENTION ROLLOUT. Act 8 supplies a ground truth that a
    heatmap does not need: the token holding the number the passage asserts.
    Rollout attribution is scored by whether it actually identifies that token,
    and compared against raw attention and against a uniform baseline. A
    heatmap proves nothing; a hit rate against a known answer does.

9c  LOGIT LENS, PER LAYER. Does the answer ever become readable from the
    residual stream? The lens is the model's own linear readout (final norm
    plus the tied unembedding) applied to each layer's residual, so "decodable"
    means decodable by a readout the model already owns -- no probe is
    smuggled in. The per-layer rank of the correct answer token is the
    measurement.

The model is the pinned canonical 6.3M (8 layers x 8 heads = 64 heads, dim
256, context 128). The held-out slice is taken from the C4 shard at a document
offset far beyond anything the training sweep touched, and the intact
baseline is measured on the SAME slice in the SAME run, so every delta is
internal to this experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from src.bench.stats import bootstrap_mean_ci, wilson_interval

SEED = 42
NULL_SETS = 40


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def build_heldout(tokenizer, *, offset: int, n_sequences: int,
                  context_length: int, path: str = "data/v2/c4_en_shard_0.jsonl",
                  ) -> list[list[int]]:
    """Tokenized C4 sequences from a document offset far from the training data.

    The canonical checkpoints were trained on a C4 memmap built from the head
    of the shard. This slice starts at ``offset`` documents in, so it is not
    the training text.
    """
    import itertools
    import json as _json

    seqs: list[list[int]] = []
    buffer: list[int] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(itertools.islice(fh, offset, offset + 4000)):
            if len(seqs) >= n_sequences:
                break
            try:
                doc = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            buffer.extend(tokenizer.encode(doc.get("text") or ""))
            while len(buffer) >= context_length and len(seqs) < n_sequences:
                seqs.append(buffer[:context_length])
                buffer = buffer[context_length:]
    if len(seqs) < n_sequences:
        raise ValueError(f"only {len(seqs)} held-out sequences available")
    return seqs


# --------------------------------------------------------------------------
# 9a -- causal head ablation
# --------------------------------------------------------------------------

def _attention_modules(model):
    return [block.attention for block in model.blocks]


def measure_ce(model, sequences: list[list[int]], *, device: str,
               batch_size: int = 8) -> float:
    """Mean cross-entropy per token over the held-out slice.

    The targets are SHIFTED: logits at position i predict token i+1, so the
    loss must compare logits[:, :-1] against x[:, 1:]. Comparing logits
    against x at the same position -- which is what a naive CE(logits, x)
    does -- scores the model's ability to predict the token it is already
    looking at, and returns a meaningless number (CE ~ 12, perplexity ~ 1.6e5
    for a model whose real held-out loss is ~2.5).
    """
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start:start + batch_size]
            x = torch.tensor(chunk, dtype=torch.long, device=device)
            logits, _, _ = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1), reduction="sum")
            total += float(loss)
            count += x[:, 1:].numel()
    return total / count


def _set_ablations(model, per_layer: dict[int, set[int]]) -> None:
    for layer, attn in enumerate(_attention_modules(model)):
        attn.ablate_heads = set(per_layer.get(layer, ()))


def head_ablation(model, sequences, *, device: str) -> list[dict]:
    """Ablate every head in turn and record the change in held-out loss."""
    _set_ablations(model, {})
    intact_ce = measure_ce(model, sequences, device=device)
    intact_ppl = math.exp(intact_ce)
    rows = []
    for layer, attn in enumerate(_attention_modules(model)):
        for head in range(attn.num_heads):
            _set_ablations(model, {layer: {head}})
            ce = measure_ce(model, sequences, device=device)
            rows.append({
                "layer": layer, "head": head,
                "delta_ce": ce - intact_ce,
                "delta_ppl": math.exp(ce) - intact_ppl,
                "ablated_ce": ce, "ablated_ppl": math.exp(ce),
            })
    _set_ablations(model, {})
    # The null's deltas are measured against THIS run's intact baseline, so it
    # must be recorded here rather than by the caller. A caller that forgets
    # gets null deltas measured against a stale baseline and z-scores that are
    # wildly wrong (they came out at -88 instead of +7.8).
    _INTACT_CE[0] = intact_ce
    return rows, intact_ce, intact_ppl


def random_ablation_null(model, sequences, *, device: str, n_sets: int,
                         rng: random.Random) -> list[dict]:
    """Ablate RANDOM head sets to build the null distribution.

    Sizes 1, 2 and 3 are sampled so the null covers the single-head case
    (which is what each real head is compared against) and the multi-head case
    (which is what the top-k combination is compared against).
    """
    attns = _attention_modules(model)
    all_heads = [(l, h) for l, a in enumerate(attns) for h in range(a.num_heads)]
    out = []
    for i in range(n_sets):
        size = (1, 2, 3)[i % 3]
        chosen = rng.sample(all_heads, size)
        per_layer: dict[int, set[int]] = {}
        for l, h in chosen:
            per_layer.setdefault(l, set()).add(h)
        _set_ablations(model, per_layer)
        ce = measure_ce(model, sequences, device=device)
        out.append({"size": size, "heads": [list(c) for c in chosen],
                    "delta_ce": ce - _INTACT_CE[0]})
    _set_ablations(model, {})
    return out


_INTACT_CE = [0.0]


def attention_entropy(model, sequences, *, device: str) -> dict:
    """Mean attention entropy per (layer, head), in nats."""
    for attn in _attention_modules(model):
        attn.record_attention = True
    entropies: list[list[float]] = [[] for _ in _attention_modules(model)]
    with torch.no_grad():
        for start in range(0, len(sequences), 8):
            chunk = sequences[start:start + 8]
            x = torch.tensor(chunk, dtype=torch.long, device=device)
            model(x)
            for layer, attn in enumerate(_attention_modules(model)):
                a = attn.last_attn.float()  # [B, H, T, T]
                ent = -(a * torch.log(a.clamp_min(1e-12))).sum(-1)  # [B, H, T]
                entropies[layer].append(ent.mean(dim=(0, 2)).tolist())
    for attn in _attention_modules(model):
        attn.record_attention = False
        attn.last_attn = None
    out = {}
    for layer, per_head in enumerate(entropies):
        for head, values in enumerate(zip(*per_head)):
            out[f"{layer},{head}"] = sum(values) / len(values)
    return out


# --------------------------------------------------------------------------
# 9b -- behavioural attention rollout
# --------------------------------------------------------------------------

def rollout_matrices(attentions: list[torch.Tensor]) -> list[torch.Tensor]:
    """Abnar & Zuidema rollout: R_1 = A_1, R_l = A_l (I + R_{l-1}).

    Each A_l is the head-averaged attention of one layer, row-normalised.
    The result is then row-normalised as well: the raw recursion gives row
    sums of 2 (because of the I + R term), so without this the "attribution"
    would not be a distribution over input positions and could not be
    compared with a uniform baseline.
    """
    eye = None
    result = []
    for a in attentions:
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)
        if eye is None:
            # On the same device as the attention matrices, or the MPS/CPU mix
            # raises.
            eye = torch.eye(a.size(-1), dtype=a.dtype, device=a.device)
        r = a @ (eye + result[-1]) if result else a
        r = r / r.sum(-1, keepdim=True).clamp_min(1e-9)
        result.append(r)
    return result


def _capture_attentions(model, ids: list[int], *, device: str):
    """Return per-layer head-averaged attention matrices for one sequence."""
    for attn in _attention_modules(model):
        attn.record_attention = True
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        model(x)
    mats = []
    for attn in _attention_modules(model):
        a = attn.last_attn.float()[0].mean(dim=0)  # [T, T]
        mats.append(a)
    for attn in _attention_modules(model):
        attn.record_attention = False
        attn.last_attn = None
    return mats


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    """First index where ``needle`` occurs in ``haystack``, or -1."""
    if not needle:
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


# --------------------------------------------------------------------------
# 9c -- logit lens
# --------------------------------------------------------------------------

def _capture_residuals(model, ids: list[int], *, device: str):
    """Residual stream after each block, for one sequence."""
    captured: list[torch.Tensor] = []
    handles = []

    def hook(_module, _inp, out):
        captured.append(out.detach())

    for block in model.blocks:
        handles.append(block.register_forward_hook(hook))
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return captured


def lens_logits(model, residuals: list[torch.Tensor]) -> list[torch.Tensor]:
    """Apply the model's own readout to each layer's residual."""
    out = []
    for r in residuals:
        h = model.final_norm(r)
        out.append(model.lm_head(h))
    return out


def _rank_of(scores: torch.Tensor, target: int) -> int:
    return int((scores > scores[target]).sum())


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def run(rung: str, *, n_sequences: int, n_rollout_items: int, device: str,
        offset: int = 300_000) -> dict:
    from src.eval.act7_sft import load_canonical
    from src.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer.load("checkpoints/canonical/tokenizer.json")
    model = load_canonical(rung).to(device)
    model.eval()
    ctx = model.config.context_length

    sequences = build_heldout(tokenizer, offset=offset,
                              n_sequences=n_sequences, context_length=ctx)
    t0 = time.time()

    # ---- 9a -------------------------------------------------------------
    rows, intact_ce, intact_ppl = head_ablation(model, sequences, device=device)
    _INTACT_CE[0] = intact_ce
    rng = random.Random(SEED)
    null_rows = random_ablation_null(model, sequences, device=device,
                                      n_sets=NULL_SETS, rng=rng)
    entropies = attention_entropy(model, sequences, device=device)

    deltas = [r["delta_ce"] for r in rows]
    mean_d = sum(deltas) / len(deltas)
    sd_d = math.sqrt(sum((d - mean_d) ** 2 for d in deltas) / (len(deltas) - 1))
    for r in rows:
        r["z_score"] = (r["delta_ce"] - mean_d) / sd_d if sd_d > 0 else 0.0
        r["mean_entropy_nats"] = entropies[f"{r['layer']},{r['head']}"]

    null_single = [n["delta_ce"] for n in null_rows if n["size"] == 1]
    null_mean = sum(null_single) / len(null_single)
    null_sd = math.sqrt(sum((d - null_mean) ** 2
                            for d in null_single) / (len(null_single) - 1))
    for r in rows:
        r["z_vs_random_null"] = ((r["delta_ce"] - null_mean) / null_sd
                                  if null_sd > 0 else 0.0)

    ranked = sorted(rows, key=lambda r: r["delta_ce"], reverse=True)
    top = ranked[:4]
    top_heads = {(r["layer"], r["head"]) for r in top}

    # Does the top-k combination matter more than k random heads?
    _set_ablations(model, {l: {h for (ll, h) in top_heads if ll == l}
                           for l in {ll for ll, _ in top_heads}})
    top_ce = measure_ce(model, sequences, device=device)
    _set_ablations(model, {})
    random_combos = []
    rng2 = random.Random(SEED + 1)
    attns = _attention_modules(model)
    all_heads = [(l, h) for l, a in enumerate(attns) for h in range(a.num_heads)]
    for _ in range(20):
        chosen = rng2.sample(all_heads, 4)
        per_layer: dict[int, set[int]] = {}
        for l, h in chosen:
            per_layer.setdefault(l, set()).add(h)
        _set_ablations(model, per_layer)
        random_combos.append(measure_ce(model, sequences, device=device) - intact_ce)
    _set_ablations(model, {})
    top_delta = top_ce - intact_ce
    rand_mean = sum(random_combos) / len(random_combos)

    # entropy relationship
    import statistics
    hi = [r for r in rows if r["z_score"] >= 2.0]
    lo = [r for r in rows if r["z_score"] <= 0.0]
    entropy_hi = statistics.mean([r["mean_entropy_nats"] for r in hi]) if hi else None
    entropy_lo = statistics.mean([r["mean_entropy_nats"] for r in lo]) if lo else None

    # ---- 9b -------------------------------------------------------------
    from src.eval.act8_hallucination import (assemble, load_passages)
    items = load_passages("data/v2/c4_en_shard_0.jsonl", limit=120,
                          tokenizer=tokenizer)[:n_rollout_items]
    rollout_hits, raw_hits = [], []
    n_ground_truth = 0
    for item in items:
        prompt, fitted = assemble("oracle", item, "", tokenizer=tokenizer,
                                  context_length=ctx)
        ids = tokenizer.encode(prompt)
        number_ids = tokenizer.encode(item["source_number"])
        pos = _find_subsequence(ids, number_ids)
        if pos < 0:
            continue
        n_ground_truth += 1
        mats = _capture_attentions(model, ids, device=device)
        roll = rollout_matrices(mats)
        last = roll[-1][-1]  # attribution of the final position
        raw = mats[-1][-1]
        rollout_hits.append(1.0 if int(last.argmax()) == pos else 0.0)
        raw_hits.append(1.0 if int(raw.argmax()) == pos else 0.0)

    # ---- 9c -------------------------------------------------------------
    lens_rows = []
    for item in items:
        prompt, fitted = assemble("oracle", item, "", tokenizer=tokenizer,
                                  context_length=ctx)
        ids = tokenizer.encode(prompt)
        number_ids = tokenizer.encode(item["source_number"])
        pos = _find_subsequence(ids, number_ids)
        if pos < 0:
            continue
        residuals = _capture_residuals(model, ids, device=device)
        lens = lens_logits(model, residuals)
        target_id = ids[pos]
        # lens[i] is [batch, seq, vocab]; the answer position is the last one.
        ranks = [_rank_of(l[0, -1].float(), target_id) for l in lens]
        lens_rows.append({"ranks": ranks,
                          "argmax_layer": next((i for i, r in enumerate(ranks)
                                                if r == 0), None)})

    lens_median = []
    if lens_rows:
        for layer in range(len(lens_rows[0]["ranks"])):
            values = sorted(r["ranks"][layer] for r in lens_rows)
            mid = len(values) // 2
            lens_median.append(values[mid] if len(values) % 2
                               else (values[mid - 1] + values[mid]) / 2)

    del model
    if device == "mps":
        torch.mps.empty_cache()

    return {
        "act": "9",
        "rung": rung,
        "model": {"num_layers": 8, "num_heads": 8, "num_ablated_heads": len(rows),
                  "embedding_dim": 256, "context_length": ctx},
        "held_out": {
            "source": "data/v2/c4_en_shard_0.jsonl",
            "document_offset": offset,
            "n_sequences": n_sequences,
            "sequence_length": ctx,
            "intact_ce_nats": round(intact_ce, 6),
            "intact_perplexity": round(intact_ppl, 4),
            "note": "the intact baseline is measured on this slice in this run, "
                    "so every delta is internal to the experiment and is not "
                    "comparable to the published 2.5152 validation loss",
        },
        "9a_head_ablation": {
            "per_head": rows,
            "delta_ce_mean": round(mean_d, 6),
            "delta_ce_sd": round(sd_d, 6),
            "null_single_head_mean_delta_ce": round(null_mean, 6),
            "null_single_head_sd_delta_ce": round(null_sd, 6),
            "null_sets": len(null_rows),
            "null_by_size": {str(s): round(sum(n["delta_ce"] for n in null_rows
                                               if n["size"] == s)
                                          / max(sum(1 for n in null_rows
                                                    if n["size"] == s), 1), 6)
                             for s in (1, 2, 3)},
            "top_heads": [{k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in r.items()} for r in ranked[:8]],
            "n_heads_z_ge_2": sum(1 for r in rows if r["z_score"] >= 2.0),
            "n_heads_z_le_0": sum(1 for r in rows if r["z_score"] <= 0.0),
            "top4_combination_delta_ce": round(top_delta, 6),
            "random_4head_mean_delta_ce": round(rand_mean, 6),
            "top4_beats_random4": bool(top_delta > rand_mean),
            "top4_bootstrap95": bootstrap_mean_ci(
                [r["delta_ce"] for r in top], seed=1),
            "entropy_of_z_ge_2_heads": entropy_hi,
            "entropy_of_z_le_0_heads": entropy_lo,
            "important_heads_are": ("low-entropy (sharp) heads" if
                                    (entropy_hi is not None and entropy_lo is not None
                                     and entropy_hi < entropy_lo) else
                                    "high-entropy (diffuse) heads" if
                                    (entropy_hi is not None and entropy_lo is not None
                                     and entropy_hi > entropy_lo) else
                                    "indistinguishable"),
        },
        "9b_rollout": {
            "n_items_with_ground_truth": n_ground_truth,
            "rollout_hit_rate": round(sum(rollout_hits) / len(rollout_hits), 4)
            if rollout_hits else None,
            "rollout_hits_wilson95": [round(v, 4) for v in
                                      wilson_interval(sum(rollout_hits),
                                                      len(rollout_hits))]
            if rollout_hits else None,
            "raw_attention_hit_rate": round(sum(raw_hits) / len(raw_hits), 4)
            if raw_hits else None,
            "raw_attention_hits_wilson95": [round(v, 4) for v in
                                            wilson_interval(sum(raw_hits),
                                                            len(raw_hits))]
            if raw_hits else None,
            "uniform_chance": round(1.0 / ctx, 4),
            "ground_truth": "the token position holding the number the passage "
                            "asserts, located in the oracle prompt",
        },
        "9c_logit_lens": {
            "n_items": len(lens_rows),
            "median_rank_of_asserted_number_per_layer": lens_median,
            "vocabulary_size": 6487,
            "n_items_argmax_is_the_number_at_some_layer": sum(
                1 for r in lens_rows if r["argmax_layer"] is not None),
            "note": "the lens is the model's own readout (final norm plus the "
                    "tied unembedding) applied to each layer's residual, so "
                    "'decodable' means decodable by a readout the model already "
                    "owns",
        },
        "wall_seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", default="6.3M")
    ap.add_argument("--sequences", type=int, default=64)
    ap.add_argument("--rollout-items", type=int, default=40)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="results/v2/act9/interpretability.json")
    args = ap.parse_args()

    out = run(args.rung, n_sequences=args.sequences,
              n_rollout_items=args.rollout_items, device=args.device)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    a = out["9a_head_ablation"]
    print(f"intact CE {out['held_out']['intact_ce_nats']} ppl "
          f"{out['held_out']['intact_perplexity']}")
    print(f"heads with z>=2: {a['n_heads_z_ge_2']}/{out['model']['num_ablated_heads']}")
    print(f"top heads: {[(h['layer'], h['head'], h['z_score']) for h in a['top_heads'][:5]]}")
    print(f"top4 delta {a['top4_combination_delta_ce']} vs random4 "
          f"{a['random_4head_mean_delta_ce']} -> {a['top4_beats_random4']}")
    print(f"rollout hit {out['9b_rollout']['rollout_hit_rate']} vs raw "
          f"{out['9b_rollout']['raw_attention_hit_rate']} vs chance "
          f"{out['9b_rollout']['uniform_chance']}")
    print(f"lens median rank per layer: {out['9c_logit_lens']['median_rank_of_asserted_number_per_layer']}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
