"""Act 9 controls: the checks that decide whether the Act 9 claims survive.

9c  LOGIT-LENS CONTROLS. The first measurement reported the median rank of the
    ASSERTED number and found it most decodable at layer 5 (rank 87 of 6,487).
    But Act 8 already established that the model has a generic context-NUMBER
    bias, so "a number is decodable" is not the claim being made. The controls
    decide which claim survives:

      (i)   the asserted number, oracle condition
      (ii)  the counterfactual-EDITED number, counterfactual condition
      (iii) a number that appears in NO passage, oracle condition
      (iv)  the asserted number, non-numeric control condition
      (v)   a random token id, for scale

    DECISION RULE, fixed before running:
      * if (ii) rises to a similar rank to (i), the lens detects "a number in
        context" and nothing more, and the claim must be REWORDED to "context
        numbers are represented, but assertion-specific content is not";
      * if (ii) stays near chance while (i) is near 87, the claim survives and
        strengthens: the model encodes WHICH number is asserted, then destroys
        it before the output.

9b  ROLLOUT CONTROLS. The same three conditions for attention-based
    attribution, plus where the attribution actually points. A hit rate of 0/34
    against a chance rate of 0.0078 is what chance predicts, so the negative is
    about the METHOD unless it is anchored against the model's behaviour on a
    passage containing no number.

9a  ATTENTION-SINK CHECKS. One head out of 64 at z = +7.82, in layer 0, with
    attention entropy 0.379 nats against 2.480 for the rest, is the classic
    signature of an attention sink rather than a task-specific head. Three
    checks: what the head actually attends to; whether ablating it collapses
    the output toward uniform (which it does not -- uniform is 8.78 nats); and
    whether a backup head covers for it. Plus a second held-out slice and a
    second null seed so the z-score carries an error bar.
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
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v9" / "act9" / "controls.json"


# --------------------------------------------------------------------------
# helpers reused from the main Act 9 module
# --------------------------------------------------------------------------

def _attention_modules(model):
    return [block.attention for block in model.blocks]


def _find_subsequence(haystack, needle):
    if not needle:
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def _capture_attentions(model, ids, *, device):
    for attn in _attention_modules(model):
        attn.record_attention = True
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        model(x)
    mats = []
    for attn in _attention_modules(model):
        mats.append(attn.last_attn.float()[0].mean(dim=0))
    for attn in _attention_modules(model):
        attn.record_attention = False
        attn.last_attn = None
    return mats


def _capture_residuals(model, ids, *, device):
    captured = []
    handles = []

    def hook(_m, _i, out):
        captured.append(out.detach())

    for block in model.blocks:
        handles.append(block.register_forward_hook(hook))
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return captured


def _rank_of(scores, target):
    return int((scores > scores[target]).sum())


def _median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# --------------------------------------------------------------------------
# 9c controls
# --------------------------------------------------------------------------

def lens_controls(model, tokenizer, items, *, device) -> dict:
    """Median rank per layer for five targets, across four prompt conditions."""
    from src.eval.act8_hallucination import assemble

    rng = random.Random(SEED)
    vocab = 6487
    # a random token id that is not the start of a number
    random_token = rng.randrange(100, vocab)

    buckets = {k: [] for k in ("asserted_oracle", "edited_counterfactual",
                               "absent_number_oracle", "asserted_nonnumeric",
                               "random_token_oracle")}
    argmax_positions = {"rollout": [], "raw": []}
    mass_on_target = {"rollout": [], "raw": []}

    for item in items:
        prompts = {
            "oracle": assemble("oracle", item, "", tokenizer=tokenizer,
                               context_length=model.config.context_length)[0],
            "counterfactual": assemble("counterfactual", item, "", tokenizer=tokenizer,
                                       context_length=model.config.context_length)[0],
            "nonnumeric": assemble("nonnumeric", item, "", tokenizer=tokenizer,
                                   context_length=model.config.context_length)[0],
        }
        ids = {k: tokenizer.encode(v) for k, v in prompts.items()}
        src_ids = tokenizer.encode(item["source_number"])
        cf_ids = tokenizer.encode(item["counterfactual_number"])
        # a number that appears in NO passage: a digit token absent from the text
        absent = next(t for t in range(30, 40)
                      if tokenizer.encode(str(t - 30)) not in ids["oracle"]
                      and t - 30 != 0)
        absent_ids = tokenizer.encode(str(absent))

        pos_src = _find_subsequence(ids["oracle"], src_ids)
        pos_cf = _find_subsequence(ids["counterfactual"], cf_ids)

        for cond, target_ids, pos in (
            ("asserted_oracle", src_ids, pos_src),
            ("edited_counterfactual", cf_ids, pos_cf),
            ("absent_number_oracle", absent_ids, -1),
            ("asserted_nonnumeric", src_ids, -1),
            ("random_token_oracle", [random_token], -1),
        ):
            key = "counterfactual" if cond == "edited_counterfactual" else (
                "nonnumeric" if cond == "asserted_nonnumeric" else "oracle")
            residuals = _capture_residuals(model, ids[key], device=device)
            lens = [model.lm_head(model.final_norm(r)) for r in residuals]
            ranks = [_rank_of(l[0, -1].float(), target_ids[0]) for l in lens]
            buckets[cond].append(ranks)

            if cond in ("asserted_oracle", "edited_counterfactual") and pos >= 0:
                mats = _capture_attentions(model, ids[key], device=device)
                from src.eval.act9_interpretability import rollout_matrices
                roll = rollout_matrices(mats)[-1][-1]
                raw = mats[-1][-1]
                argmax_positions["rollout"].append(int(roll.argmax()))
                argmax_positions["raw"].append(int(raw.argmax()))
                mass_on_target["rollout"].append(float(roll[pos]))
                mass_on_target["raw"].append(float(raw[pos]))

    out = {}
    for cond, rows in buckets.items():
        per_layer = []
        for layer in range(len(rows[0])):
            per_layer.append(_median([r[layer] for r in rows]))
        out[cond] = {"median_rank_per_layer": per_layer,
                     "n_items": len(rows),
                     "best_layer": min(range(len(per_layer)),
                                        key=lambda i: per_layer[i]),
                     "best_median_rank": min(per_layer)}
    out["_random_token_id"] = random_token
    out["_argmax_positions"] = {
        "rollout": argmax_positions["rollout"],
        "raw": argmax_positions["raw"],
    }
    out["_mass_on_target"] = mass_on_target
    return out


# --------------------------------------------------------------------------
# 9a sink checks
# --------------------------------------------------------------------------

def sink_checks(model, sequences, *, device, top_heads) -> dict:
    """What does the top head attend to, and is it redundant?"""
    from src.eval.act9_interpretability import (measure_ce, _set_ablations)

    layer, head = top_heads[0]
    attn = _attention_modules(model)[layer]

    # mean attention distribution of that head over positions
    attn.record_attention = True
    sums = []
    with torch.no_grad():
        for start in range(0, len(sequences), 8):
            x = torch.tensor(sequences[start:start + 8], dtype=torch.long,
                             device=device)
            model(x)
            a = attn.last_attn.float()  # [B, H, T_query, T_key]
            h = a[:, head]
            # attention mass paid BY this head TO each key position, averaged
            # over query positions and then over the batch -> [T_key]
            sums.append(h.mean(dim=1).mean(dim=0).cpu())
    attn.record_attention = False
    attn.last_attn = None
    dist = torch.stack(sums).mean(dim=0)
    total = float(dist.sum())
    dist = dist / total
    uniform = 1.0 / len(dist)
    pos0 = float(dist[0])
    last = float(dist[-1])
    top5 = torch.topk(dist, 5)
    # concentration: fraction of mass on the single most-attended position
    concentration = float(top5.values[0])

    intact_ce = measure_ce(model, sequences, device=device)
    _set_ablations(model, {layer: {head}})
    ablated_ce = measure_ce(model, sequences, device=device)
    _set_ablations(model, {})
    uniform_ce = math.log(model.config.vocabulary_size)

    # redundancy: pairwise ablation on the top heads
    singles = {}
    for (l, h) in top_heads:
        _set_ablations(model, {l: {h}})
        singles[(l, h)] = measure_ce(model, sequences, device=device) - intact_ce
    _set_ablations(model, {})
    pairs = []
    for i in range(len(top_heads)):
        for j in range(i + 1, len(top_heads)):
            (l1, h1), (l2, h2) = top_heads[i], top_heads[j]
            _set_ablations(model, {l1: {h1}, l2: {h2}})
            combined = measure_ce(model, sequences, device=device) - intact_ce
            _set_ablations(model, {})
            additive = singles[(l1, h1)] + singles[(l2, h2)]
            pairs.append({
                "heads": [[l1, h1], [l2, h2]],
                "combined_delta_ce": round(combined, 6),
                "sum_of_singles_delta_ce": round(additive, 6),
                "redundancy": round(1.0 - combined / additive, 4) if additive else None,
            })

    return {
        "head": {"layer": layer, "head": head},
        "attention_distribution": {
            "position_0_mass": round(pos0, 5),
            "last_position_mass": round(last, 5),
            "uniform_mass": round(uniform, 5),
            "top5_positions": [int(i) for i in top5.indices.tolist()],
            "top5_mass": [round(float(v), 5) for v in top5.values.tolist()],
            "single_position_concentration": round(concentration, 5),
            "verdict": ("attention SINK: mass concentrates on one position" if
                        concentration > 0.3 else
                        "not a single-position sink"),
        },
        "output_collapse": {
            "intact_ce_nats": round(intact_ce, 6),
            "ablated_ce_nats": round(ablated_ce, 6),
            "uniform_ce_nats": round(uniform_ce, 6),
            "verdict": ("the ablation DEGRADES the model, it does not collapse it "
                        "toward uniform" if ablated_ce < uniform_ce - 1.0 else
                        "the ablation collapses the output toward uniform"),
        },
        "pairwise_redundancy": pairs,
        "singles_delta_ce": {f"{l},{h}": round(v, 6)
                             for (l, h), v in singles.items()},
    }


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def run(rung: str, *, n_sequences: int, n_items: int, device: str,
        slices=(300_000, 320_000)) -> dict:
    from src.eval.act7_sft import load_canonical
    from src.eval.act8_hallucination import load_passages
    from src.eval.act9_interpretability import (build_heldout, head_ablation,
                                                random_ablation_null)
    from src.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer.load("checkpoints/canonical/tokenizer.json")
    model = load_canonical(rung).to(device)
    model.eval()
    ctx = model.config.context_length
    t0 = time.time()

    items = load_passages("data/v2/c4_en_shard_0.jsonl", limit=120,
                          tokenizer=tokenizer)[:n_items]
    lens = lens_controls(model, tokenizer, items, device=device)

    # 9a on TWO slices and TWO null seeds, so the z-score has an error bar.
    per_slice = []
    for offset in slices:
        sequences = build_heldout(tokenizer, offset=offset,
                                  n_sequences=n_sequences, context_length=ctx)
        rows, intact_ce, intact_ppl = head_ablation(model, sequences, device=device)
        rng = random.Random(SEED)
        null_rows = random_ablation_null(model, sequences, device=device,
                                         n_sets=40, rng=rng)
        null_single = [n["delta_ce"] for n in null_rows if n["size"] == 1]
        nm = sum(null_single) / len(null_single)
        nsd = math.sqrt(sum((d - nm) ** 2 for d in null_single)
                        / (len(null_single) - 1))
        for r in rows:
            r["z_vs_random_null"] = (r["delta_ce"] - nm) / nsd if nsd > 0 else 0.0
        top = sorted(rows, key=lambda r: r["delta_ce"], reverse=True)[:5]
        per_slice.append({
            "offset": offset,
            "intact_ce_nats": round(intact_ce, 6),
            "intact_perplexity": round(intact_ppl, 4),
            "null_mean": round(nm, 6), "null_sd": round(nsd, 6),
            "top_heads": [{"layer": r["layer"], "head": r["head"],
                           "delta_ce": round(r["delta_ce"], 6),
                           "z_vs_random_null": round(r["z_vs_random_null"], 4)}
                          for r in top],
            "z_of_layer0_head5": next(
                (round(r["z_vs_random_null"], 4) for r in rows
                 if r["layer"] == 0 and r["head"] == 5), None),
            "n_heads_z_ge_2": sum(1 for r in rows if r["z_vs_random_null"] >= 2),
        })

    sequences = build_heldout(tokenizer, offset=slices[0], n_sequences=n_sequences,
                              context_length=ctx)
    rows, _, _ = head_ablation(model, sequences, device=device)
    top_heads = [(r["layer"], r["head"])
                 for r in sorted(rows, key=lambda r: r["delta_ce"], reverse=True)[:5]]
    sink = sink_checks(model, sequences, device=device, top_heads=top_heads)

    # 9b: where does the attribution actually point?
    pos = lens.pop("_argmax_positions")
    mass = lens.pop("_mass_on_target")
    rollout_args = {
        "rollout_argmax_position_is_last": sum(
            1 for p in pos["rollout"] if p == len(sequences[0]) - 1),
        "raw_argmax_position_is_last": sum(
            1 for p in pos["raw"] if p == len(sequences[0]) - 1),
        "n": len(pos["rollout"]),
        "rollout_mean_attribution_mass_on_the_number": (
            round(sum(mass["rollout"]) / len(mass["rollout"]), 6)
            if mass["rollout"] else None),
        "raw_mean_attribution_mass_on_the_number": (
            round(sum(mass["raw"]) / len(mass["raw"]), 6) if mass["raw"] else None),
        "note": "a hit rate of 0/34 against a chance rate of 0.0078 is what "
                "chance predicts, so this is a statement about the METHOD; the "
                "mass and argmax measurements say whether the attribution has "
                "any structure at all",
    }

    del model
    if device == "mps":
        torch.mps.empty_cache()

    return {
        "act": "9",
        "purpose": "the controls that decide whether the Act 9 claims survive",
        "9c_lens_controls": lens,
        "9b_attribution_controls": rollout_args,
        "9a_sink_checks": sink,
        "9a_two_slices": per_slice,
        "wall_seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", default="6.3M")
    ap.add_argument("--sequences", type=int, default=64)
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = run(args.rung, n_sequences=args.sequences, n_items=args.items,
              device=args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    lc = out["9c_lens_controls"]
    print("9c MEDIAN RANK OF THE TARGET PER LAYER")
    print(f"  {'condition':<28}" + "".join(f"L{i:<7}" for i in range(8)))
    for k in ("asserted_oracle", "edited_counterfactual", "absent_number_oracle",
              "asserted_nonnumeric", "random_token_oracle"):
        r = lc[k]["median_rank_per_layer"]
        print(f"  {k:<28}" + "".join(f"{v:<8.0f}" for v in r))
    print(f"\n  random token id used: {lc['_random_token_id']}")
    s = out["9a_sink_checks"]
    print(f"\n9a SINK CHECK on L{s['head']['layer']} H{s['head']['head']}")
    print(f"  attention: {s['attention_distribution']}")
    print(f"  collapse : {s['output_collapse']}")
    for p in out["9a_two_slices"]:
        print(f"  slice @{p['offset']}: intact CE {p['intact_ce_nats']}  "
              f"L0H5 z={p['z_of_layer0_head5']}  heads z>=2: {p['n_heads_z_ge_2']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
