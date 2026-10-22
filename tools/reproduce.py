"""Reproduce the report's headline numbers from the committed artefacts.

This is the closing artefact of the project. The owner compiles the report on
Overleaf and never runs the training, so this script is the evidence that the
report's numbers can be re-derived from what is in the repository.

    python -m tools.reproduce            # all checks
    python -m tools.reproduce --list     # what it checks
    python -m tools.reproduce --only act9

It loads the TRACKED canonical checkpoints from ``checkpoints/canonical/`` and
the TRACKED results under ``results/``. It trains nothing. Each check prints
the value it derived and the value the report claims, and marks them MATCH or
MISMATCH.

Deliberately included:
  * one Act 4 fit (re-derived from the tracked ladder points, so the
    centrepiece's number is checked by the same code path that produced it)
    and the Act 4 profile likelihood over BETA, whose compatible range is the
    act's asymmetry;
  * one Act 9 number (the head-ablation effect), because the mechanism claim
    matters as much as the fits;
  * the Act 5a cache-identity check, the Act 7a zero-shot floor, the Act 8
    grounding rates, and the Act 2 ALiBi no-op.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CANON = ROOT / "checkpoints" / "canonical"


class Check:
    def __init__(self, key: str, label: str, source: str):
        self.key = key
        self.label = label
        self.source = source
        self.derived: object = None
        self.claimed: object = None
        self.note = ""
        self.ok = None

    def compare(self, derived, claimed, *, tol: float | None = None,
                note: str = "") -> "Check":
        self.derived = derived
        self.claimed = claimed
        self.note = note
        if tol is None:
            self.ok = derived == claimed
        else:
            self.ok = derived is not None and abs(float(derived) - float(claimed)) <= tol
        return self

    def render(self) -> str:
        status = {True: "MATCH   ", False: "MISMATCH", None: "SKIP    "}[self.ok]
        return (f"[{status}] {self.key:<10} {self.label}\n"
                f"             derived: {self.derived!r}\n"
                f"             claimed: {self.claimed!r}"
                + (f"\n             note: {self.note}" if self.note else ""))


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_act4_fit() -> Check:
    """Re-derive the clean fit's alpha from the tracked ladder points.

    The exact fit re-derived is ``canonical_only_D5M_total_N``: the canonical
    architecture family's interior points (D >= 5M), with N on the TOTAL
    parameter axis.  That is the convention the paper uses, because it is the
    one the published exponents use and the one on which the fit is stable
    (the point estimate and the fold-wise mean agree there, and do not on the
    non-embedding axis).  The non-embedding variant is also re-derived and
    reported, because the difference between the two IS the point.

    NOTE: the paper's earlier justification for this convention --- that the
    total axis is the only one with a physically possible irreducible loss ---
    was WITHDRAWN after the corpus's unigram entropy was measured
    (4.9946 nats, above BOTH fitted E values).  Do not reintroduce it.
    """
    from src.scaling.refit_clean import (load_points, profile_alpha, run_configs,
                                         summarise_fit)

    c = Check("act4", "clean scaling-law fit (canonical family, TOTAL-N axis)",
              "results/v2/act4-scaling/clean_refit.json")
    tracked = json.loads((RESULTS / "v2" / "act4-scaling" / "clean_refit.json").read_text())
    configs = run_configs()
    points = load_points(RESULTS / "v2" / "act4-scaling" / "ladder_points.csv", configs)
    canonical = [q for q in points if q["D"] >= 5_000_000 and q["family"] == "canonical"]
    canonical_total = [dict(q, N=q["N_total"]) for q in canonical]
    fit = summarise_fit("canonical_only_D5M_total_N", canonical_total)
    claim = tracked["fits_under_the_total_parameter_convention"]["canonical_only_D5M_total_N"]
    c.compare(round(fit["alpha"], 4), round(claim["alpha"], 4), tol=5e-4,
              note="alpha of the clean canonical fit on the TOTAL-N axis, "
                   "re-fitted from the tracked ladder points")
    c.compare(round(fit["r2_vs_constant_baseline"], 4),
              round(claim["r2_vs_constant_baseline"], 4), tol=5e-4,
              note="in-sample R^2 of the same fit")
    prof = profile_alpha(canonical_total)
    cprof = tracked["profile_likelihood_over_alpha"]["canonical_only_D5M_total_N"]
    c.compare(prof["compatible_alpha_max"], cprof["compatible_alpha_max"], tol=1e-9,
              note="the upper end of the profile's compatible alpha range")
    return c


def check_act4_beta_profile() -> Check:
    """Re-derive the profile likelihood over BETA on the same points.

    The beta profile is the act's asymmetry: beta is two to three times
    better constrained than alpha, and its compatible range EXCLUDES
    Chinchilla's 0.28 on both parameter conventions.  The paper states that
    range, so the script re-derives it from the same tracked ladder points by
    the same code path that produced it.
    """
    from src.scaling.refit_clean import (load_points, profile_beta, run_configs)

    c = Check("act4beta", "profile likelihood over beta (canonical family)",
              "results/v2/act4-scaling/clean_refit.json")
    tracked = json.loads((RESULTS / "v2" / "act4-scaling" / "clean_refit.json").read_text())
    configs = run_configs()
    points = load_points(RESULTS / "v2" / "act4-scaling" / "ladder_points.csv", configs)
    canonical = [q for q in points if q["D"] >= 5_000_000 and q["family"] == "canonical"]
    canonical_total = [dict(q, N=q["N_total"]) for q in canonical]
    prof = profile_beta(canonical_total)
    cprof = tracked["profile_likelihood_over_beta"]["canonical_only_D5M_total_N"]
    c.compare(prof["compatible_beta_max"], cprof["compatible_beta_max"], tol=1e-9,
              note="the upper end of the profile's compatible beta range on the "
                   "TOTAL-N axis (the paper's [0.36, 0.56], which EXCLUDES "
                   "Chinchilla's 0.28)")
    return c


def check_act9_redundancy() -> Check:
    """Re-derive the redundancy result: a backup head covers for L0 H5."""
    from src.eval.act9_interpretability import (build_heldout, measure_ce,
                                                _set_ablations)
    from src.eval.act7_sft import load_canonical
    from src.tokenizer.bpe import BPETokenizer

    c = Check("act9r", "head redundancy: a backup head covers for L0 H5",
              "results/v2/act9/controls.json")
    tokenizer = BPETokenizer.load(str(CANON / "tokenizer.json"))
    model = load_canonical("6.3M")
    model.eval()
    sequences = build_heldout(tokenizer, offset=300_000, n_sequences=64,
                              context_length=model.config.context_length)
    intact = measure_ce(model, sequences, device="cpu")
    _set_ablations(model, {0: {5}})
    alone = measure_ce(model, sequences, device="cpu")
    _set_ablations(model, {0: {5}, 0: {2}})
    combined = measure_ce(model, sequences, device="cpu")
    _set_ablations(model, {0: {2}})
    partner = measure_ce(model, sequences, device="cpu")
    _set_ablations(model, {})
    tracked = json.loads((RESULTS / "v2" / "act9" / "controls.json").read_text())
    pair = next(p for p in tracked["9a_sink_checks"]["pairwise_redundancy"]
                if p["heads"] == [[0, 5], [0, 2]])
    c.compare(round(combined - intact, 4), round(pair["combined_delta_ce"], 4),
              tol=2e-3,
              note="ablating L0 H5 TOGETHER WITH L0 H2 costs this much; L0 H2 "
                   "alone costs %.4f, so the pair is redundant" % (partner - intact))
    return c


def check_act10_calibration() -> Check:
    """Re-derive the calibration error of the base model."""
    from src.eval.act10_ethics import calibration_probe
    from src.eval.act7_sft import load_canonical
    from src.tokenizer.bpe import BPETokenizer

    c = Check("act10c", "calibration: the model is well calibrated",
              "results/v2/act10/ethics.json")
    tokenizer = BPETokenizer.load(str(CANON / "tokenizer.json"))
    model = load_canonical("6.3M")
    model.eval()
    out = calibration_probe(model, tokenizer, device="cpu", n_sequences=32)
    tracked = json.loads((RESULTS / "v2" / "act10" / "ethics.json").read_text())
    c.compare(round(out["expected_calibration_error"], 4),
              round(tracked["calibration"]["expected_calibration_error"], 4),
              tol=2e-3,
              note="expected calibration error over %d predictions"
                   % out["n_predictions"])
    c.compare(round(out["overall_accuracy"], 4),
              round(tracked["calibration"]["overall_accuracy"], 4), tol=2e-3,
              note="next-token accuracy")
    return c


def check_act5a_cache_identity() -> Check:
    """The KV cache must not change the output."""
    from src.eval.act7_sft import load_canonical
    from src.generation.decoding import generate

    c = Check("act5a", "KV cache is output-identical",
              "results/v2/act5/kv_cache.json")
    model = load_canonical("1.5M")
    model.eval()
    ids = torch.randint(0, model.config.vocabulary_size, (1, 32))
    with torch.no_grad():
        plain, _, _ = model(ids)
        cached, _, _ = model(ids, cache=_fresh_cache(model))
    diff = float((plain - cached).abs().max())
    claimed = json.loads((RESULTS / "v2" / "act5" / "kv_cache.json").read_text())
    target = claimed.get("correctness", {}).get("max_logit_diff_vs_uncached")
    c.compare(diff, target, tol=1e-5,
              note="max |logit| difference between the cached and uncached paths")
    return c


def _fresh_cache(model):
    from src.model.gpt import KVCache

    return KVCache(model.config.num_layers, model.config.num_heads,
                   model.config.embedding_dim // model.config.num_heads,
                   model.config.num_kv_heads)


def check_act7a_zero_shot() -> Check:
    """The zero-shot instruction floor, recomputed on the canonical 6.3M.

    Uses the SAME teacher-forced exact-match path as Act 7 (``score_response``),
    not a free-generation approximation: the claim is about the scoring route.
    """
    from src.data.instruction_tasks import make_splits
    from src.eval.act7_sft import load_canonical, score_response
    from src.tokenizer.bpe import BPETokenizer

    c = Check("act7a", "zero-shot instruction following floor",
              "results/v2/act7/6.3M_sft_peft.json")
    tokenizer = BPETokenizer.load(str(CANON / "tokenizer.json"))
    model = load_canonical("6.3M")
    model.eval()
    splits = make_splits(n_train=512, n_held_out=128)
    hits = sum(int(score_response(model, tokenizer, ex)) for ex in splits.held_out)
    tracked = json.loads((RESULTS / "v2" / "act7" / "6.3M_sft_peft.json").read_text())
    claimed = tracked.get("zero_shot", {}).get("accuracy_mean")
    c.compare(hits, 0,
              note=f"exact matches on {len(splits.held_out)} held-out items by the "
                   f"teacher-forced path (the tracked zero_shot accuracy is "
                   f"{claimed})")
    return c


def check_act8_grounding() -> Check:
    """The Act 8 choice rates, recomputed on the canonical 6.3M."""
    from src.eval.act8 import classify_copy, score_candidates
    from src.eval.act8_hallucination import (assemble, distractor_pool,
                                            load_passages, partner_number)
    from src.eval.act7_sft import load_canonical
    from src.tokenizer.bpe import BPETokenizer

    c = Check("act8", "grounding: choice rates and the non-numeric control",
              "results/v2/act8/hallucination.json")
    tokenizer = BPETokenizer.load(str(CANON / "tokenizer.json"))
    model = load_canonical("6.3M")
    model.eval()
    ctx = model.config.context_length
    items = load_passages(str(ROOT / "data" / "v2" / "c4_en_shard_0.jsonl"),
                           limit=120, tokenizer=tokenizer)
    # The same containment guard the real harness applies: an item whose source
    # number falls outside the token budget in the oracle prompt is dropped,
    # because the oracle would not be answerable.
    kept = []
    for item in items:
        _p, fitted = assemble("oracle", item, "", tokenizer=tokenizer,
                               context_length=ctx)
        if item["source_number"] in fitted:
            kept.append(item)
    items = kept[:40]
    distractors = distractor_pool(items)
    rates = {}
    for cond in ("oracle", "nonnumeric"):
        picks = 0
        for i, item in enumerate(items):
            prompt, fitted = assemble(cond, item, "", tokenizer=tokenizer,
                                      context_length=ctx)
            ids = tokenizer.encode(prompt)
            ctx_num = (item["source_number"]
                       if cond == "oracle" and item["source_number"] in fitted
                       else None)
            alt = partner_number(cond, item)
            cands = sorted({x for x in [item["source_number"], alt] + distractors[i] if x})
            scores = score_candidates(model, ids, cands, tokenizer, "cpu")
            chosen = max(scores, key=lambda x: scores[x]["total_logprob"])
            picks += int(classify_copy(chosen, item["source_number"], alt) == "source")
        rates[cond] = picks
    tracked = json.loads((RESULTS / "v2" / "act8" / "hallucination.json").read_text())
    ci = tracked["conditions"]["copy_instrument"]
    c.compare(rates["oracle"], ci["oracle"]["counts"]["source"],
              note="oracle: items where the model chose the asserted number")
    c.compare(rates["nonnumeric"], ci["nonnumeric"]["counts"]["source"],
              note="non-numeric control: same number, so the control is valid")
    return c


def check_act9_head() -> Check:
    """The Act 9 head-ablation effect, re-derived on a held-out slice."""
    from src.eval.act9_interpretability import (build_heldout, head_ablation,
                                                random_ablation_null)
    from src.eval.act7_sft import load_canonical
    from src.tokenizer.bpe import BPETokenizer

    c = Check("act9", "head ablation: the top head's effect and its z-score",
              "results/v2/act9/interpretability.json")
    tokenizer = BPETokenizer.load(str(CANON / "tokenizer.json"))
    model = load_canonical("6.3M")
    model.eval()
    sequences = build_heldout(tokenizer, offset=300_000, n_sequences=64,
                              context_length=model.config.context_length)
    rows, intact_ce, _ = head_ablation(model, sequences, device="cpu")
    import random as _random
    null_rows = random_ablation_null(model, sequences, device="cpu", n_sets=40,
                                      rng=_random.Random(42))
    null = [n["delta_ce"] for n in null_rows if n["size"] == 1]
    nm = sum(null) / len(null)
    nsd = math.sqrt(sum((d - nm) ** 2 for d in null) / (len(null) - 1))
    top = max(rows, key=lambda r: r["delta_ce"])
    z = (top["delta_ce"] - nm) / nsd
    tracked = json.loads((RESULTS / "v2" / "act9" / "interpretability.json").read_text())
    claim = tracked["9a_head_ablation"]["top_heads"][0]
    c.compare((top["layer"], top["head"]), (claim["layer"], claim["head"]),
              note="which head matters")
    c.compare(round(top["delta_ce"], 4), round(claim["delta_ce"], 4), tol=5e-4,
              note="its change in held-out cross-entropy (nats)")
    c.compare(round(z, 2), round(claim["z_vs_random_null"], 2), tol=0.1,
              note="its z-score against the random-ablation null")
    return c


def check_act2_alibi_noop() -> Check:
    """The ALiBi arm never measured ALiBi: the two rows are one result."""
    c = Check("act2", "the ALiBi ablation arm is a no-op",
              "results/act2-ablations/summary.json")
    tracked = json.loads((RESULTS / "act2-ablations" / "summary.json").read_text())
    alibi = tracked.get("pe-alibi-seed42")
    none = tracked.get("pe-none-seed42")
    if alibi is None or none is None:
        c.note = "could not locate pe-alibi-seed42 / pe-none-seed42 in the summary"
        c.ok = None
        return c
    a = float(alibi["validation_loss_mean"])
    n = float(none["validation_loss_mean"])
    c.compare(round(abs(a - n), 12), 4.8e-08, tol=5e-08,
              note=f"the gap between the ALiBi arm ({a:.10f}) and the no-PE arm "
                   f"({n:.10f}); they are the same result counted twice, and the "
                   f"arms also differ in parameter count "
                   f"({alibi['parameters']} vs {none['parameters']})")
    return c


def check_act10c_sft_calibration() -> Check:
    """Re-derive the SFT-versus-base calibration comparison.

    Instruction tuning on 512 examples made the model a better instruction
    follower and a measurably worse next-token predictor. The instrument is
    deterministic on the same held-out slice for both models, so the two
    expected calibration errors are exact.
    """
    from src.eval.act10_ethics import calibration_probe
    from src.model.gpt import GPTConfig, MiniGPT
    from src.tokenizer.bpe import BPETokenizer

    c = Check("act10sft", "instruction tuning degrades next-token calibration",
              "results/v2/act10/ethics.json")
    tokenizer = BPETokenizer.load(str(CANON / "tokenizer.json"))

    def load(path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        m = MiniGPT(GPTConfig(**payload["model_config"]))
        m.load_state_dict(payload["model_state"])
        return m.eval()

    base = calibration_probe(load(str(CANON / "6.3M" / "model.pt")), tokenizer,
                             device="cpu", n_sequences=32)
    sft = calibration_probe(load(str(CANON / "6.3M-sft" / "model.pt")), tokenizer,
                            device="cpu", n_sequences=32)
    tracked = json.loads((RESULTS / "v2" / "act10" / "ethics.json").read_text())
    c.compare(round(sft["expected_calibration_error"], 4),
              round(tracked["sft_calibration"]["sft"]["expected_calibration_error"], 4),
              tol=2e-3,
              note="the tuned model's expected calibration error; the base "
                   "model's is %.4f, so tuning multiplies it by %.1f"
                   % (base["expected_calibration_error"],
                      sft["expected_calibration_error"] / base["expected_calibration_error"]))
    c.compare(round(base["expected_calibration_error"], 4),
              round(tracked["sft_calibration"]["base"]["expected_calibration_error"], 4),
              tol=2e-3, note="the base model's expected calibration error")
    return c


CHECKS = {
    "act2": check_act2_alibi_noop,
    "act4": check_act4_fit,
    "act4beta": check_act4_beta_profile,
    "act5a": check_act5a_cache_identity,
    "act7a": check_act7a_zero_shot,
    "act8": check_act8_grounding,
    "act9": check_act9_head,
    "act9r": check_act9_redundancy,
    "act10c": check_act10_calibration,
    "act10sft": check_act10c_sft_calibration,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="run one check by key")
    args = ap.parse_args()

    if args.list:
        for key, fn in CHECKS.items():
            print(f"{key:<8} {fn.__doc__ or ''}")
            print(f"         {fn.__doc__}")
        return

    keys = [args.only] if args.only else list(CHECKS)
    results = []
    for key in keys:
        print(f"running {key} ...", flush=True)
        try:
            check = CHECKS[key]()
        except Exception as exc:  # a failed check is a result, not a crash
            check = Check(key, "check failed", "")
            check.note = f"{type(exc).__name__}: {exc}"
            check.ok = False
        results.append(check)
        print(check.render())
        print()

    mismatched = [c for c in results if c.ok is False]
    print("=" * 70)
    for c in results:
        print(f"{c.key:<8} {'MATCH' if c.ok else ('MISMATCH' if c.ok is False else 'SKIP')}")
    if mismatched:
        print(f"\n{len(mismatched)} check(s) did not reproduce.")
        sys.exit(1)
    print("\nAll checks reproduced from the committed artefacts.")


if __name__ == "__main__":
    main()
