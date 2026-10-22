"""Generate the source CSVs for every LaTeX table in the final report.

Reads only the tracked results under ``results/`` and writes one CSV per
table into ``report/tables/``.  A reader can re-derive every number in the
paper's tables from these files, and the files are the same numbers the
figures are drawn from, so the two cannot disagree.

Outputs
    act4_ladder_rungs.csv          4 the ladder, split by architecture family
    act4_fit_results.csv           4 the clean fit and the fit variants
    act4_contamination.csv         4 the withdrawn fit scored on each point set
    act5_kv_cache.csv                 5a latency and cache memory
    act5_mqa.csv                      5b paired MQA/MHA differences
    act5_speculative.csv              5c speedup and acceptance vs K
    act6_decoding_sweep.csv           6a the 15-configuration sweep
    act6_bestofn.csv                  6b best-of-n
    act6_selfconsistency.csv          6c majority voting on MNLI
    act7_sft_icl.csv                  7a/7b zero-shot floor, ICL, SFT curve
    act7_peft_matched_budget.csv      7c matched-budget PEFT head-to-head
    act7_lora_rank_sweep.csv          7d LoRA rank sweep
    act7_icl_control.csv              7b the 18-item same-task A/B control
    act8_choice_rates.csv             8b/8c the choice rates, the headline
    act8_logprob_contrasts.csv        8c the paired log-probability contrasts
    act8_tool_use.csv                 8d tool use
    act9_head_ablation.csv            9a the per-head causal ablation
    act9_logit_lens.csv               9b/9c rollout hit rate and lens ranks

Usage:
    python -m src.eval.report_tables [--results-dir DIR] [--tables-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "results"
DEFAULT_TABLES = REPO_ROOT / "report" / "tables"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing tracked result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(tables_dir: Path, name: str, header: list[str],
           rows: list[list]) -> Path:
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / name
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return out


# ── Act 4 ────────────────────────────────────────────────────────────────────

def _fold_stats(fit: dict) -> tuple[float, float, float, float]:
    """(mean, sample s.d., min, max) of the leave-one-rung-out alphas.

    RETIRED as a way to quote alpha: a standard deviation over four folds of
    a model whose alpha is not identified is false precision, and the paper
    quotes the fold-wise estimates INDIVIDUALLY and the profile likelihood
    alongside them.  The helper is kept for the record.
    """
    alphas = [float(v["alpha"]) for v in fit.get("loro", {}).values()
              if v.get("alpha") is not None]
    mean = sum(alphas) / len(alphas)
    if len(alphas) > 1:
        sd = (sum((a - mean) ** 2 for a in alphas) / (len(alphas) - 1)) ** 0.5
    else:
        sd = 0.0
    return mean, sd, min(alphas), max(alphas)


def act4_tables(results: Path, tables: Path) -> None:
    clean = _load(results / "v2" / "act4-scaling" / "clean_refit.json")
    tracked = _load(results / "v2" / "act4-scaling" / "fit_results.json")
    per_run = clean["provenance"]["per_run"]

    # The ladder, split by architecture family.  One row per (rung, family).
    grouped: dict[tuple[str, str], list[dict]] = {}
    for key, run in per_run.items():
        rung, seed = key.split("/")
        grouped.setdefault((rung, run["family"]), []).append(run)
    rows = []
    for (rung, family), runs in sorted(grouped.items(),
                                        key=lambda kv: (kv[0][0], kv[0][1])):
        runs = sorted(runs, key=lambda r: int(r["final_val_loss"]))
        rows.append([
            rung, family, runs[0]["model_class"],
            runs[0]["positional_encoding"],
            "; ".join(str(r["non_embedding_params"]) for r in runs),
            "; ".join(str(r["total_params"]) for r in runs),
            runs[0]["steps"], runs[0]["tokens"], len(runs),
            "; ".join(f"{r['final_val_loss']:.4f}" for r in runs),
            "results/v2/act4-scaling/clean_refit.json",
        ])
    _write(tables, "act4_ladder_rungs.csv",
           ["rung", "family", "model_class", "positional_encoding",
            "non_embedding_params_per_run", "total_params_per_run",
            "steps", "training_tokens",
            "n_runs", "final_validation_losses", "source"],
           rows)

    # The fit variants.  The paper's fit is the canonical family on the
    # TOTAL-parameter axis; the non-embedding variant of the same family is
    # reported beside it, because the DIFFERENCE between the two conventions
    # is the point of the act.  The fold-wise alphas are recorded
    # INDIVIDUALLY: a standard deviation over four folds of an unidentified
    # model is false precision, and it is not quoted.
    order = [
        ("canonical_only_D5M_total_N",
         clean["fits_under_the_total_parameter_convention"]
         ["canonical_only_D5M_total_N"], "total",
         "canonical_only_D5M_total_N"),
        ("canonical_only_D5M",
         clean["fits"]["canonical_only_D5M"], "non-embedding",
         "canonical_only_D5M_non_embedding_N"),
        ("other_family_only_D5M",
         clean["fits"]["other_family_only_D5M"], "non-embedding", None),
        ("as_tracked_12_runs_D5M",
         clean["fits"]["as_tracked_12_runs_D5M"], "non-embedding", None),
    ]
    prof = clean["profile_likelihood_over_alpha"]
    rows = []
    for key, fit, convention, pkey in order:
        loro = fit.get("loro", {})
        pr = prof.get(pkey) if pkey else None
        rng = (f"[{pr['compatible_alpha_min']:.2f}, "
               f"{pr['compatible_alpha_max']:.2f}]"
               if pr else "n/a (no profile recorded for this point set)")
        contains = ("yes" if pr["contains_0_34"] else "no") if pr else "n/a"
        rows.append([
            key, convention, fit["n_points"], fit["n_rungs"],
            len(fit["distinct_N"]),
            "; ".join(str(n) for n in fit["distinct_N"]),
            f"{fit['E']:.4f}", f"{fit['A']:.4f}", f"{fit['beta']:.4f}",
            f"{fit['alpha']:.4f}", f"{fit['r2_vs_constant_baseline']:.4f}",
            "; ".join(f"{r}: {v['alpha']:.4f}"
                      for r, v in sorted(loro.items())
                      if v.get("alpha") is not None),
            len(loro),
            "; ".join(f"{r}: R2={v['r2']:.4f}" for r, v in sorted(loro.items())
                      if v.get("r2") is not None),
            rng, contains,
            "results/v2/act4-scaling/clean_refit.json",
        ])
    _write(tables, "act4_fit_results.csv",
           ["fit", "n_axis_convention", "n_points", "n_rungs", "n_distinct_N",
            "distinct_N", "E", "A", "beta", "alpha_point_estimate",
            "r2_vs_constant_baseline", "fold_alpha_by_rung", "n_folds",
            "loro_r2_by_rung", "profile_alpha_compatible_range",
            "profile_contains_chinchilla_0_34", "source"],
           rows)

    # The profile likelihood over alpha AND over beta: the PRIMARY statement
    # of the act.  The beta profile is what makes the asymmetry explicit --
    # beta is two to three times better constrained than alpha, because each
    # run contributes interior checkpoints at FIXED N and RISING D, so
    # within-run D variation constrains beta, while alpha depends on the
    # NUMBER AND SPREAD of distinct N values the ladder holds.
    bprof = clean["profile_likelihood_over_beta"]
    rows = []
    for key, label in (
            ("canonical_only_D5M_total_N",
             "canonical family, TOTAL-parameter N (the paper's axis)"),
            ("canonical_only_D5M_non_embedding_N",
             "canonical family, non-embedding N")):
        p = prof[key]
        rows.append([
            "alpha", label, p["n_points"], p["delta_r2"], f"{p['best_r2']:.6f}",
            f"{p['best_alpha']:.2f}", f"{p['compatible_alpha_min']:.2f}",
            f"{p['compatible_alpha_max']:.2f}",
            f"{p['compatible_alpha_max'] - p['compatible_alpha_min']:.2f}",
            "0.34", "yes" if p["contains_0_34"] else "no",
            "results/v2/act4-scaling/clean_refit.json",
        ])
    for key, label in (
            ("canonical_only_D5M_total_N",
             "canonical family, TOTAL-parameter N (the paper's axis)"),
            ("canonical_only_D5M_non_embedding_N",
             "canonical family, non-embedding N")):
        p = bprof[key]
        rows.append([
            "beta", label, p["n_points"], p["delta_r2"], f"{p['best_r2']:.6f}",
            f"{p['best_beta']:.2f}", f"{p['compatible_beta_min']:.2f}",
            f"{p['compatible_beta_max']:.2f}",
            f"{p['compatible_beta_max'] - p['compatible_beta_min']:.2f}",
            "0.28", "yes" if p["contains_0_28"] else "no",
            "results/v2/act4-scaling/clean_refit.json",
        ])
    _write(tables, "act4_profile_likelihood.csv",
           ["parameter", "convention", "n_points", "delta_r2_tolerance",
            "best_r2", "best_value", "compatible_min", "compatible_max",
            "compatible_width", "published_value", "contains_published_value",
            "source"],
           rows)

    # The measured unigram entropy of the corpus under the ladder's own
    # 6,487-token vocabulary, against the two fitted E values.  This is what
    # falsified the "physically possible E" argument.
    ent = _load(results / "v2" / "act4-scaling" / "unigram_entropy.json")
    fits = ent["fits_compared"]
    rows = []
    for region in ("held_out_region", "training_region_sample",
                   "full_corpus"):
        r = ent[region]
        rows.append([
            region, r["tokens"], f"{r['entropy_nats']:.4f}",
            "results/v2/act4-scaling/unigram_entropy.json",
        ])
    for convention, key in (("total_N", "total_N"),
                             ("non_embedding_N", "non_embedding_N")):
        f = fits[key]
        rows.append([
            f"fitted E ({convention})", "n/a", f"{f['E']:.3f}",
            "results/v2/act4-scaling/unigram_entropy.json",
        ])
    for convention, key in (("total_N", "total_N"),
                             ("non_embedding_N", "non_embedding_N")):
        f = fits[key]
        rows.append([
            f"ratio of fitted E to the held-out entropy ({convention})",
            "n/a", f"{f['ratio_to_held_out_entropy']:.4f}",
            "results/v2/act4-scaling/unigram_entropy.json",
        ])
    _write(tables, "act4_unigram_entropy.csv",
           ["quantity", "tokens", "entropy_nats", "source"], rows)

    # The contamination diagnostic: the WITHDRAWN tracked fit's own parameters
    # evaluated on every candidate point set.
    on = clean["tracked_params_on_candidate_sets"]
    rows = []
    for key, label in (("as_tracked_12_runs_D5M",
                        "as tracked (D >= 5M, 12 runs)"),
                       ("other_family_only_D5M", "other family only (D >= 5M)"),
                       ("canonical_only_D5M", "canonical family only (D >= 5M)")):
        rows.append([
            label, on[key]["n"], f"{on[key]['r2']:+.6f}",
            f"{tracked['secondary']['alpha']:.4f}",
            f"{tracked['secondary']['beta']:.4f}",
            f"{tracked['secondary']['E']:.4f}",
            f"{tracked['secondary']['r2']:.6f}",
            "results/v2/act4-scaling/clean_refit.json; "
            "results/v2/act4-scaling/fit_results.json",
        ])
    _write(tables, "act4_contamination.csv",
           ["candidate point set", "n_points", "r2_of_the_withdrawn_fit",
            "withdrawn_alpha", "withdrawn_beta", "withdrawn_E",
            "r2_the_withdrawn_fit_claimed", "source"],
           rows)

    # The per-rung N AUDIT: every rung's N against three independent sources
    # (the ladder CSV's column, the run's own recorded metric, and the
    # architecture the run implies), plus the LADDER design target.
    #
    # The first two columns are machine-derived here, from the tracked ladder
    # points and from the tracked run records.  The third and fourth are
    # quoted from the project ledger (docs/implementation.md, section "Act 4,
    # part 3", item 4), which is where the audit itself was recorded; they are
    # NOT in a result JSON, and the caveat column of the ledger row says so.
    audit_quoted = {
        # rung: (architecture-implied N, LADDER design target, D/N)
        "0.8M": ("809,728 (MiniGPT) / 793,344 (other family) -- NEITHER",
                 "786,432", "22.7"),
        "1.5M": ("1,567,200 (canonical) / 1,546,720 (other family)",
                 "1,536,000", "19.6 / 19.9"),
        "2.6M": ("2,694,144", "2,654,208", "19.7"),
        "4.1M": ("4,235,616", "4,128,768", "19.5"),
        "6.3M": ("6,351,360", "6,291,456", "19.8"),
        "9.0M": ("8,992,224", "8,957,952", "10.0"),
    }
    # N recorded by the run itself, per rung, from the tracked run records.
    metric_by_rung: dict[str, set[int]] = {}
    for key, run in per_run.items():
        rung = key.split("/")[0]
        metric_by_rung.setdefault(rung, set()).add(
            int(run["non_embedding_params"]))
    # N as held in the ladder CSV's own column, per rung.
    csv_by_rung: dict[str, set[int]] = {}
    with (results / "v2" / "act4-scaling" / "ladder_points.csv").open(
            encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            csv_by_rung.setdefault(r["rung"], set()).add(
                int(r["non_embedding_params"]))
    rows = []
    for rung in ("0.8M", "1.5M", "2.6M", "4.1M", "6.3M", "9.0M"):
        csv_n = sorted(csv_by_rung.get(rung, []))
        metric_n = sorted(metric_by_rung.get(rung, []))
        arch, target, dn = audit_quoted[rung]
        csv_metric_agree = ("yes" if csv_n == metric_n and len(csv_n) == 1
                            else "NO -- the rung carries more than one N")
        all_three = ("no" if rung == "0.8M"
                     else "no -- the rung carries two families"
                     if rung == "1.5M" else "yes")
        rows.append([
            rung,
            "; ".join(f"{n:,}" for n in csv_n),
            "; ".join(f"{n:,}" for n in metric_n),
            csv_metric_agree,
            arch, target, dn, all_three,
            "results/v2/act4-scaling/ladder_points.csv; "
            "results/v2/act4-scaling/clean_refit.json; "
            "docs/implementation.md",
        ])
    _write(tables, "act4_n_audit.csv",
           ["rung", "N_in_the_ladder_csv", "N_in_the_run_recorded_metric",
            "csv_and_metric_agree", "N_implied_by_the_architecture",
            "LADDER_design_target", "D_over_N", "all_three_sources_agree",
            "source"],
           rows)

    # The revised rescue plan.  The arms and their costs are quoted from the
    # project ledger (docs/implementation.md, section "Act 4, part 3", item 5),
    # which is where the spending decision is recorded; no rescue result is
    # reported anywhere in the paper, because the rescue is in flight.
    rescue_plan = [
        ["0.3M @ D=20N", "~3 min",
         "+1 distinct N at the very bottom: extends log N from "
         "ln(9.0/0.8) = 2.42 to ln(9.0/0.3) = 3.40, a 41% increase in N range "
         "for minutes of GPU"],
        ["0.4M @ D=20N", "~4 min",
         "a further small N, the same leverage argument"],
        ["0.8M @ D=20N", "~1 min",
         "replaces the unidentifiable third architecture at the BOTTOM of the "
         "N range, where a power law has the most leverage, and fixes the "
         "worst hold-out fold"],
        ["4.1M @ D=20N", "92 min",
         "+1 distinct N, the whole 4.1M rung replaced with a canonical one, "
         "and a matched-(N, D) comparison against the discarded 4.1M family "
         "(same N, same D, same data, different position table)"],
        ["9.0M @ D=10N", "~120 min",
         "half the cost of 20N, a NOVEL D/N ratio so the design is no longer "
         "a single line, and a DIRECT matched-(N, D) comparison against the "
         "discarded 9.0M family, which sat at exactly 9.0M / D=10N: the "
         "contamination becomes an experiment"],
        ["9.0M @ D=20N", "~240 min",
         "a second D ratio at the top rung, constraining the within-run shape "
         "as well as the level; kept because it was already half trained"],
    ]
    _write(tables, "act4_rescue_plan.csv",
           ["arm", "cost", "what it buys", "source"],
           [[a, c, w, "docs/implementation.md"] for a, c, w in rescue_plan])


# ── Act 9 ────────────────────────────────────────────────────────────────────

def act9_tables(results: Path, tables: Path) -> None:
    d = _load(results / "v2" / "act9" / "interpretability.json")
    a = d["9a_head_ablation"]

    rows = []
    for r in a["per_head"]:
        rows.append([
            r["layer"], r["head"], f"{r['delta_ce']:.6f}",
            f"{r['delta_ppl']:.6f}", f"{r['ablated_ce']:.6f}",
            f"{r['ablated_ppl']:.6f}", f"{r['z_score']:.4f}",
            f"{r['z_vs_random_null']:.4f}",
            f"{r['mean_entropy_nats']:.4f}",
            "yes" if r["z_score"] >= 2 else ("no" if r["z_score"] <= 0
                                             else "between 0 and 2"),
        ])
    _write(tables, "act9_head_ablation.csv",
           ["layer", "head", "delta_ce_nats", "delta_ppl", "ablated_ce",
            "ablated_ppl", "z_score", "z_vs_random_null",
            "mean_attention_entropy_nats", "z_band"], rows)

    b = d["9b_rollout"]
    _write(tables, "act9_rollout.csv",
           ["method", "hits", "n_items", "hit_rate", "wilson95_low",
            "wilson95_high"],
           [
               ["attention rollout (all 8 layers)",
                int(round(b["rollout_hit_rate"] * b["n_items_with_ground_truth"])),
                b["n_items_with_ground_truth"], b["rollout_hit_rate"],
                f"{b['rollout_hits_wilson95'][0]:.4f}",
                f"{b['rollout_hits_wilson95'][1]:.4f}"],
               ["raw attention (last layer)",
                int(round(b["raw_attention_hit_rate"]
                          * b["n_items_with_ground_truth"])),
                b["n_items_with_ground_truth"], b["raw_attention_hit_rate"],
                f"{b['raw_attention_hits_wilson95'][0]:.4f}",
                f"{b['raw_attention_hits_wilson95'][1]:.4f}"],
               ["uniform chance (1/128)", "", b["n_items_with_ground_truth"],
                b["uniform_chance"], "", ""],
           ])

    c = d["9c_logit_lens"]
    rows = []
    for layer, rank in enumerate(c["median_rank_of_asserted_number_per_layer"]):
        rows.append([layer, f"{rank:.1f}", c["vocabulary_size"],
                     c["n_items"],
                     c["n_items_argmax_is_the_number_at_some_layer"]])
    _write(tables, "act9_logit_lens.csv",
           ["layer", "median_rank_of_the_asserted_number", "vocabulary_size",
            "n_items", "n_items_argmax_is_the_number_at_some_layer"], rows)

    act9_control_tables(results, tables)


def act9_control_tables(results: Path, tables: Path) -> None:
    """The Act 9 CONTROL tables.

    These are the measurements that decide whether the Act 9 claims survive:
    the logit-lens controls (is the lens number-specific or
    assertion-specific?), the attribution controls (is the negative about the
    method?), and the sink / redundancy / two-slice checks on the head claim.
    """
    ctl = _load(results / "v2" / "act9" / "controls.json")
    lens = ctl["9c_lens_controls"]

    labels = {
        "asserted_oracle": "(i) asserted number, oracle",
        "edited_counterfactual": "(ii) counterfactual-EDITED number, "
                                 "counterfactual",
        "absent_number_oracle": "(iii) a number appearing in NO passage, "
                                "oracle",
        "asserted_nonnumeric": "(iv) asserted number, non-numeric control",
        "random_token_oracle": "(v) random token id",
    }
    rows = []
    for key in ("asserted_oracle", "edited_counterfactual",
                "absent_number_oracle", "asserted_nonnumeric",
                "random_token_oracle"):
        rec = lens[key]
        rows.append([
            labels[key], rec["n_items"],
            "; ".join(f"{v:.1f}" for v in rec["median_rank_per_layer"]),
            rec["best_layer"], f"{rec['best_median_rank']:.1f}",
            "results/v2/act9/controls.json",
        ])
    _write(tables, "act9_9c_controls.csv",
           ["condition", "n_items",
            "median_rank_per_layer_L0_to_L7", "best_layer",
            "best_median_rank", "source"], rows)

    att = ctl["9b_attribution_controls"]
    _write(tables, "act9_9b_attribution.csv",
           ["quantity", "value", "n", "uniform_or_reference", "source"],
           [
               ["attention rollout: mean attribution mass on the answer token",
                att["rollout_mean_attribution_mass_on_the_number"], att["n"],
                0.00781, "results/v2/act9/controls.json"],
               ["raw last-layer attention: mean attribution mass on the answer "
                "token",
                att["raw_mean_attribution_mass_on_the_number"], att["n"],
                0.00781, "results/v2/act9/controls.json"],
               ["attention rollout: argmax lands on the final position",
                f"{att['rollout_argmax_position_is_last']}/{att['n']}",
                att["n"], "0/71 is what a non-collapsing attribution predicts",
                "results/v2/act9/controls.json"],
               ["raw attention: argmax lands on the final position",
                f"{att['raw_argmax_position_is_last']}/{att['n']}",
                att["n"], "0/71 is what a non-collapsing attribution predicts",
                "results/v2/act9/controls.json"],
               ["uniform mass over a 128-token context",
                0.00781, att["n"], "1/128", "results/v2/act9/controls.json"],
           ])

    sink = ctl["9a_sink_checks"]
    dist = sink["attention_distribution"]
    coll = sink["output_collapse"]
    src = "results/v2/act9/controls.json"
    rows = [
        ["mean attention mass on position 0 (L0 H5)",
         f"{dist['position_0_mass']:.5f}",
         f"uniform {dist['uniform_mass']:.5f}",
         "mildly position-0 biased, not a single-position sink", src],
        ["mass on the top five positions",
         "; ".join(f"{v:.5f}" for v in dist["top5_mass"]),
         f"uniform {dist['uniform_mass']:.5f}",
         "nearly flat: the head is not a sink", src],
        ["single-position concentration",
         f"{dist['single_position_concentration']:.5f}", "n/a",
         dist["verdict"], src],
        ["held-out CE, intact vs L0 H5 ablated",
         f"{coll['intact_ce_nats']:.6f} -> {coll['ablated_ce_nats']:.6f}",
         f"uniform-vocabulary CE {coll['uniform_ce_nats']:.6f}",
         coll["verdict"], src],
    ]
    for pair in sink["pairwise_redundancy"]:
        if pair["heads"] not in ([[0, 5], [0, 2]], [[0, 5], [0, 0]],
                                 [[0, 5], [1, 6]]):
            continue
        a, b = pair["heads"]
        rows.append([
            f"pairwise ablation L{a[0]} H{a[1]} with L{b[0]} H{b[1]}",
            f"combined {pair['combined_delta_ce']:+.6f}",
            f"sum of singles {pair['sum_of_singles_delta_ce']:+.6f}",
            f"redundancy {pair['redundancy']:+.4f} (1 = the two heads are "
            f"interchangeable)", src,
        ])
    for sl in ctl["9a_two_slices"]:
        rows.append([
            f"two-slice check: C4 offset {sl['offset']}",
            f"L0 H5 z = {sl['z_of_layer0_head5']:+.4f}",
            f"intact CE {sl['intact_ce_nats']:.6f} "
            f"(perplexity {sl['intact_perplexity']:.4f})",
            f"{sl['n_heads_z_ge_2']} heads at z >= 2; null mean "
            f"{sl['null_mean']:.6f}, s.d. {sl['null_sd']:.6f}", src,
        ])
    _write(tables, "act9_9a_controls.csv",
           ["check", "value", "reference", "reading", "source"], rows)


# ── Act 5 ────────────────────────────────────────────────────────────────────

def act5_tables(results: Path, tables: Path) -> None:
    kv = _load(results / "v2" / "act5" / "kv_cache.json")
    mqa = _load(results / "v2" / "act5" / "mqa_quality_2.6m.json")
    _write(tables, "act5_canonical_models.csv",
           ["rung", "non_embedding_params", "steps", "training_tokens",
            "validation_loss", "source"],
           [
               ["1.5M", mqa["canonical_1.5M_pin"]["non_embedding_params"],
                mqa["canonical_1.5M_pin"]["step"], 30720000,
                mqa["canonical_1.5M_pin"]["val_loss"],
                "results/v2/act5/mqa_quality_2.6m.json"],
               ["2.6M", 2654208, 12960, 53084160, 2.8111,
                "docs/implementation.md (canonical rung table)"],
               ["6.3M", 6291456, 30720, 125829120, 2.5152,
                "docs/implementation.md (canonical rung table)"],
           ])

    _write(tables, "act5_kv_cache.csv",
           ["context_tokens", "uncached_ms", "cached_ms", "speedup",
            "n_repeats_note"],
           [[r["context"], f"{r['uncached']:.1f}", f"{r['cached']:.1f}",
             f"{r['speedup']:.2f}", "microbenchmark on a purpose-built model"]
            for r in kv["latency_ms"]])

    mem = kv["cache_memory_mb"]
    _write(tables, "act5_kv_cache_memory.csv",
           ["model", "T_128_MB", "T_256_MB", "T_512_MB", "T_1024_MB"],
           [[r["model"], r["128"], r["256"], r["512"], r["1024"]] for r in mem])

    mqa = _load(results / "v2" / "act5" / "mqa_quality_2.6m.json")
    rows = []
    for seed, r in sorted(mqa["runs"].items(), key=lambda kv: int(kv[0])):
        rows.append([seed, r["mqa_g1_val_loss"], r["mha_g8_val_loss"],
                     r["diff_mqa_minus_mha"],
                     "G=1 arm resumed in 0.73 s; provenance inferred, "
                     "not re-verified" if seed == "42" else ""])
    _write(tables, "act5_mqa.csv",
           ["seed", "mqa_g1_val_loss", "mha_g8_val_loss",
            "diff_mqa_minus_mha", "provenance_note"], rows)

    av = _load(results / "v2" / "act5" / "attention_variants.json")
    v1_curve = av["quality_v1_path"]["val_loss_by_G"]
    _write(tables, "act5_mqa_context_only.csv",
           ["G", "v1_path_val_loss", "v1_path_cache_mb_at_T1024",
            "status"],
           [[g, v1_curve[g],
             next((s["cache_mb_at_T1024"] for s in av["num_kv_heads_sweep"]
                   if str(s["G"]) == g), ""),
             "CONTEXT ONLY: v1 TinyShakespeare path, 500 steps, character "
             "tokenizer, single seed"]
            for g in ("1", "2", "4", "8")])

    s17 = _load(results / "v2" / "act5" / "speculative_1.7x_multiprompt.json")
    s42 = _load(results / "v2" / "act5" / "speculative_4.2x_multiprompt.json")
    rows = []
    for K in ("1", "2", "3", "5", "7"):
        a, b = s17["per_K"][K], s42["per_K"][K]
        rows.append([K, a["median_speedup"], a["median_acceptance"],
                     b["median_speedup"], b["median_acceptance"],
                     s17["n_prompts"], s17["max_new_tokens"]])
    _write(tables, "act5_speculative.csv",
           ["K", "median_speedup_1.7x_gap", "median_acceptance_1.7x_gap",
            "median_speedup_4.2x_gap", "median_acceptance_4.2x_gap",
            "n_prompts", "max_new_tokens"], rows)


# ── Act 6 ────────────────────────────────────────────────────────────────────

def act6_tables(results: Path, tables: Path) -> None:
    d = _load(results / "v2" / "act6" / "6.3M_decoding.json")
    _write(tables, "act6_decoding_sweep.csv",
           ["config", "bits_per_char_reference", "bpc_sd",
            "repetition_rate_4gram", "distinct2", "seconds", "n_generations"],
           [[r["config"], r["bits_per_char_reference"], r["bpc_sd"],
             r["repetition_rate_4gram"], r["distinct2"], r["seconds"],
             r["n_generations"]] for r in d["decoding_sweep"]])

    _write(tables, "act6_bestofn.csv",
           ["n", "bpc_of_reranked_pick", "bpc_mean_of_all_samples",
            "improvement_bits", "n_generations"],
           [[r["n"], r["bpc_of_reranked_pick"], r["bpc_mean_of_all_samples"],
             r["improvement_bits"], r["n_generations"]]
            for r in d["best_of_n"]])

    sc = d["self_consistency_mnli"]
    _write(tables, "act6_selfconsistency.csv",
           ["n_votes", "accuracy", "wilson_95_low", "wilson_95_high",
            "correct", "total", "chance_level"],
           [[r["n_votes"], r["accuracy"], r["wilson_95_low"],
             r["wilson_95_high"], r["correct"], r["total"],
             f"{sc['chance_level']:.4f}"] for r in sc["results"]])


# ── Act 7 ────────────────────────────────────────────────────────────────────

def act7_tables(results: Path, tables: Path) -> None:
    d = _load(results / "v2" / "act7" / "6.3M_sft_peft.json")

    rows = []
    for r in d["icl_vs_sft"]:
        runs = r["accuracy_runs"]
        distinct = len(set(runs))
        rows.append([
            r["n_train"], r["method"], r["accuracy_mean"],
            "; ".join(f"{v:.4f}" for v in runs),
            distinct,
            "n=1 in substance: one batch per epoch, so the seeded shuffle "
            "has nothing to shuffle" if distinct == 1 else "n=3",
            r["trainable_parameters"],
        ])
    _write(tables, "act7_sft_icl.csv",
           ["n_train", "method", "accuracy_mean", "per_seed_runs",
            "distinct_seed_values", "effective_n", "trainable_parameters"],
           rows)

    head = sorted(d["peft_head_to_head"], key=lambda x: -x["accuracy_mean"])
    _write(tables, "act7_peft_matched_budget.csv",
           ["method", "size", "trainable_parameters", "budget",
            "final_train_loss", "accuracy_mean", "wilson_95_low",
            "wilson_95_high", "n_items"],
           [[h["method"], h["size"], h["trainable_parameters"], h["budget"],
             h["final_train_loss"], h["accuracy_mean"], h["wilson_95_low"],
             h["wilson_95_high"], 128] for h in head])

    _write(tables, "act7_lora_rank_sweep.csv",
           ["rank", "trainable_parameters", "accuracy_mean", "per_seed_runs"],
           [[s["rank"], s["trainable_parameters"], s["accuracy_mean"],
             "; ".join(f"{v:.4f}" for v in s["accuracy_runs"])]
            for s in sorted(d["lora_rank_sweep"], key=lambda x: x["rank"])])

    ctl = d["icl_control"]
    _write(tables, "act7_icl_control.csv",
           ["measurement", "value", "n", "interval"],
           [
               ["with demonstration: first generated token is the "
                "demonstration's answer", ctl[
                    "first_token_is_demonstration_answer_with_demo"],
                ctl["with_demo"]["n"], "exact count"],
               ["zero-shot: first generated token is the demonstration's answer",
                ctl["first_token_is_demonstration_answer_zero_shot"],
                ctl["zero_shot"]["n"], "exact count"],
               ["with demonstration: exact match on the query's answer",
                ctl["with_demo"]["hits"], ctl["with_demo"]["n"],
                f"Wilson 95% [{ctl['with_demo']['wilson95'][0]:.4f}, "
                f"{ctl['with_demo']['wilson95'][1]:.4f}]"],
               ["zero-shot: exact match on the query's answer",
                ctl["zero_shot"]["hits"], ctl["zero_shot"]["n"],
                f"Wilson 95% [{ctl['zero_shot']['wilson95'][0]:.4f}, "
                f"{ctl['zero_shot']['wilson95'][1]:.4f}]"],
               ["items whose answer changed because of the demonstration",
                ctl["items_changed_by_demonstration"], ctl["with_demo"]["n"],
                "exact count"],
               ["zero-shot generations that were only punctuation",
                ctl["zero_shot_degenerate_punctuation"], ctl["zero_shot"]["n"],
                "exact count"],
           ])


# ── Act 8 ────────────────────────────────────────────────────────────────────

def act8_tables(results: Path, tables: Path) -> None:
    d = _load(results / "v2" / "act8" / "hallucination.json")
    head = d["headline"]["choice_rates"]
    copy = d["conditions"]["copy_instrument"]
    chance = d["chance_baseline"]

    rows = []
    for cond in ("oracle", "closed_book", "retrieved", "counterfactual",
                 "nonnumeric"):
        c = copy[cond]
        key = ("counterfactual_chooses_the_EDITED_number" if cond ==
               "counterfactual" else f"{cond}_chooses_the_asserted_number")
        cr = head.get(key, {})
        adj = chance["per_condition"].get(cond, {})
        rows.append([
            cond, c["n_items"], c["counts"]["source"], c["rates"]["source"],
            f"[{c['chose_source_wilson95'][0]:.4f}, "
            f"{c['chose_source_wilson95'][1]:.4f}]",
            f"{chance['mean_uniform_chance_per_item']:.4f}",
            f"{adj.get('chance_adjusted_mean', float('nan')):.4f}",
            f"[{adj.get('bootstrap95', {}).get('ci_low', float('nan')):.4f}, "
            f"{adj.get('bootstrap95', {}).get('ci_high', float('nan')):.4f}]",
            "counts the ORIGINAL number, which is the asserted number; the "
            "edited number is the separate row below"
            if cond == "counterfactual" else "",
        ])
    # The counterfactual condition's edited-number choice is a separate
    # statistic with no chance-adjusted value, so it gets its own row.
    c = copy["counterfactual"]
    edited = head["counterfactual_chooses_the_EDITED_number"]
    tracks = d["contrasts"]["counterfactual_tracks_the_edit"]
    rows.append([
        "counterfactual_edited_number", c["n_items"], edited["count"],
        edited["rate"],
        f"[{edited['wilson95'][0]:.4f}, {edited['wilson95'][1]:.4f}]",
        "n/a", "n/a",
        f"[{tracks['bootstrap95']['ci_low']:.4f}, "
        f"{tracks['bootstrap95']['ci_high']:.4f}]",
        "the share of items whose preferred answer is the EDITED number; the "
        "bootstrap here is over the rate, not over a chance-adjusted difference",
    ])
    _write(tables, "act8_choice_rates.csv",
           ["condition", "n_items", "count", "rate", "wilson95",
            "mean_per_item_chance", "chance_adjusted_mean",
            "chance_adjusted_bootstrap95", "note"], rows)

    retr = d["retrieval"]
    _write(tables, "act8_retrieval.csv",
           ["retrieval metric", "before the fix", "after the fix", "note"],
           [
               ["distinct documents returned", 1, retr[
                   "distinct_documents_returned"],
                "1 is degenerate: the same chunk came back for every item"],
               ["document hit rate", "n/a", retr["document_hit_rate"],
                "the anchor is derived from the target document, so this is "
                "optimistic relative to a user's paraphrase"],
               ["retrieved chunk contains the asserted number", "0/40",
                retr["answer_present_rate"], ""],
               ["chose the asserted number under 'retrieved'", "n/a",
                f"{copy['retrieved']['counts']['source']}/40",
                f"chance-adjusted "
                f"{chance['per_condition']['retrieved']['chance_adjusted_mean']:.4f}, "
                f"bootstrap "
                f"[{chance['per_condition']['retrieved']['bootstrap95']['ci_low']:.4f}, "
                f"{chance['per_condition']['retrieved']['bootstrap95']['ci_high']:.4f}]"],
               ["followed the retrieved number when the chunk contained one",
                "n/a", copy["retrieved"]["followed_context_rate"],
                f"Wilson 95% "
                f"[{copy['retrieved']['followed_context_wilson95'][0]:.4f}, "
                f"{copy['retrieved']['followed_context_wilson95'][1]:.4f}]"],
           ])

    _write(tables, "act8_logprob_contrasts.csv",
           ["statistic", "n", "mean_nats", "bootstrap95_low", "bootstrap95_high",
            "p_value", "significant_at_0.05"],
           [[k, v["n"], f"{v['mean']:.4f}",
             f"{v['bootstrap95']['ci_low']:.4f}",
             f"{v['bootstrap95']['ci_high']:.4f}",
             f"{v.get('p_value', float('nan')):.4f}",
             "yes" if (v.get("p_value") is not None and
                       v["p_value"] < 0.05) else "no"]
            for k, v in d["logprob_contrasts"].items()])

    tool = _load(results / "v2" / "act8" / "tool_use.json")
    g, s = tool["greedy"], tool["sampled"]
    _write(tables, "act8_tool_use.csv",
           ["rate", "greedy_count", "greedy_n", "greedy_rate",
            "greedy_wilson95", "sampled_count", "sampled_n", "sampled_rate",
            "sampled_wilson95"],
           [
               ["well-formed CALC(...) call", g["well_formed_calls"],
                g["n_generations"], g["well_formed_rate"],
                f"[{g['well_formed_wilson95'][0]:.4f}, "
                f"{g['well_formed_wilson95'][1]:.4f}]",
                s["well_formed_calls"], s["n_generations"],
                s["well_formed_rate"],
                f"[{s['well_formed_wilson95'][0]:.4f}, "
                f"{s['well_formed_wilson95'][1]:.4f}]"],
               ["echoes the call template", g["echoes_call_template"],
                g["n_generations"], g["echoes_call_template_rate"], "",
                s["echoes_call_template"], s["n_generations"],
                s["echoes_call_template_rate"], ""],
               ["malformed call", g["malformed_call"], g["n_generations"],
                g["malformed_call_rate"], "",
                s["malformed_call"], s["n_generations"],
                s["malformed_call_rate"], ""],
               ["direct answer without a tool, and correct",
                g["direct_correct"], g["n_generations"],
                g["direct_correct_rate"], "",
                s["direct_correct"], s["n_generations"],
                s["direct_correct_rate"], ""],
           ])


def do_not_assert_table(results: Path, tables: Path) -> None:
    """The published list of claims the report does NOT make.

    The list is derived from the caveat fields that the Acts 7 and 8 result
    files carry, so a reader can trace each line back to the tracked record
    that forced it.  The prose table in the paper is the same content.
    """
    a7 = _load(results / "v2" / "act7" / "6.3M_sft_peft.json")
    a8 = _load(results / "v2" / "act8" / "hallucination.json")
    tool = _load(results / "v2" / "act8" / "tool_use.json")

    rows = [
        ["That the clean alpha is a point estimate",
         "The profile likelihood is the statement: the compatible range is "
         "[0.08, 0.54] on the total-parameter axis, and the point estimate "
         "+0.209 is not an interval",
         "4", "results/v2/act4-scaling/clean_refit.json"],
        ["That the fitted E is a measured irreducible loss",
         "It is an EXTRAPOLATED asymptote from a four-rung ladder, not the "
         "corpus's floor. The corpus's measured unigram entropy under the "
         "6,487-token vocabulary is 4.9946 nats on the held-out region, and "
         "BOTH fitted E values lie below it (1.348 is 0.27x it, 0.706 is "
         "0.14x it)",
         "4", "results/v2/act4-scaling/unigram_entropy.json"],
        ["That the ladder is CONSISTENT with Chinchilla's exponent",
         "It is compatible with published exponents ACROSS THEIR ENTIRE "
         "RANGE, and therefore discriminates between none of them: the "
         "total-N alpha range [0.08, 0.54] contains 0.34 and also 0.50 and "
         "most of the plausible range, so it excludes nothing that matters",
         "4", "results/v2/act4-scaling/clean_refit.json"],
        ["That beta is consistent with Chinchilla",
         "No compatibility claim may be made for beta at all: the total-N "
         "beta range is [0.36, 0.56] and EXCLUDES Chinchilla's 0.28 (the "
         "non-embedding range [0.34, 0.48] excludes it too)",
         "4", "results/v2/act4-scaling/clean_refit.json"],
        ["That our alpha is 0.34, or that it agrees with 0.34",
         "The compatible range is far too wide to identify 0.34, the point "
         "estimate is +0.209, and the fold-wise estimates run from +0.118 to "
         "+0.348",
         "4", "results/v2/act4-scaling/clean_refit.json"],
        ["That the profile's compatible range is an error bar",
         "It is a statement about what the data cannot distinguish, not a "
         "confidence interval and not a fit covariance; the range's width is "
         "itself the finding of the act",
         "4", "results/v2/act4-scaling/clean_refit.json"],
        ["That the non-embedding and total-parameter axes give the same answer",
         "They do not: the non-embedding axis's compatible range is [0.08, "
         "0.26] and does not contain 0.34, and its point estimate (+0.098) "
         "sits far from its own fold mean (+0.190)",
         "4", "results/v2/act4-scaling/clean_refit.json"],
        ["That the 0.8M rung is one of the two known architectures",
         "Its recorded N of 694,144 matches NO MiniGPT or ALiBi shape at any "
         "(dim, layers, heads) in dims 16-256 and depths 1-10, while its "
         "embedding parameters (830,336 = 6,487 x 128) show the vocabulary "
         "and dim are as expected: it is a THIRD, unidentifiable architecture "
         "with no architecture block, run on CPU, at D/N = 22.7",
         "4", "docs/implementation.md"],
        ["That the ladder's training code is in the repository",
         "It is not. src/scaling/ was EMPTY before commit f1eaf77; the "
         "committed ladder script builds the SECOND family's architecture "
         "(ALiBi, 4 heads) with the FIRST family's shapes, and the script "
         "that trained the canonical MiniGPT rungs was never committed at "
         "all",
         "4", "docs/implementation.md"],
        ["That the other family's runs were broken or divergent",
         "They were good models with an unlucky initialisation: they reproduce "
         "from the repository's own code (133.25 / 184.49 / 224.99 against the "
         "recorded 129.57 / 183.96 / 223.77) and they finished at 3.578 nats "
         "against the canonical 3.057",
         "4", "docs/implementation.md"],
        ["That the refit is a minor correction",
         "It invalidated alpha = 0.445, the 'R^2 >= 0.985 on every rung' "
         "headline, and the comparison with Chinchilla, because the other "
         "family owned the entire top of the ladder",
         "4", "results/v2/act4-scaling/clean_refit.json"],
        ["That the fold standard deviation was an uncertainty on alpha",
         "A standard deviation over four folds of a model whose alpha is not "
         "identified is false precision, and the point estimate sat far from "
         "the fold mean; the fold-wise estimates are quoted individually "
         "instead",
         "4", "results/v2/act4-scaling/clean_refit.json"],
        ["That the rescue has been run",
         "It is in flight and unfinished: the Act 4 numbers in this paper are "
         "the current four-rung clean fit, and no rescue number is reported",
         "4", "src/scaling/rescue.py"],
        ["That the calibration result means the model can follow instructions",
         "It measures next-token prediction on the training distribution; "
         "every instruction, grounding and tool task in Acts 7 and 8 is still "
         "at the floor",
         "10c", "results/v2/act10/ethics.json"],
        ["That in-context learning is absent at 6.3M",
         "The demonstration is copied, not ignored: its answer is the first "
         "generated token on 18/18 same-task items",
         "7b", "results/v2/act7/6.3M_sft_peft.json"],
        ["That the adapters-vs-LoRA ordering is established",
         "Wilson intervals on 128 items overlap by about +/-0.04",
         "7c", "results/v2/act7/6.3M_sft_peft.json"],
        ["That prefix tuning 'scored zero because of a bug'",
         "It did not fit: its final training loss is 3.2576",
         "7c", "results/v2/act7/6.3M_sft_peft.json"],
        ["That the n = 16 row of the SFT curve is an n = 3 result",
         "One batch per epoch, so the seeded shuffle has nothing to shuffle",
         "7b", "results/v2/act7/6.3M_sft_peft.json"],
        ["That the three ICL seeds are independent",
         "The model is frozen and the forward is deterministic, so the runs "
         "are bit-identical",
         "7b", "results/v2/act7/6.3M_sft_peft.json"],
        ["That the SFT results transfer to natural instructions",
         "The task family is synthetic, not natural instructions",
         "7", "results/v2/act7/6.3M_sft_peft.json"],
        ["That the ICL numbers are comparable to free-running generation "
         "numbers",
         "ICL is scored teacher-forced",
         "7b", "results/v2/act7/6.3M_sft_peft.json"],
        ["That the LoRA rank curve has plateaued",
         "It rises monotonically to r = 64 with no plateau",
         "7d", "results/v2/act7/6.3M_sft_peft.json"],
        ["That LoRA at rank 1 is a fair matched-budget comparison",
         "Rank is pinned low by construction, and 7d shows that point is not "
         "where the curve flattens",
         "7c, 7d", "results/v2/act7/6.3M_sft_peft.json"],
        ["That LIMA is contradicted",
         "We expose LIMA's premise (latent capability) as unsupported at this "
         "scale; the curve is learning a skill, not aligning one",
         "7", "results/v2/act7/6.3M_sft_peft.json"],
        ["That the 6.3M model can ground a passage",
         "It identifies the asserted number at chance, and a non-numeric "
         "passage scores the same 9/40",
         "8b", "results/v2/act8/hallucination.json"],
        ["That hallucination rates can be compared between conditions",
         "The absolute rate is a ceiling of 1.000 in every condition",
         "8b", "results/v2/act8/hallucination.json"],
        ["That the model learned an operation from a tool-call format",
         "It emits CALC(s): the format is acquired, the operation is absent",
         "8d", "results/v2/act8/tool_use.json"],
        ["That any result here measures factual truth",
         "Numbers proxy for atomic claims and there is no fact base, so this "
         "is groundedness",
         "8b", "results/v2/act8/hallucination.json"],
        ["That the tool-call template echo means the model learned the "
         "operation",
         "0 of 40 well-formed CALC(...) calls, and in all 19 greedy template "
         "echoes the argument is exactly 's'; the format is acquired and the "
         "operation is absent",
         "8d, 9c", "results/v2/act8/tool_use.json; "
         "results/v2/act9/interpretability.json"],
        ["That the head-ablation deltas are comparable to the published "
         "validation loss",
         "The intact baseline (CE 2.8551, perplexity 17.376) was measured on "
         "this 64-sequence slice in this run, not the published 2.5152; every "
         "delta is internal to the experiment",
         "9a", "results/v2/act9/interpretability.json"],
        ["That head ablation deletes the head",
         "It removes the head's CONTRIBUTION (its attention output is zeroed "
         "before it is mixed back into the residual stream); the head still "
         "computes",
         "9a", "results/v2/act9/interpretability.json"],
        ["That the rollout negative is a general claim about attention "
         "interpretability",
         "It is one attribution method on one prompt family (the Act 8 "
         "number-assertion family): 0 of 34 items, against a uniform chance "
         "of 0.0078",
         "9b", "results/v2/act9/interpretability.json"],
        ["That layer 0 head 5 is the head that carries the model",
         "A backup head covers for it: ablating L0 H5 together with L0 H2 "
         "costs only +0.0234 nats, the same as ablating L0 H2 alone "
         "(redundancy 0.97), and L0 H5 with L0 H0 likewise +0.0226. The "
         "effect is carried by a GROUP of layer-0 heads; L0 H5 is the one "
         "whose removal is most visible in isolation",
         "9a", "results/v2/act9/controls.json"],
        ["That the logit lens shows the model encodes WHICH number is "
         "asserted",
         "The counterfactual-EDITED number reaches the same layer-5 median "
         "rank as the asserted number (82 against 82) and the non-numeric "
         "control reaches 86, so the lens is number-specific and not "
         "assertion-specific; the strong version of this claim is withdrawn",
         "9c", "results/v2/act9/controls.json"],
    ]
    _write(tables, "do_not_assert_claims.csv",
           ["claim we do not make", "because", "act", "source"], rows)

    # the caveat fields that forced each line, for traceability
    caveat_rows = []
    for i, c in enumerate(a7.get("caveats", [])):
        caveat_rows.append(["act7", f"caveat_{i + 1}", c])
    caveat_rows.append(["act8", "claim_format",
                         a8["claim_format"]["measures"]])
    caveat_rows.append(["act8", "chance_baseline",
                         a8["chance_baseline"]["why_per_item"]])
    caveat_rows.append(["act8", "unit_note", a8["unit_note"]])
    caveat_rows.append(["act8", "tool_use_caveat",
                         tool["template_completion_evidence"]["caveat"]])
    _write(tables, "act7_act8_caveats.csv",
           ["act", "caveat_id", "caveat_text"], caveat_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    args = ap.parse_args()
    act4_tables(args.results_dir, args.tables_dir)
    act5_tables(args.results_dir, args.tables_dir)
    act6_tables(args.results_dir, args.tables_dir)
    act7_tables(args.results_dir, args.tables_dir)
    act8_tables(args.results_dir, args.tables_dir)
    act9_tables(args.results_dir, args.tables_dir)
    do_not_assert_table(args.results_dir, args.tables_dir)


if __name__ == "__main__":
    main()
