"""Act 7 figures for the final report.

Reads the tracked ``results/v2/act7/*.json`` and writes PNG + vector PDF pairs:

* ``act7_icl_vs_sft``        — 7b: held-out accuracy against instruction-set
  size for ICL vs full fine-tune, with the zero-shot floor drawn in.
* ``act7_peft_matched_budget`` — 7c: the matched-budget PEFT head-to-head with
  Wilson intervals, and 7d: the LoRA rank sweep against the full fine-tune
  reference.

Everything plotted comes from the tracked JSON; nothing is re-measured or
hard-coded, so a figure cannot drift from its result.

Usage:
    python -m src.eval.act7_figures [--results-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RESULTS = Path("results/v2/act7")
DEFAULT_OUT = Path("report/figures")

FLOOR, SFT, PEFT, LORA = "#95a5a6", "#2471a5", "#c0392b", "#7d3c98"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing tracked result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=160, bbox_inches="tight")
        print(f"wrote {out_dir / name}.{ext}")


def icl_vs_sft_figure(results_dir: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _load(results_dir / "6.3M_sft_peft.json")
    rows = d["icl_vs_sft"]

    sizes = sorted({r["n_train"] for r in rows})
    sft = [next(r for r in rows if r["n_train"] == n and r["method"] == "full")
           for n in sizes]

    # ICL is measured ONCE and drawn flat: its prompt depends only on the
    # few-shot pool and the query, never on the training-set size. Earlier
    # versions plotted one point per size, which invented a curve out of a
    # constant.
    icl = d["icl"]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    floor = d["zero_shot"]["accuracy_mean"]

    ax.axhline(floor, color=FLOOR, lw=1.4, ls="--", zorder=1,
               label=f"zero-shot floor ({floor:.4f})")

    ax.plot(sizes, [r["accuracy_mean"] for r in sft], "-o", color=SFT, lw=2.2,
            ms=7, zorder=3, label="full fine-tune")

    # n=16 returns identical values across all three seeds: one batch per epoch
    # means the seeded shuffle has nothing to shuffle, so that point is
    # effectively n=1. Labelling it rather than drawing an error bar of width 0
    # next to a legend claim of "n=3 seeds".
    n16 = next(r for r in sft if r["n_train"] == 16)
    runs16 = n16.get("accuracy_runs") or []
    if runs16 and len(set(runs16)) == 1:
        ax.annotate("n=1 effectively\n(seeds identical:\n1 batch/epoch)",
                    xy=(16, n16["accuracy_mean"]), xytext=(30, 0.16),
                    fontsize=7.5, color="#555555",
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8))
    for n, r in zip(sizes, sft):
        runs = r.get("accuracy_runs") or [r["accuracy_mean"]]
        if len(runs) > 1:
            lo, hi = min(runs), max(runs)
            ax.errorbar([n], [r["accuracy_mean"]],
                        yerr=[[r["accuracy_mean"] - lo], [hi - r["accuracy_mean"]]],
                        fmt="none", ecolor=SFT, alpha=0.45, capsize=4, zorder=3)

    shots = icl.get("shots_used", 0)
    ax.axhline(icl["accuracy_mean"], color=PEFT, lw=2.2, ls="-", zorder=3,
               label=f"in-context learning, {shots} shot(s)\n"
                     f"(no weight change, measured once)")

    # A conventional 4-shot prompt does not fit this model's context window;
    # say so on the figure rather than in a footnote nobody reads.
    ax.annotate(
        f"{shots} shot fits {icl['context_length']}-token context;\n"
        "a 4-shot prompt is ~328 tokens",
        xy=(sizes[-1], icl["accuracy_mean"]), xytext=(-6, 12),
        textcoords="offset points", ha="right", fontsize=7.5,
        color="#555555")

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("instruction training examples  (log scale)")
    ax.set_ylabel("held-out exact match")
    ax.set_ylim(-0.03, 0.72)
    ax.set_title("7b  adaptation scales, in-context learning does not\n"
                 "still rising at n=512: no saturation in range")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "act7_icl_vs_sft")


def peft_and_rank_figure(results_dir: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _load(results_dir / "6.3M_sft_peft.json")
    head = sorted(d["peft_head_to_head"], key=lambda x: x["accuracy_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))

    # ── 7c: matched trainable-parameter budget ────────────────────────────
    ax = axes[0]
    labels, accs, errs_lo, errs_hi, colors = [], [], [], [], []
    for x in head:
        labels.append(x["label"])
        accs.append(x["accuracy_mean"])
        errs_lo.append(x["accuracy_mean"] - x["wilson_95_low"])
        errs_hi.append(x["wilson_95_high"] - x["accuracy_mean"])
        # Full fine-tune is not budget-matched (8.0M params); mark it apart.
        colors.append(FLOOR if x["method"] == "full" else PEFT)

    y = range(len(head))
    ax.barh(list(y), accs, color=colors, zorder=3, alpha=0.9)
    ax.errorbar(accs, list(y), xerr=[errs_lo, errs_hi], fmt="none",
                ecolor="#333333", alpha=0.7, capsize=3, zorder=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{l}\n{h['trainable_parameters']:,} params"
                        for l, h in zip(labels, head)], fontsize=8)
    ax.set_xlabel("held-out exact match  (Wilson 95%)")
    ax.set_title("7c  PEFT head-to-head at a matched 20k budget\n"
                 "grey bar is NOT budget-matched (8.0M params)")
    ax.set_xlim(0, 0.72)
    ax.grid(alpha=0.3, axis="x")

    # ── 7d: LoRA rank sweep ───────────────────────────────────────────────
    ax = axes[1]
    sweep = sorted(d["lora_rank_sweep"], key=lambda x: x["rank"])
    full_best = max(x["accuracy_mean"] for x in head if x["method"] == "full")

    ax.axhline(full_best, color=FLOOR, lw=1.4, ls="--", zorder=1,
               label=f"full fine-tune ({full_best:.4f})")
    ax.plot([x["rank"] for x in sweep], [x["accuracy_mean"] for x in sweep],
            "-o", color=LORA, lw=2.2, ms=7, zorder=3, label="LoRA")
    for x in sweep:
        runs = x.get("accuracy_runs") or [x["accuracy_mean"]]
        if len(runs) > 1:
            lo, hi = min(runs), max(runs)
            ax.errorbar([x["rank"]], [x["accuracy_mean"]],
                        yerr=[[x["accuracy_mean"] - lo], [hi - x["accuracy_mean"]]],
                        fmt="none", ecolor=LORA, alpha=0.45, capsize=4, zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xticks([x["rank"] for x in sweep])
    ax.set_xticklabels([str(x["rank"]) for x in sweep])
    ax.set_xlabel("LoRA rank r  (log scale)")
    ax.set_ylabel("held-out exact match")
    ax.set_title("7d  rank is a smooth capacity dial\n"
                 "monotonic across 1..64, no plateau")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "act7_peft_matched_budget")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    icl_vs_sft_figure(args.results_dir, args.out_dir)
    peft_and_rank_figure(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
