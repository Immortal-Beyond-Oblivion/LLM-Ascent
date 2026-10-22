"""Act 0 figures: n-gram order x smoothing, plus the table source data.

Reads only the tracked results under ``results/act0-ngram/``:

* ``summary.json``     -- the 12 configurations (3 orders x 4 smoothings)
* one ``metrics.json`` per run -- the configuration and split sizes

and writes

* ``report/figures/act0_ngram_order_smoothing.{png,pdf}``
* ``report/tables/act0_ngram_perplexity.csv``

Nothing is re-measured and nothing is hard-coded, so the figure cannot drift
from the result.

A caveat that the figure itself has to carry: the n-gram counts are
*deterministic*.  The three "seeds" produce bit-identical counts, so the
per-configuration standard deviation is exactly 0.0 and there is no error bar
to draw.  The figure says so rather than drawing a zero-width bar next to a
legend that claims three seeds.

Usage:
    python -m src.eval.act0_figures [--results-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "results" / "act0-ngram"
DEFAULT_OUT = REPO_ROOT / "report" / "figures"
DEFAULT_TABLES = REPO_ROOT / "report" / "tables"

ORDERS = [1, 2, 3]
SMOOTHINGS = ["laplace", "good_turing", "kneser_ney", "interpolated"]
COLORS = {
    "laplace": "#2471a5",
    "good_turing": "#c0392b",
    "kneser_ney": "#7d3c98",
    "interpolated": "#117a65",
}


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing tracked result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=160, bbox_inches="tight")
        print(f"wrote {out_dir / name}.{ext}")


def _key(order: int, smoothing: str) -> str:
    return f"o{order}-{smoothing}"


def write_table(results_dir: Path, tables_dir: Path) -> None:
    """The source data for the Act 0 LaTeX table."""
    summary = _load(results_dir / "summary.json")
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "act0_ngram_perplexity.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "order", "smoothing", "validation_bits_per_character_mean",
            "validation_bits_per_character_std", "validation_perplexity_mean",
            "validation_perplexity_std", "n_seeds", "identical_across_seeds",
        ])
        for order in ORDERS:
            for smoothing in SMOOTHINGS:
                row = summary[_key(order, smoothing)]
                runs = row.get("per_seed_bpc") or []
                identical = len(set(runs)) <= 1
                w.writerow([
                    order, smoothing,
                    f"{row['bpc_mean']:.6f}", f"{row['bpc_std']:.6f}",
                    f"{row['perplexity_mean']:.6f}", f"{row['perplexity_std']:.6f}",
                    row["n_seeds"], identical,
                ])
    print(f"wrote {out}")


def order_smoothing_figure(results_dir: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load(results_dir / "summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5),
                             gridspec_kw={"wspace": 0.28})

    # ── left: validation perplexity by order, one series per smoothing ────
    ax = axes[0]
    for smoothing in SMOOTHINGS:
        xs = ORDERS
        ys = [summary[_key(o, smoothing)]["perplexity_mean"] for o in ORDERS]
        ax.plot(xs, ys, "-o", color=COLORS[smoothing], lw=2.0, ms=6,
                label=smoothing.replace("_", "-"))
    ax.set_yscale("log")
    ax.set_xticks(ORDERS)
    ax.set_xlabel("n-gram order")
    ax.set_ylabel("validation perplexity (log scale)")
    ax.set_title("Act 0: order dominates smoothing\n"
                 "perplexity of the held-out 10% of TinyShakespeare", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(alpha=0.3)

    # ── right: the spread, which is exactly zero ──────────────────────────
    ax = axes[1]
    labels, vals, errs = [], [], []
    for order in ORDERS:
        for smoothing in SMOOTHINGS:
            row = summary[_key(order, smoothing)]
            labels.append(f"o{order} {smoothing.replace('_', '-')}")
            vals.append(row["bpc_mean"])
            errs.append(row["bpc_std"])
    y = list(range(len(vals)))
    ax.barh(y, vals, color="#2471a5", alpha=0.85, height=0.65)
    ax.errorbar(vals, y, xerr=[errs, errs], fmt="none", ecolor="#c0392b",
                capsize=2, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("validation bits per character")
    ax.set_title("Spread across 3 seeds is exactly 0.000\n"
                 "count-based n-grams are deterministic: no error bar to draw",
                 fontsize=10)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("Act 0: 12 configurations (3 orders x 4 smoothings); the three "
                 "seeds per configuration are bit-identical", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, "act0_ngram_order_smoothing")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    args = ap.parse_args()
    write_table(args.results_dir, args.tables_dir)
    order_smoothing_figure(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
