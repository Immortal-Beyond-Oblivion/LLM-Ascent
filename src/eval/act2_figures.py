"""Act 2 figures: Transformer component ablations.

Reads only the tracked results under ``results/act2-ablations/``:

* ``summary.json`` -- the 20 configurations, all at seed 42
* one ``metrics.json`` per configuration -- the full configuration

and writes

* ``report/figures/act2_ablations.{png,pdf}``
* ``report/tables/act2_ablation_summary.csv``

Caveat that the figure itself carries, because it is the single most
caveated section of the report: every configuration was run at **one seed**.
There is no error bar to draw, no significance test is possible, and the
numbers RANK configurations; they do not establish effects.  The figure says
so in its own title area rather than relying on the caption alone.

There are 18 configurations in ``summary.json`` (3 normalisation families of 2
placements plus a no-normalisation control, 5 positional encodings, 3 head
counts and 3 FFN activations).  The script asserts that the set of
configurations it plots is exactly the set in ``summary.json``, so a silently
missing or extra run cannot slip through.

Usage:
    python -m src.eval.act2_figures [--results-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "results" / "act2-ablations"
DEFAULT_OUT = REPO_ROOT / "report" / "figures"
DEFAULT_TABLES = REPO_ROOT / "report" / "tables"

# The 20 configurations, grouped by the component being ablated.
FAMILIES = [
    ("normalisation", ["layernorm-pre", "layernorm-post", "rmsnorm-pre",
                        "rmsnorm-post", "batchnorm-pre", "batchnorm-post",
                        "none-pre"]),
    ("positional encoding", ["pe-learned", "pe-sinusoidal", "pe-rope",
                              "pe-alibi", "pe-none"]),
    ("attention heads", ["heads-1", "heads-2", "heads-4"]),
    ("FFN activation", ["ffn-gelu", "ffn-relu", "ffn-swiglu"]),
]
FAMILY_COLORS = {
    "normalisation": "#2471a5",
    "positional encoding": "#c0392b",
    "attention heads": "#7d3c98",
    "FFN activation": "#117a65",
}
FAMILY_MARKERS = {
    "normalisation": "o",
    "positional encoding": "s",
    "attention heads": "^",
    "FFN activation": "D",
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


def _configs(results_dir: Path) -> list[tuple[str, str, dict]]:
    """(family, short name, summary row) for all 20 configurations."""
    summary = _load(results_dir / "summary.json")
    out: list[tuple[str, str, dict]] = []
    for family, names in FAMILIES:
        for short in names:
            key = f"{short}-seed42"
            if key not in summary:
                raise KeyError(f"missing configuration {key} in summary.json")
            out.append((family, short, summary[key]))
    seen = set(summary)
    expected = {f"{s}-seed42" for _, names in FAMILIES for s in names}
    if seen != expected:
        raise ValueError(
            "summary.json configurations do not match the expected set:\n"
            f"  unexpected: {sorted(seen - expected)}\n"
            f"  missing:    {sorted(expected - seen)}")
    return out


def write_table(results_dir: Path, tables_dir: Path) -> None:
    rows = _configs(results_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "act2_ablation_summary.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "family", "configuration", "validation_loss", "parameters",
            "n_seeds", "error_bars", "significance_test",
        ])
        for family, short, row in rows:
            w.writerow([
                family, short, f"{row['validation_loss_mean']:.6f}",
                row["parameters"], row["n_seeds"],
                "none (n=1)", "none (n=1)",
            ])
    print(f"wrote {out}")


def ablations_figure(results_dir: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _configs(results_dir)
    labels = [f"{short}  ({row['parameters']:,})" for _, short, row in rows]
    vals = [row["validation_loss_mean"] for _, _, row in rows]
    colors = [FAMILY_COLORS[fam] for fam, _, _ in rows]

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    y = list(range(len(vals)))
    ax.barh(y, vals, color=colors, height=0.68, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    # Room above the first bar for the caveat, so that it cannot overlap a
    # bar or its value label.
    ax.set_ylim(len(vals) - 0.5, -1.9)
    ax.set_xlabel("validation loss (nats/token), lower is better")
    ax.set_xlim(min(vals) - 0.03, max(vals) + 0.05)

    handles = [
        plt.Line2D([], [], marker=FAMILY_MARKERS[f], ls="", color=c,
                   label=f)
        for f, c in FAMILY_COLORS.items()
    ]
    ax.legend(handles=handles, fontsize=8.5, frameon=False,
              loc="upper right", ncol=2)
    ax.set_title("Act 2: Transformer component ablations, 18 configurations\n"
                 "SINGLE SEED (n=1) -- exploratory, no error bars, "
                 "no significance testing", fontsize=10)
    ax.annotate("n=1: these numbers RANK configurations;\n"
                "they do not establish effects",
                xy=(0.02, 0.985), xycoords="axes fraction", va="top",
                fontsize=8.5, color="#c0392b", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "act2_ablations")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    args = ap.parse_args()
    write_table(args.results_dir, args.tables_dir)
    ablations_figure(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
