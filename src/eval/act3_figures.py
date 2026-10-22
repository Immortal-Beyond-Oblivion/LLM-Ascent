"""Act 3 figures: tokenizer families under two context controls.

Reads only the tracked results under ``results/act3-tokenizers/``:

* ``summary.json`` -- validation bits per character for every (tokenizer,
  control) pair
* one ``tokenizer_stats.json`` per run -- vocabulary size, characters per
  token, and the context length that the control used

and writes

* ``report/figures/act3_tokenizer_comparison.{png,pdf}``
* ``report/tables/act3_tokenizer_comparison.csv``

Two controls are run for every tokenizer:

``ecn``  equal character count -- the same number of characters in the window
``ecs``  equal character span -- the same *span of text* in the window

The second control is the one that makes the comparison meaningful: it removes
the v1 confound in which a 64-token window shows a character model 64
characters of text and a BPE model about 104.

Caveat that the figure itself carries: every configuration was run at **one
seed** (42).  There is no error bar, and the numbers rank tokenizers rather
than establishing an effect.

Usage:
    python -m src.eval.act3_figures [--results-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "results" / "act3-tokenizers"
DEFAULT_OUT = REPO_ROOT / "report" / "figures"
DEFAULT_TABLES = REPO_ROOT / "report" / "tables"

CONTROLS = [
    ("ecn", "standard", "equal character count"),
    ("ecs", "equal_char_span", "equal character span"),
]
TOKENS = ["character", "bpe", "wordpiece", "unigram"]
COLORS = {
    "character": "#2471a5",
    "bpe": "#c0392b",
    "wordpiece": "#7d3c98",
    "unigram": "#117a65",
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


def _stats(results_dir: Path, tokenizer: str, control: str) -> dict:
    return _load(results_dir / "act3-tokenizers" /
                 f"{tokenizer}-seed42-{control}" / "tokenizer_stats.json")


def write_table(results_dir: Path, tables_dir: Path) -> None:
    summary = _load(results_dir / "summary.json")
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "act3_tokenizer_comparison.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "tokenizer", "control", "validation_bits_per_character",
            "validation_perplexity_per_character", "vocabulary_size",
            "chars_per_token", "context_length_used", "n_seeds",
        ])
        for tokenizer in TOKENS:
            for control, summary_tag, _ in CONTROLS:
                key = f"{tokenizer}-{summary_tag}"
                st = _stats(results_dir, tokenizer, control)
                w.writerow([
                    tokenizer, control,
                    f"{summary[key]['bpc_mean']:.6f}",
                    f"{st['validation_perplexity_per_character']:.6f}",
                    st["vocabulary_size"],
                    f"{st['chars_per_token']:.6f}",
                    st["context_length_used"],
                    summary[key]["n_seeds"],
                ])
    print(f"wrote {out}")


def comparison_figure(results_dir: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load(results_dir / "summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7),
                             gridspec_kw={"wspace": 0.26})

    # ── left: bits per character, both controls ───────────────────────────
    ax = axes[0]
    x = list(range(len(TOKENS)))
    width = 0.36
    for j, (control, summary_tag, label) in enumerate(CONTROLS):
        vals = [summary[f"{t}-{summary_tag}"]["bpc_mean"] for t in TOKENS]
        pos = [i + (j - 0.5) * width for i in x]
        ax.bar(pos, vals, width=width, color=[COLORS[t] for t in TOKENS],
               alpha=1.0 if j == 0 else 0.55, zorder=3,
               edgecolor="#222222", linewidth=0.6)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.03, f"{v:.3f}", ha="center", fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#888888", alpha=1.0,
                      edgecolor="#222222", label="equal character count (ecn)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#888888", alpha=0.55,
                      edgecolor="#222222", label="equal character span (ecs)"),
    ]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper left")
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in TOKENS], fontsize=9.5)
    ax.set_ylabel("validation bits per character (lower is better)")
    ax.set_ylim(0, max(summary[f"{t}-{tag}"]["bpc_mean"]
                       for t in TOKENS for _, tag, _ in CONTROLS) * 1.28)
    ax.set_title("Act 3: tokenizer families under two context controls\n"
                 "SINGLE SEED (n=1) -- exploratory, no error bars", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # ── right: the confound the second control removes ───────────────────
    ax = axes[1]
    ctx = [_stats(results_dir, t, "ecs")["context_length_used"] for t in TOKENS]
    cpt = [_stats(results_dir, t, "ecs")["chars_per_token"] for t in TOKENS]
    span = [c * t for c, t in zip(ctx, cpt)]
    ax.bar(x, span, color=[COLORS[t] for t in TOKENS], width=0.6, zorder=3)
    for i, (s, c, t) in enumerate(zip(span, ctx, cpt)):
        ax.text(i, s * 0.5, f"{s:.0f} chars\n{c} tok\n{t:.2f} ch/tok",
                ha="center", va="center", fontsize=6.8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in TOKENS], fontsize=9.5)
    ax.set_ylabel("characters of text visible in the window")
    ax.set_ylim(0, max(span) * 1.18)
    ax.set_title("The equal-character-span control equalises this\n"
                 "the v1 confound was that BPE saw ~1.6x more text per step",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Act 3: 4 tokenizer families x 2 context controls, seed 42, "
                 "500 steps", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, out_dir, "act3_tokenizer_comparison")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    args = ap.parse_args()
    write_table(args.results_dir, args.tables_dir)
    comparison_figure(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
