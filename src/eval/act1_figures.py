"""Act 1 figures: recurrent baselines, and the gradient demonstration.

Reads only the tracked results under ``results/act1-rnn/``:

* ``summary.json`` -- the per-model aggregates (mean/std over 3 seeds) and the
  gradient-norm demonstration

and writes

* ``report/figures/act1_rnn_lstm_gru.{png,pdf}``
* ``report/tables/act1_recurrent_baselines.csv``

The gradient panel is deliberately a NULL-result panel.  The demonstration was
built to show vanishing gradients; what it measured is the opposite direction
(the total gradient norm *rises* over the 100 steps for both the plain RNN and
the LSTM), so the figure plots the measured numbers and labels them as a null
rather than illustrating the effect the experiment was designed to show.

Usage:
    python -m src.eval.act1_figures [--results-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "results" / "act1-rnn"
DEFAULT_OUT = REPO_ROOT / "report" / "figures"
DEFAULT_TABLES = REPO_ROOT / "report" / "tables"

MODELS = ["rnn", "lstm", "gru"]
COLORS = {"rnn": "#2471a5", "lstm": "#c0392b", "gru": "#7d3c98"}


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing tracked result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=160, bbox_inches="tight")
        print(f"wrote {out_dir / name}.{ext}")


def write_table(results_dir: Path, tables_dir: Path) -> None:
    summary = _load(results_dir / "summary.json")
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "act1_recurrent_baselines.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "model", "validation_loss_mean", "validation_loss_std",
            "validation_perplexity_mean", "parameters", "n_seeds",
        ])
        for name in MODELS:
            row = summary[name]
            w.writerow([
                name, f"{row['validation_loss_mean']:.6f}",
                f"{row['validation_loss_std']:.6f}",
                f"{row['validation_perplexity_mean']:.6f}",
                row["parameters"], row["n_seeds"],
            ])
        for name in ("rnn", "lstm"):
            key = f"{name}_grad_norms"
            if key in summary:
                g = summary[key]
                w.writerow([
                    f"{name}_grad_norm_initial", f"{g['initial']:.6f}",
                    f"{g['mean']:.6f}", f"{g['final']:.6f}", "", 1,
                ])
    print(f"wrote {out}")


def baselines_figure(results_dir: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = _load(results_dir / "summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5),
                             gridspec_kw={"wspace": 0.3})

    # ── left: validation loss, mean +/- std over three seeds ───────────────
    ax = axes[0]
    means = [summary[m]["validation_loss_mean"] for m in MODELS]
    stds = [summary[m]["validation_loss_std"] for m in MODELS]
    x = list(range(len(MODELS)))
    ax.bar(x, means, color=[COLORS[m] for m in MODELS], width=0.6, alpha=0.9,
           zorder=3)
    ax.errorbar(x, means, yerr=[stds, stds], fmt="none", ecolor="#222222",
                capsize=5, lw=1.4, zorder=4)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.05, f"{m:.3f} ± {s:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m.upper()}\n{summary[m]['parameters']:,} params"
                        for m in MODELS], fontsize=9)
    ax.set_ylabel("validation loss (nats/token)")
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.35)
    ax.set_title("Act 1: RNN, LSTM and GRU are indistinguishable at n=3\n"
                 "error bars are +/-1 sample std over 3 seeds", fontsize=10)
    ax.annotate("every gap is inside one standard deviation",
                xy=(0.5, 0.93), xycoords="axes fraction", ha="center",
                fontsize=8.5, color="#c0392b", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # ── right: the gradient demonstration, which is a NULL ────────────────
    ax = axes[1]
    names, initials, finals, means_g = [], [], [], []
    for m in ("rnn", "lstm"):
        key = f"{m}_grad_norms"
        if key not in summary:
            continue
        g = summary[key]
        names.append(m.upper())
        initials.append(g["initial"])
        finals.append(g["final"])
        means_g.append(g["mean"])
    for i, (ini, fin, avg) in enumerate(zip(initials, finals, means_g)):
        ax.plot([0, 1], [ini, fin], "-", color=COLORS[names[i].lower()], lw=2.2)
        ax.plot([0], [ini], "o", color=COLORS[names[i].lower()], ms=8)
        ax.plot([1], [fin], "s", color=COLORS[names[i].lower()], ms=8)
        ax.plot([0.5], [avg], "^", color="#444444", ms=7)
        ax.text(1.04, fin, f"{fin:.3f}", fontsize=9, va="center")
        ax.text(-0.06, ini, f"{ini:.3f}", fontsize=9, va="center", ha="right")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["initial", "final"], fontsize=9)
    ax.set_ylabel("total gradient norm (L2 over all parameters)")
    ax.set_xlim(-0.28, 1.22)
    ax.set_title("Gradient demonstration: a NULL\n"
                 "norms RISE, they do not vanish (copy task, seq len 20)",
                 fontsize=10)
    ax.annotate("no vanishing gradient at sequence length 20",
                xy=(0.03, 0.42), xycoords="axes fraction", ha="left",
                va="bottom", fontsize=8.5, color="#c0392b", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Act 1: recurrent baselines on TinyShakespeare, 500 steps, "
                 "3 seeds", fontsize=10.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, out_dir, "act1_rnn_lstm_gru")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    args = ap.parse_args()
    write_table(args.results_dir, args.tables_dir)
    baselines_figure(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
