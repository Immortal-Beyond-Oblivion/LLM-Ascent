"""Act 9 control figures. Reads tracked results only.

  act9_lens_controls   the logit-lens controls: the median rank of five
                       different targets under the model's own readout, per
                       layer.  The point of the figure is that the
                       counterfactual-EDITED number reaches the same rank as
                       the asserted number, so the lens is number-specific
                       and not assertion-specific.

The source is `results/v2/act9/controls.json`, which is tracked; nothing here
re-measures anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONTROLS = ROOT / "results" / "v2" / "act9" / "controls.json"
FIGURES = ROOT / "report" / "figures"

SFT = "#1f77b4"
PEFT = "#d62728"
OK = "#2ca02c"
FLOOR = "#8c8c8c"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"

# (key, label, colour) in the order the table prints them.
CONDITIONS = [
    ("asserted_oracle", "(i) asserted number, oracle", SFT),
    ("edited_counterfactual", "(ii) EDITED number, counterfactual", PEFT),
    ("absent_number_oracle", "(iii) number in no passage, oracle", FLOOR),
    ("asserted_nonnumeric", "(iv) asserted number, non-numeric", ORANGE),
    ("random_token_oracle", "(v) random token id", PURPLE),
]


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURES / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


def lens_controls_figure() -> None:
    d = json.loads(CONTROLS.read_text(encoding="utf-8"))
    lens = d["9c_lens_controls"]
    n_items = lens["asserted_oracle"]["n_items"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8),
                             gridspec_kw={"wspace": 0.28})

    # ── left: per-layer median rank, one line per condition ────────────────
    ax = axes[0]
    for key, label, colour in CONDITIONS:
        ranks = lens[key]["median_rank_per_layer"]
        ax.plot(range(len(ranks)), ranks, "-o", color=colour, lw=2.0, ms=5.5,
                label=label)
    best = lens["asserted_oracle"]["best_layer"]
    ax.scatter([best], [lens["asserted_oracle"]["best_median_rank"]],
               color=OK, zorder=6, s=120)
    ax.text(0.02, 0.97,
            "the EDITED number (ii) reaches the same rank as the\n"
            "asserted number (i) at layer 5: 82 against 82, while a\n"
            "number in no passage is at 484 and a random token at 735",
            transform=ax.transAxes, fontsize=8.5, color=PEFT, va="top",
            ha="left")
    ax.set_yscale("log")
    ax.set_ylim(top=max(max(lens[k]["median_rank_per_layer"])
                        for k, _, _ in CONDITIONS) * 6)
    ax.set_xticks(range(8))
    ax.set_xlabel("layer")
    ax.set_ylabel("median rank of the target under the model's own readout")
    ax.set_title("Act 9c controls: the lens is number-specific,\n"
                 "not assertion-specific", fontsize=9.5)
    ax.legend(fontsize=7.6, loc="lower left")
    ax.grid(alpha=0.25)

    # ── right: the layer-5 rank, one bar per condition ──────────────────────
    ax = axes[1]
    best_layer = lens["asserted_oracle"]["best_layer"]
    values = [lens[key]["median_rank_per_layer"][best_layer]
              for key, _, _ in CONDITIONS]
    bars = ax.bar(range(len(values)), values,
                  color=[c for _, _, c in CONDITIONS], width=0.6)
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.08, f"{v:.0f}",
                ha="center", fontsize=10)
    ax.set_yscale("log")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(["(i)", "(ii)", "(iii)", "(iv)", "(v)"], fontsize=10)
    ax.set_ylabel(f"median rank at layer {best_layer} (log scale)")
    ax.set_title(f"Act 9c controls: the median rank at layer {best_layer}\n"
                 f"n = {n_items} items per condition, vocabulary of 6,487",
                 fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Act 9: context numbers are represented in the residual "
                 "stream; assertion-specific content is not", fontsize=11,
                 y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, "act9_lens_controls")


def main() -> None:
    lens_controls_figure()


if __name__ == "__main__":
    main()
