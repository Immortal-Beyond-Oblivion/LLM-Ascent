"""Act 9 figures. Reads tracked results only.

  act9_head_ablation   per-head change in held-out loss against the
                        random-ablation null
  act9_logit_lens      the per-layer rank of the asserted number, which is
                        the mechanistic link to Acts 7b and 8d
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "v2"
FIGURES = ROOT / "report" / "figures"

FLOOR = "#8c8c8c"
PEFT = "#d62728"
SFT = "#1f77b4"
OK = "#2ca02c"


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURES / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


def head_ablation_figure() -> None:
    d = json.loads((RESULTS / "act9" / "interpretability.json").read_text())
    a = d["9a_head_ablation"]
    rows = a["per_head"]
    layers = d["model"]["num_layers"]
    heads = d["model"]["num_heads"]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    width = 0.8 / heads
    null_mean = a["null_single_head_mean_delta_ce"]
    null_sd = a["null_single_head_sd_delta_ce"]

    for h in range(heads):
        xs = [l + (h - (heads - 1) / 2) * width for l in range(layers)]
        ys = [next(r["delta_ce"] for r in rows
                   if r["layer"] == l and r["head"] == h) for l in range(layers)]
        # The null band: a single random head's effect, mean +/- 2 s.d.
        ax.bar(xs, ys, width=width * 0.9, color=FLOOR, alpha=0.75,
               label="per-head change in held-out CE" if h == 0 else None)

    ax.axhspan(null_mean - 2 * null_sd, null_mean + 2 * null_sd,
               color=PEFT, alpha=0.12, zorder=0,
               label="random single-head null, mean $\\pm$ 2 s.d.")
    ax.axhline(null_mean, color=PEFT, lw=1.2, ls="--")

    top = a["top_heads"][0]
    ax.annotate(f"L{top['layer']} H{top['head']}:  $\\Delta$CE "
                f"{top['delta_ce']:+.3f} nats, z = {top['z_score']:+.1f}\n"
                f"$\\Delta$PPL {top['delta_ppl']:+.1f}, attention entropy "
                f"{top['mean_entropy_nats']:.2f} nats",
                xy=(top["layer"], top["delta_ce"]),
                xytext=(top["layer"] + 1.4, top["delta_ce"] * 0.92),
                fontsize=8.5, color=PEFT,
                arrowprops=dict(arrowstyle="->", color=PEFT, lw=1.1))

    ax.set_xticks(range(layers))
    ax.set_xticklabels([f"layer {l}" for l in range(layers)])
    ax.set_xlabel("layer (heads within a layer, left to right)")
    ax.set_ylabel("change in held-out cross-entropy (nats)")
    ax.set_title("Act 9a: causal head ablation against a random-head null\n"
                 "64 heads, 64 held-out C4 sequences of 128 tokens; "
                 "the intact baseline is measured in this run", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "act9_head_ablation")


def logit_lens_figure() -> None:
    d = json.loads((RESULTS / "act9" / "interpretability.json").read_text())
    c = d["9c_logit_lens"]
    ranks = c["median_rank_of_asserted_number_per_layer"]
    layers = list(range(len(ranks)))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5),
                             gridspec_kw={"wspace": 0.3})

    ax = axes[0]
    ax.plot(layers, ranks, "-o", color=SFT, lw=2.2, ms=7)
    best = min(range(len(ranks)), key=lambda i: ranks[i])
    ax.scatter([best], [ranks[best]], color=OK, zorder=5, s=110)
    ax.annotate(f"most decodable: layer {best}\nmedian rank {ranks[best]:.0f} "
                f"of {c['vocabulary_size']}\n"
                "(the asserted number; the controls show the\n"
                "EDITED number reaches the same rank)",
                xy=(best, ranks[best]), xytext=(best + 0.7, ranks[best] * 0.55),
                fontsize=8.5, color=OK,
                arrowprops=dict(arrowstyle="->", color=OK, lw=1.1))
    ax.annotate("not carried to the output:\nrank rises again at the last layers",
                xy=(len(ranks) - 1, ranks[-1]), xytext=(len(ranks) - 3.4, ranks[-1] * 1.55),
                fontsize=8.5, color=PEFT,
                arrowprops=dict(arrowstyle="->", color=PEFT, lw=1.1))
    ax.set_yscale("log")
    ax.set_xticks(layers)
    ax.set_xlabel("layer")
    ax.set_ylabel("median rank of the asserted number")
    ax.set_title("Act 9c: logit lens, per layer\n"
                 "the model's own readout applied to each layer's residual",
                 fontsize=9.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    b = d["9b_rollout"]
    labels = ["attention\nrollout", "raw attention\n(last layer)"]
    rates = [b["rollout_hit_rate"], b["raw_attention_hit_rate"]]
    los = [b["rollout_hits_wilson95"][0], b["raw_attention_hits_wilson95"][0]]
    his = [b["rollout_hits_wilson95"][1], b["raw_attention_hits_wilson95"][1]]
    ax.bar([0, 1], rates, color=[SFT, FLOOR], width=0.55)
    ax.errorbar([0, 1], rates, yerr=[[r - l for r, l in zip(rates, los)],
                                     [h - r for r, h in zip(rates, his)]],
                fmt="none", ecolor="black", capsize=5, lw=1.2)
    ax.axhline(b["uniform_chance"], color=PEFT, lw=1.4, ls="--",
               label=f"uniform chance ({b['uniform_chance']:.4f})")
    for i, r in enumerate(rates):
        ax.text(i, r + 0.02, f"{r:.3f}", ha="center", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("hit rate on the asserted-number token")
    ax.set_ylim(0, 0.14)
    ax.set_title("Act 9b: does attribution find the answer token?\n"
                 f"n = {b['n_items_with_ground_truth']} items, Wilson 95%",
                 fontsize=9.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Act 9: a context number is most decodable mid-stack and is "
                 "not carried to the output; attention attribution never "
                 "finds the answer token", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, "act9_logit_lens")


def main() -> None:
    head_ablation_figure()
    logit_lens_figure()


if __name__ == "__main__":
    main()
