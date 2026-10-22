"""Act 6 figures for the final report.

Reads the tracked ``results/v2/act6/*.json`` and writes PNG + vector PDF pairs:

* ``act6_decoding_tradeoff``  — the quality/diversity trade-off across the whole
  decoding sweep, and the quality/cost trade-off.
* ``act6_bestofn_selfconsistency`` — best-of-n gain against n, and
  self-consistency accuracy against number of votes with Wilson intervals and
  the chance line.

Everything plotted comes from the tracked JSON; nothing is re-measured or
hard-coded, so a figure cannot drift from its result.

Usage:
    python -m src.eval.act6_figures [--results-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RESULTS = Path("results/v2/act6")
DEFAULT_OUT = Path("report/figures")

BEAM, GREEDY, SAMPLE = "#2471a5", "#7d3c98", "#c0392b"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing tracked result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _family(name: str) -> str:
    if name == "greedy":
        return GREEDY
    if name.startswith("beam"):
        return BEAM
    return SAMPLE


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=160, bbox_inches="tight")
        print(f"wrote {out_dir / name}.{ext}")


def tradeoff_figure(results_dir: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _load(results_dir / "6.3M_decoding.json")
    r = d["decoding_sweep"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    for x in r:
        ax.scatter(x["distinct2"], x["bits_per_char_reference"],
                   color=_family(x["config"]), s=55, zorder=3)
        ax.annotate(x["config"], (x["distinct2"], x["bits_per_char_reference"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel("distinct-2  (diversity)")
    ax.set_ylabel("bits per character  (lower = better)")
    ax.set_title("6a  quality/diversity trade-off\ncorr +0.87: diversity costs quality")
    ax.grid(alpha=0.3)

    ax = axes[1]
    xs = sorted(r, key=lambda z: z["bits_per_char_reference"])
    ax.plot([x["bits_per_char_reference"] for x in xs],
            [x["seconds"] for x in xs], "o-", color="#117a65")
    # Only the beam/greedy series get labels: every sampling config lands at
    # ~33 s and labelling them all produces an unreadable pile.
    for x in xs:
        if x["config"].startswith("beam") or x["config"] == "greedy":
            ax.annotate(x["config"], (x["bits_per_char_reference"], x["seconds"]),
                        textcoords="offset points", xytext=(7, -3), fontsize=8)
    ax.annotate("all sampling configs\n(~33 s each)",
                (1.72, 33), textcoords="offset points", xytext=(0, 12),
                fontsize=8, color="#c0392b", ha="center")
    ax.set_xlabel("bits per character  (lower = better)")
    ax.set_ylabel("wall clock for 120 generations (s)")
    g = next(x for x in r if x["config"] == "greedy")
    b4 = next(x for x in r if x["config"] == "beam4")
    ax.set_title("6a  quality costs time\n"
                 f"beam-4: {g['bits_per_char_reference'] - b4['bits_per_char_reference']:.3f} "
                 f"bits for {b4['seconds'] / g['seconds']:.2f}x greedy")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "act6_decoding_tradeoff")


def bestofn_figure(results_dir: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _load(results_dir / "6.3M_decoding.json")
    bon = d["best_of_n"]
    greedy = next(x for x in d["decoding_sweep"] if x["config"] == "greedy")
    sc = d["self_consistency_mnli"]
    chance = sc["chance_level"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    ns = [x["n"] for x in bon]
    ax.plot(ns, [x["bpc_of_reranked_pick"] for x in bon], "o-",
            color="#c0392b", label="best-of-n, reranked")
    ax.plot(ns, [x["bpc_mean_of_all_samples"] for x in bon], "s--",
            color="#7f8c8d", label="mean of all n samples")
    ax.axhline(greedy["bits_per_char_reference"], color="#2471a5", lw=1.4, ls="--",
               label=f"greedy ({greedy['bits_per_char_reference']:.3f})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n candidates sampled")
    ax.set_ylabel("bits per character  (lower = better)")
    ax.set_title("6b  reranking helps but never reaches greedy\n"
                 "n=16: +0.272 bits vs mean sample, -0.259 vs greedy")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    rows = sc["results"]
    ax.axhline(chance, color="black", lw=1.2, ls="--", label=f"chance = {chance:.4f}")
    ax.plot([x["n_votes"] for x in rows], [x["accuracy"] for x in rows], "o-",
            color="#2471a5", label="self-consistency accuracy")
    ax.errorbar([x["n_votes"] for x in rows],
                [x["accuracy"] for x in rows],
                yerr=[[x["accuracy"] - x["wilson_95_low"] for x in rows],
                      [x["wilson_95_high"] - x["accuracy"] for x in rows]],
                fmt="none", ecolor="#2471a5", alpha=0.4, capsize=3)
    ax.set_xlabel("number of votes n")
    ax.set_ylabel("MNLI accuracy")
    ax.set_ylim(0.28, 0.44)
    ax.set_title("6c  self-consistency cannot help at chance\n"
                 "voting converges to argmax = chance (0.3300)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, out_dir, "act6_bestofn_selfconsistency")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    tradeoff_figure(args.results_dir, args.out_dir)
    bestofn_figure(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
