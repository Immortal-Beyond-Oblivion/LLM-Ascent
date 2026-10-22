"""Act 5 figures for the final report.

Reads the TRACKED results in ``results/v2/act5/`` and writes PNG + vector PDF
pairs into ``report/figures/``:

* ``act5_kv_cache``      — decode latency vs context (cached/uncached) and cache
                           memory growth, the two halves of the 5a claim.
* ``act5_speculative``   — speedup and acceptance vs draft length K at a 1.7x
                           and a 4.2x model gap, against the break-even line.
* ``act5_mqa_paired``    — the six paired MQA-minus-MQA losses behind the 5b
                           significance claim, with the mean and its spread.

Every number plotted comes from a tracked JSON file. Nothing is re-measured
here and nothing is hard-coded, so the figures cannot drift from the results.

Usage:
    python -m src.eval.act5_figures [--results-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

DEFAULT_RESULTS = Path("results/v2/act5")
DEFAULT_OUT = Path("report/figures")


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing tracked result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def kv_cache_figure(results_dir: Path, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _load(results_dir / "kv_cache.json")
    lat = d["latency_ms"]
    mem = d["cache_memory_mb"]
    ctx = [r["context"] for r in lat]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(ctx, [r["uncached"] for r in lat], "o-", label="no cache (recompute prefix)",
            color="#c0392b")
    ax.plot(ctx, [r["cached"] for r in lat], "s-", label="KV cache", color="#2471a5")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("decode latency (ms)")
    ax.set_title("5a  KV cache: latency grows linearly, not quadratically")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")

    ax2 = ax.twinx()
    ax2.plot(ctx, [r["speedup"] for r in lat], "^--", color="#117a65", label="speedup")
    ax2.set_ylabel("speedup (x)")
    ax2.set_ylim(0, max(r["speedup"] for r in lat) * 1.25)
    for x, r in zip(ctx, lat):
        ax2.annotate(f"{r['speedup']:.2f}x", (x, r["speedup"]),
                     textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8, color="#117a65")

    ax = axes[1]
    for i, row in enumerate(mem):
        name = row["model"].split()[0]
        ax.plot([128, 256, 512, 1024],
                [row[str(c)] if str(c) in row else row[c] for c in (128, 256, 512, 1024)],
                "o-", label=name, color=["#2471a5", "#b9770e"][i % 2])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("KV cache size (MB)")
    ax.set_title("5a  cache memory is exactly linear in context")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, out_dir, "act5_kv_cache")


def speculative_figure(results_dir: Path, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gaps = [("1.7x", "speculative_1.7x_multiprompt.json", "#2471a5"),
            ("4.2x", "speculative_4.2x_multiprompt.json", "#c0392b")]
    series = []
    for label, fname, colour in gaps:
        d = _load(results_dir / fname)
        per = {int(k): v for k, v in d["per_K"].items()}
        series.append((label, colour, per))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.axhline(1.0, color="black", lw=1.2, ls="--", label="break-even (no speedup)")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":", label="2x slower")
    for label, colour, per in series:
        ks = sorted(per)
        ax.plot(ks, [per[k]["median_speedup"] for k in ks], "o-",
                label=f"{label} gap", color=colour)
        ax.fill_between(ks, [per[k]["min"] for k in ks], [per[k]["max"] for k in ks],
                        color=colour, alpha=0.15)
    ax.set_xlabel("draft length K")
    ax.set_ylabel("wall-clock speedup vs plain cached decode")
    ax.set_title("5c  speculative decoding never beats plain decoding here")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for label, colour, per in series:
        ks = sorted(per)
        ax.plot(ks, [per[k]["median_acceptance"] for k in ks], "o-",
                label=f"{label} gap", color=colour)
    ax.set_xlabel("draft length K")
    ax.set_ylabel("median acceptance rate")
    ax.set_title("5c  acceptance is high but falls with K")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, out_dir, "act5_speculative")


def mqa_figure(results_dir: Path, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _load(results_dir / "mqa_quality_2.6m.json")
    runs = d["runs"]
    seeds = sorted(runs, key=int)
    diffs = [runs[s]["diff_mqa_minus_mha"] for s in seeds]
    md = d["statistics_n6"]["mean_diff"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = range(len(seeds))
    ax.bar(xs, diffs, color="#c0392b", alpha=0.85, width=0.6)
    ax.axhline(md, color="black", lw=1.2, ls="--",
               label=f"mean = {md:+.4f} nats")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("validation loss (MQA G=1  −  MHA G=8)")
    st = d["statistics_n6"]
    ax.set_title("5b  MQA costs quality at 2.6M\n"
                 f"n=6 paired, t={st['t']:.2f}, two-sided p={st['p_two_sided']:.4f}, "
                 f"{st['direction_consistent']} consistent")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    return _save(fig, out_dir, "act5_mqa_paired")


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        written.append(p)
    print(f"wrote {written[0]} and {written[1]}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    kv_cache_figure(args.results_dir, args.out_dir)
    speculative_figure(args.results_dir, args.out_dir)
    mqa_figure(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
