"""Act 4 figures: the scaling-law ladder, and loss at fixed data budget.

Reads only the tracked results under ``results/v2/act4-scaling/``:

* ``ladder_points.csv``  -- every (rung, seed, N, D, loss) point of the
  sweep, with the ``family`` / ``config_hash`` / ``model_class`` /
  ``positional_encoding`` tags written by ``src/scaling/refit_clean.py``
* ``equal_D_table.csv``  -- validation loss at fixed training-token budgets
* ``clean_refit.json``   -- the family-separated re-fit: the canonical fit
  under BOTH parameter conventions, the other-family fit, the mixed fit, the
  leave-one-rung-out folds, the profile likelihood over alpha, and the
  diagnostic that evaluates the WITHDRAWN tracked parameters on each
  candidate point set
* ``fit_results.json``   -- the withdrawn tracked fit (alpha = 0.445), kept
  only so that the figure can say which fit was withdrawn

and writes

* ``report/figures/act4_scaling_ladder.{png,pdf}``
* ``report/figures/act4_loss_vs_N_fixed_D.{png,pdf}``
* ``report/figures/act4_profile_likelihood.{png,pdf}``
* ``report/tables/act4_equal_D_table.csv``

The ladder is NOT one model family: 11 of the 16 runs are the canonical
MiniGPT (learned absolute positions, 8 heads) and 5 are a different family
with no learned position table, and the two are separable by the first logged
validation loss (8.7-8.9 nats against 129.6-223.8 nats).  The clean fit is
therefore drawn on the canonical family only (2,292 interior points, 4 rungs),
and the other family is drawn in a different colour so that the contamination
is visible in the figure rather than only in the prose.

The paper's fit is the canonical family on the TOTAL-parameter axis, because
that is the convention the published exponents use and the one on which the
fit is stable.  The profile-likelihood figure shows the alpha profile and the
beta profile, both conventions, side by side, because the DIFFERENCE between
them is the point: the total-N alpha range [0.08, 0.54] is compatible with
published exponents across their ENTIRE range and therefore discriminates
between none of them, while the total-N beta range [0.36, 0.56] EXCLUDES
Chinchilla's 0.28.  The figure also carries the measured unigram entropy
(4.9946 nats), because the fitted E is an extrapolated asymptote and not the
corpus's floor.

Usage:
    python -m src.eval.act4_figures [--results-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "results" / "v2" / "act4-scaling"
DEFAULT_OUT = REPO_ROOT / "report" / "figures"
DEFAULT_TABLES = REPO_ROOT / "report" / "tables"

RUNG_COLORS = {
    "0.8M": "#1f77b4",
    "1.5M": "#ff7f0e",
    "2.6M": "#2ca02c",
    "4.1M": "#d62728",
    "6.3M": "#9467bd",
    "9.0M": "#8c564b",
}
# The two architecture families the ladder mixed.  Canonical = MiniGPT with a
# learned absolute position table; other = the family with no position table,
# which starts two orders of magnitude worse in loss.
FAMILY_COLORS = {"canonical": "#1f77b4", "other": "#d62728"}
FAMILY_STYLE = {"canonical": "-", "other": "--"}
FAMILY_LABEL = {
    "canonical": "canonical family (MiniGPT, learned absolute positions)",
    "other": "other family (no learned position table)",
}
# The two budgets highlighted in the fixed-D figure: the inversion at D=2M and
# two budgets where every rung reached.
FIXED_D = [2_000_000, 10_000_000, 40_000_000]
# The minimum training-token budget of the points that entered the clean fit.
D_MIN_FIT = 5_000_000


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing tracked result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=160, bbox_inches="tight")
        print(f"wrote {out_dir / name}.{ext}")


def _read_ladder(results_dir: Path) -> list[dict]:
    path = results_dir / "ladder_points.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing tracked result: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append({
            "rung": r["rung"],
            "seed": int(r["seed"]),
            "N": int(r["non_embedding_params"]),
            "D": int(r["training_tokens"]),
            "loss": float(r["validation_loss"]),
            "in_fit": r.get("role") == "canonical_ladder",
            "family": r.get("family") or "",
            "model_class": r.get("model_class") or "",
            "positional_encoding": r.get("positional_encoding") or "",
        })
    return out


def _fold_alpha(fit: dict) -> tuple[float, float, float, float]:
    """(mean, s.d., min, max) of the leave-one-rung-out fold estimates.

    RETIRED as a way to quote alpha: a standard deviation over four folds of
    an unidentified model is false precision, and the paper quotes the
    fold-wise estimates INDIVIDUALLY together with the profile likelihood.
    The helper is kept because the fold-wise values it reads are the ones the
    profile figure plots.
    """
    alphas = [float(v["alpha"]) for v in fit.get("loro", {}).values()
              if v.get("alpha") is not None]
    if not alphas:
        raise ValueError("the fit carries no leave-one-rung-out alpha values")
    mean = sum(alphas) / len(alphas)
    if len(alphas) > 1:
        var = sum((a - mean) ** 2 for a in alphas) / (len(alphas) - 1)
        sd = var ** 0.5
    else:
        sd = 0.0
    return mean, sd, min(alphas), max(alphas)


def _read_equal_d(results_dir: Path) -> tuple[list[int], dict[int, dict[str, float]]]:
    path = results_dir / "equal_D_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing tracked result: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    ds = [int(r["D"]) for r in rows]
    table: dict[int, dict[str, float]] = {}
    for r in rows:
        table[int(r["D"])] = {
            k: (float(v) if v not in ("", None) else None)
            for k, v in r.items() if k != "D"
        }
    return ds, table


def write_table(results_dir: Path, tables_dir: Path) -> None:
    """Copy the fixed-D table into the report's table directory, with a
    provenance header, so the LaTeX table can be re-derived from it."""
    src = results_dir / "equal_D_table.csv"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "act4_equal_D_table.csv"
    with src.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# source: results/v2/act4-scaling/equal_D_table.csv"])
        w.writerow(["# rows are validation loss (nats) at a fixed training-token "
                    "budget D; columns are rungs (non-embedding parameters)"])
        w.writerows(rows)
    print(f"wrote {out}")


def ladder_figure(results_dir: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = _read_ladder(results_dir)
    clean = _load(results_dir / "clean_refit.json")
    # The paper's fit: the canonical family on the TOTAL-parameter axis.
    canonical = clean["fits_under_the_total_parameter_convention"][
        "canonical_only_D5M_total_N"]
    prof = clean["profile_likelihood_over_alpha"]["canonical_only_D5M_total_N"]

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8),
                             gridspec_kw={"wspace": 0.22})

    # Axis setup first, so that the limits are not overwritten by the run loop.
    axes[1].set_yscale("log")
    for ax, (lo, hi) in zip(axes, ((2.4, 9.6), (2.4, 10.0))):
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Training tokens D (millions)")
        ax.set_ylabel("Validation loss (nats)")
        ax.grid(alpha=0.3)

    # One line per RUN, coloured by architecture family.  The ladder is not
    # one model family, and the figure has to show that, because the fit is
    # drawn on the canonical family only.
    runs: dict[tuple[str, str], list[dict]] = {}
    for p in points:
        runs.setdefault((p["rung"], p["family"]), []).append(p)
    for (rung, family), pts in sorted(runs.items()):
        pts = sorted(pts, key=lambda p: p["D"])
        for ax in axes:
            ax.plot([p["D"] / 1e6 for p in pts], [p["loss"] for p in pts],
                    FAMILY_STYLE.get(family, "-"),
                    color=FAMILY_COLORS.get(family, "#8c8c8c"),
                    lw=1.1, alpha=0.85)

    # The interior region that entered the fit.
    for ax in axes:
        ax.axvspan(D_MIN_FIT / 1e6, ax.get_xlim()[1], color="#999999",
                   alpha=0.10, zorder=0)
    axes[0].annotate("shaded: the interior region the fit uses (D >= 5M);\n"
                     "y-axis clipped at 2.4 nats, so the initial transient\n"
                     "(loss > 200 nats) is off scale",
                     xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8,
                     color="#555555", va="bottom")

    handles = []
    for family, color in FAMILY_COLORS.items():
        handles.append(plt.Line2D([], [], color=color,
                                  ls=FAMILY_STYLE[family], lw=1.6,
                                  label=FAMILY_LABEL[family]))
    for ax in axes:
        leg = ax.legend(handles=handles, fontsize=9, framealpha=0.95,
                        loc="upper right")
        leg.set_zorder(5)
    axes[0].set_title("Act 4 - the ladder mixes TWO architecture families",
                      fontsize=11)
    axes[1].set_title("Act 4 - the same ladder, log y", fontsize=11)

    prov = clean["provenance"]
    tracked = _load(results_dir / "fit_results.json")["secondary"]
    on = clean["tracked_params_on_candidate_sets"]
    entropy = _load(results_dir / "unigram_entropy.json")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.09)
    _save(fig, out_dir, "act4_scaling_ladder")


def fixed_d_figure(results_dir: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = _read_ladder(results_dir)
    _, table = _read_equal_d(results_dir)
    clean = _load(results_dir / "clean_refit.json")

    # One N per rung, taken from the CANONICAL run of that rung where one
    # exists (the 4.1M and 9.0M rungs have none), and from the other family
    # otherwise.  The family of each rung is recorded so the figure can be
    # read against the family split.
    N_of_rung: dict[str, int] = {}
    family_of_rung: dict[str, str] = {}
    for rung in sorted({p["rung"] for p in points}):
        cand = [p for p in points if p["rung"] == rung]
        canonical = [p for p in cand if p["family"] == "canonical"]
        pick = (canonical or cand)[0]
        N_of_rung[rung] = pick["N"]
        family_of_rung[rung] = pick["family"]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for color, D in zip(colors, FIXED_D):
        row = table.get(D, {})
        xs, ys, ms = [], [], []
        for rung, value in row.items():
            if value is None:
                continue
            xs.append(N_of_rung.get(rung, 0) / 1e6)
            ys.append(value)
            ms.append("o" if family_of_rung.get(rung) == "canonical" else "s")
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        ms = [ms[i] for i in order]
        ax.plot(xs, ys, "-", color=color, lw=2.0, ms=8, label=f"D={D//1_000_000}M")
        for x, y, m in zip(xs, ys, ms):
            ax.plot([x], [y], m, color=color, ms=8, mec="black", mew=0.8)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7.5)

    # The inversion at D = 2M: 1.5M scores better than 2.6M.
    inv = table.get(2_000_000, {})
    if "1.5M" in inv and "2.6M" in inv:
        x1 = N_of_rung.get("1.5M", 0) / 1e6
        x2 = N_of_rung.get("2.6M", 0) / 1e6
        ax.annotate("inversion:\n1.5M beats 2.6M\n(gap 0.057 nats, inside\n"
                    "the 0.186 seed spread) -- a NULL",
                    xy=(x2, inv["2.6M"]), xytext=(x2 + 0.65, 4.72),
                    fontsize=8, color="#c0392b", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

    handles = [
        plt.Line2D([], [], marker="o", color="#555555", ls="none", ms=8,
                   label="rung with a canonical run"),
        plt.Line2D([], [], marker="s", color="#555555", ls="none", ms=8,
                   label="rung with only other-family runs"),
    ]
    leg1 = ax.legend(fontsize=9, framealpha=0.95, loc="upper right")
    ax.legend(handles=handles, fontsize=8, framealpha=0.95, loc="lower left")
    ax.add_artist(leg1)
    ax.set_xlabel("Non-embedding parameters N (millions)")
    ax.set_ylabel("Validation loss (nats)")
    ax.set_title("Act 4 - Loss vs N at fixed D (from the tracked equal-D table)\n"
                 "the D=2M inversion is a NULL (seed spread 0.186 > gap 0.057);"
                 " the three canonical-rung columns predate the current runs",
                 fontsize=9.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "act4_loss_vs_N_fixed_D")


def profile_figure(results_dir: Path, out_dir: Path) -> None:
    """The profile likelihood over alpha AND over beta, under both
    parameter conventions.

    This is the PRIMARY statement of Act 4.  The point estimate of a fit
    whose alpha is not identified is not a confidence statement, so the
    paper quotes the set of alphas the data cannot distinguish from each
    other: the profile likelihood, with alpha held fixed and the other four
    parameters re-optimised, and R^2 recorded at each step of a 0.02-wide
    grid.

    Top row: the alpha profile (left) and the beta profile (right), each
    under both parameter conventions, with each curve's own best fit and its
    0.001 tolerance line marked, the compatible range shaded, and the
    published value marked (0.34 for alpha, 0.28 for beta).  The beta panel
    is the informative one: the published beta falls OUTSIDE the compatible
    range.

    Bottom row: the four leave-one-rung-out alpha estimates, quoted
    individually (left), and the compatible-range widths side by side
    (right), which is the diagnosis of the asymmetry.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = _load(results_dir / "clean_refit.json")
    prof = clean["profile_likelihood_over_alpha"]
    bprof = clean["profile_likelihood_over_beta"]
    total = prof["canonical_only_D5M_total_N"]
    nonemb = prof["canonical_only_D5M_non_embedding_N"]
    btotal = bprof["canonical_only_D5M_total_N"]
    bnonemb = bprof["canonical_only_D5M_non_embedding_N"]
    fit_total = clean["fits_under_the_total_parameter_convention"][
        "canonical_only_D5M_total_N"]
    fit_non = clean["fits"]["canonical_only_D5M"]
    entropy = _load(results_dir / "unigram_entropy.json")

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.2),
                             gridspec_kw={"wspace": 0.22, "hspace": 0.42})

    def _profile(ax, curves, key, published, title, ylabel_note):
        """One profile panel: two conventions, shaded compatible ranges."""
        for p, colour, style, label in curves:
            xs = [c[key] for c in p["curve"]]
            ys = [c["r2"] for c in p["curve"]]
            ax.plot(xs, ys, style, color=colour, lw=1.8, label=label)
            ax.axhline(p["best_r2"] - p["delta_r2"], color=colour, lw=0.9,
                       ls=":", alpha=0.9)
            ax.axvspan(p["compatible_%s_min" % key.split("_")[0]],
                       p["compatible_%s_max" % key.split("_")[0]],
                       color=colour, alpha=0.12, zorder=0)
        ax.axvline(published, color="#333333", lw=1.3, ls="-.")
        ax.set_xlabel(key.split("_")[0])
        ax.set_ylabel(r"$R^2$ (against a constant-loss baseline)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, framealpha=0.95, loc="lower left")
        ax.set_title(title, fontsize=10)
        ax.text(0.985, 0.155,
                "published %s = %.2f (dash-dot line)"
                % (key.split("_")[0], published),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                color="#333333")
        ax.text(0.985, 0.02, ylabel_note, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color="#555555")

    # ── (0,0) the alpha profile ───────────────────────────────────────────
    _profile(
        axes[0][0],
        ((total, "#1f77b4", "-", "total-parameter N (the paper's axis)"),
         (nonemb, "#d62728", "--", "non-embedding N")),
        "alpha", 0.34,
        "Act 4 - the profile likelihood over ALPHA\n"
        "(dotted: each curve's own best fit minus 0.001 in $R^2$)",
        "total N: compatible [%.2f, %.2f] (width %.2f)\n"
        "contains 0.34 AND 0.50: no discrimination"
        % (total["compatible_alpha_min"], total["compatible_alpha_max"],
           total["compatible_alpha_max"] - total["compatible_alpha_min"]))
    axes[0][0].set_ylim(0.950, 0.9962)

    # ── (0,1) the beta profile ────────────────────────────────────────────
    _profile(
        axes[0][1],
        ((btotal, "#1f77b4", "-", "total-parameter N (the paper's axis)"),
         (bnonemb, "#d62728", "--", "non-embedding N")),
        "beta", 0.28,
        "Act 4 - the profile likelihood over BETA\n"
        "(the published 0.28 falls OUTSIDE the compatible range)",
        "total N: compatible [%.2f, %.2f] (width %.2f)\n"
        "EXCLUDES Chinchilla's 0.28"
        % (btotal["compatible_beta_min"], btotal["compatible_beta_max"],
           btotal["compatible_beta_max"] - btotal["compatible_beta_min"]))
    axes[0][1].set_ylim(0.250, 0.9962)

    # ── (1,0) the fold-wise estimates, individually ───────────────────────
    ax = axes[1][0]
    rungs = sorted(fit_total["loro"].keys())
    xs = list(range(len(rungs)))
    tot_alphas = [fit_total["loro"][r]["alpha"] for r in rungs]
    non_alphas = [fit_non["loro"][r]["alpha"] for r in rungs]
    ax.axhspan(total["compatible_alpha_min"], total["compatible_alpha_max"],
               color="#1f77b4", alpha=0.12, zorder=0,
               label="total-N compatible range")
    ax.axhline(0.34, color="#333333", lw=1.2, ls="-.", label="Chinchilla 0.34")
    ax.plot(xs, tot_alphas, "o", color="#1f77b4", ms=9, label="total N")
    ax.plot(xs, non_alphas, "s", color="#d62728", ms=8, mfc="none",
            label="non-embedding N")
    ax.axhline(fit_total["alpha"], color="#1f77b4", lw=0.9, ls=":")
    ax.axhline(fit_non["alpha"], color="#d62728", lw=0.9, ls=":")
    for x, a in zip(xs, tot_alphas):
        ax.annotate(f"{a:.3f}", (x, a), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=7.5, color="#1f77b4")
    for x, a in zip(xs, non_alphas):
        ax.annotate(f"{a:.3f}", (x, a), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=7.5, color="#d62728")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"hold out\n{r}" for r in rungs], fontsize=8.5)
    ax.set_ylabel("fold-wise alpha")
    ax.set_ylim(0.0, 0.60)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, framealpha=0.95, loc="upper left")
    ax.set_title("Act 4 - the four leave-one-rung-out estimates, quoted "
                 "individually\n(dotted: each convention's full-fit point "
                 "estimate)", fontsize=10)

    # ── (1,1) the diagnosis: the compatible-range widths ─────────────────
    ax = axes[1][1]
    labels = ["alpha\ntotal N", "beta\ntotal N", "alpha\nnon-emb.", "beta\nnon-emb."]
    widths = [
        total["compatible_alpha_max"] - total["compatible_alpha_min"],
        btotal["compatible_beta_max"] - btotal["compatible_beta_min"],
        nonemb["compatible_alpha_max"] - nonemb["compatible_alpha_min"],
        bnonemb["compatible_beta_max"] - bnonemb["compatible_beta_min"],
    ]
    ranges = [
        "[%.2f, %.2f]" % (total["compatible_alpha_min"],
                          total["compatible_alpha_max"]),
        "[%.2f, %.2f]" % (btotal["compatible_beta_min"],
                          btotal["compatible_beta_max"]),
        "[%.2f, %.2f]" % (nonemb["compatible_alpha_min"],
                          nonemb["compatible_alpha_max"]),
        "[%.2f, %.2f]" % (bnonemb["compatible_beta_min"],
                          bnonemb["compatible_beta_max"]),
    ]
    colours = ["#1f77b4", "#1f77b4", "#d62728", "#d62728"]
    bars = ax.bar(labels, widths, color=colours, alpha=0.85, width=0.62)
    for rect, w, rg in zip(bars, widths, ranges):
        ax.text(rect.get_x() + rect.get_width() / 2, w + 0.012,
                "%.2f\n%s" % (w, rg), ha="center", fontsize=8.5)
    ax.set_ylabel("width of the compatible range")
    ax.set_ylim(0.0, 0.60)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Act 4 - why: beta is 2-3x better constrained than alpha\n"
                 "alpha depends on the NUMBER AND SPREAD of distinct N values "
                 "(the ladder has four);\nbeta is constrained by within-run D "
                 "variation (hundreds of points per run)",
                 fontsize=9.5)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.06)
    _save(fig, out_dir, "act4_profile_likelihood")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    args = ap.parse_args()
    write_table(args.results_dir, args.tables_dir)
    ladder_figure(args.results_dir, args.out_dir)
    fixed_d_figure(args.results_dir, args.out_dir)
    profile_figure(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
