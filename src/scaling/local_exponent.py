"""Act 4: the local-exponent diagnostic, and the joint (alpha, E) profile.

WHY. The alpha profile's width [0.08, 0.54] was reported as "the data cannot
distinguish these exponents". But a wide profile in a FIVE-parameter fit can
also be an (E, alpha) DEGENERACY: E and alpha trade off against each other, so
the profile traces a RIDGE in E-alpha space rather than a flat basin in alpha.
The way to tell the two apart is to look at the ladder's LOCAL exponent.

The local exponent between two neighbouring rungs is

    alpha_loc(N) = -[dL/dlnN] / (L - E)

and if the ladder really is a power law, that quantity is a CONSTANT equal to
alpha. Whether it is constant, and at what value, depends on E -- so the
diagnostic is E-sensitive even though a perfect power law would not be.

Two versions are computed, because they answer different questions:

  * ASCENDING-D, RAW (the back-of-envelope version): the slope in lnN is taken
    across the rungs' final validation losses, so the B/D^beta term is still
    present. It is a weighted average of alpha and beta, which overstates alpha
    wherever the D-term is small and drags toward beta at the top of the ladder.
  * D-TERM REMOVED: each rung's own interior points are fitted with
    L_i(D) = c_i + b_i D^-beta, and c_i is that rung's asymptote, so the
    D-term is removed by construction before the slope is taken.

The (alpha, E) joint profile then shows whether the width is a ridge.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "v2" / "act4-scaling"


#: How far above its own run's final validation loss a point may sit and
#: still count as "past the transient". Scale-free, so it means the same
#: thing at every rung. The old filter was an absolute D >= 5,000,000, which
#: kept 7 of the 0.3M rung's 32 points and produced a local exponent of
#: -1.16 for that interval.
TRANSIENT_MARGIN = 0.5


def load_canonical(csv_path: Path, *, d_min: int | None = None):
    """Interior points of the canonical ladder, grouped by run.

    Selection is by ROLE, from the manifest that writes this file. The role
    says what a run is FOR; a family label says what it looks like, and the
    ladder contains a run that looks canonical and matches no known shape.

    "Interior" means past the run's own transient, which is a property of the
    loss. An absolute token floor is not: it is past the transient for a rung
    whose run reaches 180M tokens and absurdly early for one that reaches 6M.
    """
    import csv

    runs = defaultdict(list)
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("role") or "").strip() != "canonical_ladder":
                continue
            loss = float(row["validation_loss"])
            final = float(row.get("final_val_loss") or 0.0)
            if d_min is not None:
                if float(row["training_tokens"]) < d_min:
                    continue
            elif not (final and loss <= final + TRANSIENT_MARGIN):
                continue
            key = (row["rung"], row["seed"], (row.get("arm") or "ladder").strip())
            runs[key].append(
                (float(row["training_tokens"]), loss,
                 float(row["non_embedding_params"])))
    return {k: sorted(v) for k, v in runs.items()}


def fit_rung_asymptote(points, *, beta: float | None = None):
    """Fit L(D) = c + b D^-beta and return (c, b, beta, r2)."""
    D = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    if beta is None:
        best = None
        for bt in np.arange(0.05, 1.51, 0.01):
            X = np.column_stack([np.ones_like(D), D ** (-bt)])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ coef
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if best is None or r2 > best[3]:
                best = (float(coef[0]), float(coef[1]), float(bt), r2)
        return best
    X = np.column_stack([np.ones_like(D), D ** (-beta)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return (float(coef[0]), float(coef[1]), float(beta),
            1 - ss_res / ss_tot if ss_tot > 0 else 0.0)


def local_exponents(rungs, *, beta: float | None = None, use_asymptote: bool = True):
    """Local exponent between adjacent rungs, for a grid of E values.

    ``rungs`` is a list of (name, N, value) sorted by N, where ``value`` is
    either the rung's raw final loss (ascending-D raw) or its fitted asymptote
    c (D-term removed).
    """
    out = []
    for i in range(len(rungs) - 1):
        n1, v1 = rungs[i][1], rungs[i][2]
        n2, v2 = rungs[i + 1][1], rungs[i + 1][2]
        dlnN = math.log(n2 / n1)
        slope = (v2 - v1) / dlnN          # dL/dlnN
        row = {"from": rungs[i][0], "to": rungs[i + 1][0],
               "N_from": int(n1), "N_to": int(n2), "dL_dlnN": slope,
               "L_mean": (v1 + v2) / 2.0}
        for E in (1.3480, 0.7060):
            row[f"alpha_loc_E{E:.3f}"] = (-slope) / ((v1 + v2) / 2.0 - E) \
                if abs((v1 + v2) / 2.0 - E) > 1e-9 else float("nan")
        out.append(row)
    return out


def joint_profile(points, *, e_grid=None, delta_r2: float = 0.001) -> dict:
    """The JOINT (alpha, E) profile: alpha conditional on E, and whether the
    profile's width in alpha is a RIDGE in E-alpha space rather than noise.

    For each E on a grid, alpha (with A, B, beta) is optimised. If the
    best-fitting alpha moves systematically with E, then the wide alpha
    profile is a degeneracy: the two parameters trade off, and the profile
    traces a ridge rather than a flat basin.
    """
    from src.scaling.refit_clean import fit_scaling

    if e_grid is None:
        e_grid = [round(float(x), 4) for x in np.arange(0.40, 2.56, 0.05)]
    N = np.array([p[0] for p in points])
    D = np.array([p[1] for p in points])
    y = np.array([p[2] for p in points])

    curve = []
    for E in e_grid:
        # Fit with E fixed: optimise A, alpha, B, beta. The refit machinery
        # takes a full 5-vector, so the E coordinate is held by re-fitting the
        # remaining four and reading alpha back.
        best = None
        for a0 in (0.15, 0.25, 0.35, 0.5):
            for b0 in (0.2, 0.35, 0.5):
                p, _ = _fit_with_fixed_E(N, D, y, E, a0, b0)
                sse = float(np.sum((y - _model(N, D, p)) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, p)
        sse, p = best
        pred = _model(N, D, p)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - sse / ss_tot if ss_tot > 0 else 0.0
        curve.append({"E": E, "alpha": float(p[2]), "beta": float(p[4]),
                      "A": float(p[1]), "B": float(p[3]), "r2": r2, "sse": sse})

    best_r2 = max(c["r2"] for c in curve)
    best_alpha = max(curve, key=lambda c: c["r2"])["alpha"]

    # The ridge: how the best-fitting alpha moves with E.
    ridge = []
    for a, b in zip(curve, curve[1:]):
        if b["E"] > a["E"]:
            ridge.append((b["alpha"] - a["alpha"]) / (b["E"] - a["E"]))
    ridge_slope = float(np.mean(ridge)) if ridge else None

    # Conditional on E being admissible for this corpus.
    admissible = [c for c in curve if c["E"] <= 2.5152]
    ok = [c["alpha"] for c in admissible if c["r2"] >= best_r2 - delta_r2]
    ok_all = [c["alpha"] for c in curve if c["r2"] >= best_r2 - delta_r2]

    return {
        "e_grid": e_grid,
        "curve": [{k: (round(v, 5) if isinstance(v, float) else v)
                   for k, v in c.items()} for c in curve],
        "best_r2": round(best_r2, 6),
        "best_alpha_at_best_E": round(best_alpha, 4),
        "ridge_dalpha_dE": round(ridge_slope, 4) if ridge_slope is not None else None,
        "is_a_ridge": bool(ridge_slope is not None and abs(ridge_slope) > 0.2),
        "unconditional_alpha_range": [round(min(ok_all), 4), round(max(ok_all), 4)]
        if ok_all else None,
        "admissible_E_max": 2.5152,
        "admissible_E_basis": ("E is the irreducible loss of an INFINITE model, so it must be "
                               "below the loss a real model reaches: the canonical 6.3M's "
                               "held-out loss is 2.5152 nats/token, and the token-space "
                               "compression ceiling is 3.5918 nats/token."),
        "conditional_alpha_range_given_admissible_E": [round(min(ok), 4), round(max(ok), 4)]
        if ok else None,
        "conditional_contains_0_34": bool(ok and min(ok) <= 0.34 <= max(ok)),
        "note": ("the unconditional alpha range is wide because alpha and E trade off. The "
                 "ridge_dalpha_dE says how fast the best-fitting alpha moves with E. With E "
                 "constrained to what the corpus admits, the conditional alpha range is the "
                 "honest interval to quote."),
    }


def _model(N, D, p):
    E, A, alpha, B, beta = p
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def _fit_with_fixed_E(N, D, y, E, a0, b0, *, iters=300):
    """Optimise (A, alpha, B, beta) with E held fixed."""
    import math

    def model(z):
        return (E + math.exp(z[0]) / np.power(N, math.exp(z[1]))
                + math.exp(z[2]) / np.power(D, math.exp(z[3])))

    def jac(z):
        A = math.exp(z[0]); alpha = math.exp(z[1])
        B = math.exp(z[2]); beta = math.exp(z[3])
        Np = np.power(N, alpha); Dp = np.power(D, beta)
        J = np.zeros((len(N), 4))
        J[:, 0] = A / Np
        J[:, 1] = -A * np.log(N) / Np
        J[:, 2] = B / Dp
        J[:, 3] = -B * np.log(D) / Dp
        return J

    span = float(y.max() - y.min())
    z = np.array([math.log(span * float(N.max())), math.log(a0),
                  math.log(span * float(D.max())), math.log(b0)])
    r = model(z) - y
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(iters):
        J = jac(z)
        JTJ = J.T @ J
        JTr = J.T @ r
        improved = False
        for _ in range(40):
            Am = JTJ + lam * np.diag(np.maximum(np.diag(JTJ), 1e-12))
            try:
                dz = np.linalg.solve(Am, -JTr)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            q = z + dz
            q[0] = min(max(q[0], math.log(1e-3)), math.log(1e9))
            q[1] = min(max(q[1], math.log(0.01)), math.log(3.0))
            q[2] = min(max(q[2], math.log(1e-3)), math.log(1e9))
            q[3] = min(max(q[3], math.log(0.01)), math.log(3.0))
            rq = model(q) - y
            cq = float(rq @ rq)
            if np.isfinite(cq) and cq < cost:
                z, r, cost = q, rq, cq
                lam = max(lam * 0.3, 1e-14)
                improved = True
                break
            lam *= 10
        if not improved or cost < 1e-14:
            break
    return (np.array([E, math.exp(z[0]), math.exp(z[1]),
                       math.exp(z[2]), math.exp(z[3])]), cost)




def _pick(rows, E: float) -> dict:
    """The row of a grid closest to a given E."""
    best = min(rows, key=lambda r: abs(r["E"] - E))
    return {"E": best["E"],
            "alpha_loc_mean": round(best["alpha_loc_mean"], 4),
            "alpha_loc_sd": round(best["alpha_loc_sd"], 4)}

def _pick(rows, E: float) -> dict:
    """The row of a grid closest to a given E."""
    best = min(rows, key=lambda r: abs(r["E"] - E))
    return {"E": best["E"],
            "alpha_loc_mean": round(best["alpha_loc_mean"], 4),
            "alpha_loc_sd": round(best["alpha_loc_sd"], 4)}


def select_E_by_flatness(rungs, *, e_grid=None) -> dict:
    """Select E by requiring the ladder's LOCAL exponent to be constant.

    A power law has a constant local exponent. alpha_loc(N) = -[dL/dlnN]/(L-E)
    depends on E, so requiring alpha_loc to be FLAT across the ladder selects E
    (and alpha with it). This is a data-driven estimate of E that does not use
    the five-parameter fit at all, and it is the cleanest way to say what the
    ladder actually says.

    Two estimators are reported, because they are different quantities: the
    ascending-D-raw ladder (the D-term is still present) and the D-term-removed
    one (each rung's asymptote), which is the one that isolates alpha.
    """
    if e_grid is None:
        e_grid = [round(float(x), 4) for x in np.arange(0.20, 2.61, 0.02)]
    out = {}
    for label, curve in (("ascending_D_raw", rungs[0]),
                         ("d_term_removed", rungs[1])):
        rows = []
        for E in e_grid:
            vals = []
            for i in range(len(curve) - 1):
                n1, v1 = curve[i][1], curve[i][2]
                n2, v2 = curve[i + 1][1], curve[i + 1][2]
                slope = (v2 - v1) / math.log(n2 / n1)
                denom = (v1 + v2) / 2.0 - E
                if abs(denom) < 1e-6:
                    continue
                vals.append(-slope / denom)
            if not vals:
                continue
            rows.append({
                "E": E,
                "alpha_loc_mean": float(np.mean(vals)),
                "alpha_loc_sd": (float(np.std(vals, ddof=1))
                                 if len(vals) > 1 else 0.0),
                "alpha_loc_values": [round(v, 4) for v in vals],
            })
        best = min(rows, key=lambda r: r["alpha_loc_sd"])
        lo = best["alpha_loc_mean"] - 2 * best["alpha_loc_sd"]
        hi = best["alpha_loc_mean"] + 2 * best["alpha_loc_sd"]
        out[label] = {
            "criterion": ("the E that minimises the standard deviation of the "
                           "local exponent across the ladder"),
            "E_selected": best["E"],
            "alpha_loc_mean": round(best["alpha_loc_mean"], 4),
            "alpha_loc_sd": round(best["alpha_loc_sd"], 4),
            "alpha_loc_values": best["alpha_loc_values"],
            "contains_0_34": bool(lo <= 0.34 <= hi),
            "at_E_1_348": _pick(rows, 1.348),
            "at_E_0_706": _pick(rows, 0.706),
            "curve": [{k: (round(v, 5) if isinstance(v, float) else v)
                       for k, v in r.items()} for r in rows],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(RESULTS / "local_exponent.json"))
    args = ap.parse_args()

    runs = load_canonical(RESULTS / "ladder_points.csv")
    # Aggregate to rungs. A rung is a parameter count, so the off-line arm
    # (the same N at a different D/N ratio) POOLS into its rung: both arms
    # describe the same model, and pooling them widens the D range over which
    # the rung's asymptote is fitted, which is exactly what that estimator
    # needs. The raw ladder, by contrast, is the D = 20N LINE, so it takes
    # only the line's arms.
    by_rung = defaultdict(list)
    by_rung_line = defaultdict(list)
    for (rung, seed, arm), pts in runs.items():
        by_rung[rung].append(pts)
        if arm in ("D20N", "ladder"):
            by_rung_line[rung].append(pts)
    order = sorted(by_rung, key=lambda r: float(r.rstrip("M")))

    raw, asym, meta = [], [], {}
    for rung in order:
        finals, csum, bsum, betas = [], [], [], []
        for pts in by_rung[rung]:
            c, b, bt, r2 = fit_rung_asymptote(pts)
            csum.append(c)
            bsum.append(b)
            betas.append(bt)
        for pts in by_rung_line.get(rung, []):
            finals.append(pts[-1][1])
        N = by_rung[rung][0][0][2]
        raw.append((rung, N, float(np.mean(finals)) if finals else float("nan")))
        asym.append((rung, N, float(np.mean(csum))))
        meta[rung] = {"N": int(N), "n_runs": len(by_rung[rung]),
                      "n_runs_on_the_D20N_line": len(by_rung_line.get(rung, [])),
                      "final_loss_mean": round(float(np.mean(finals)), 4) if finals else None,
                      "asymptote_c_mean": round(float(np.mean(csum)), 4),
                      "b_mean": round(float(np.mean(bsum)), 4),
                      "beta_per_run": [round(x, 4) for x in betas],
                      "D_over_N": round(by_rung[rung][0][-1][0] / N, 3)}

    loc_raw = local_exponents(raw, use_asymptote=False)
    loc_asym = local_exponents(asym, use_asymptote=True)

    def stats(rows, key):
        vals = [r[key] for r in rows if not math.isnan(r[key])]
        if not vals:
            return {}
        return {"mean": round(float(np.mean(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "spread": round(float(np.max(vals) - np.min(vals)), 4),
                "values": [round(v, 4) for v in vals]}

    # The joint (alpha, E) profile needs the point set on the TOTAL-N axis, so
    # the per-run total parameter counts are needed. Both the role and the
    # total parameter count are columns of the point file, so neither has to
    # be recovered by globbing the artefact tree.
    import csv as _csv
    pts_total = []
    with open(RESULTS / "ladder_points.csv", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            if (row.get("role") or "").strip() != "canonical_ladder":
                continue
            loss = float(row["validation_loss"])
            final = float(row.get("final_val_loss") or 0.0)
            if not (final and loss <= final + TRANSIENT_MARGIN):
                continue
            t = float(row.get("total_params") or 0.0)
            if t:
                pts_total.append((t, float(row["training_tokens"]),
                                  float(row["validation_loss"])))
    joint = joint_profile(pts_total)
    flat = select_E_by_flatness((raw, asym))

    out = {
        "act": "4",
        "joint_alpha_E_profile": joint,
        "E_selected_by_local_exponent_flatness": flat,
        "quantity": "local exponent between adjacent rungs, alpha_loc = -[dL/dlnN]/(L-E)",
        "why": ("a wide alpha profile in a five-parameter fit can be an (E, alpha) "
                "degeneracy rather than data uncertainty. If the ladder is a power "
                "law, the local exponent is a constant equal to alpha, and whether "
                "it IS constant depends on E."),
        "per_rung": meta,
        "ascending_D_raw": {
            "note": ("the slope is taken across the rungs' FINAL validation losses, so the "
                     "B/D^beta term is still present. This quantity is a weighted average "
                     "of alpha and beta: it overstates alpha where the D-term is small "
                     "and is dragged toward beta at the top of the ladder."),
            "rows": [{k: (round(v, 5) if isinstance(v, float) else v)
                      for k, v in r.items()} for r in loc_raw],
            "alpha_loc_E1.348": stats(loc_raw, "alpha_loc_E1.348"),
            "alpha_loc_E0.706": stats(loc_raw, "alpha_loc_E0.706"),
        },
        "d_term_removed": {
            "note": ("each rung's own interior points are fitted with L_i(D) = c_i + b_i "
                     "D^-beta (beta fitted per run), so c_i is the rung's asymptote and the "
                     "D-term is removed BEFORE the slope is taken."),
            "rows": [{k: (round(v, 5) if isinstance(v, float) else v)
                      for k, v in r.items()} for r in loc_asym],
            "alpha_loc_E1.348": stats(loc_asym, "alpha_loc_E1.348"),
            "alpha_loc_E0.706": stats(loc_asym, "alpha_loc_E0.706"),
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("=== LOCAL EXPONENT, ascending-D raw (the D-term is still present) ===")
    print(f"{'interval':<20}{'dL/dlnN':>10}{'a_loc E=1.348':>16}{'a_loc E=0.706':>16}")
    for r in loc_raw:
        print(f"{r['from']+' -> '+r['to']:<20}{r['dL_dlnN']:>10.4f}"
              f"{r['alpha_loc_E1.348']:>16.4f}{r['alpha_loc_E0.706']:>16.4f}")
    print("  spread: E=1.348 ->",
          out["ascending_D_raw"]["alpha_loc_E1.348"]["spread"],
          " E=0.706 ->", out["ascending_D_raw"]["alpha_loc_E0.706"]["spread"])
    print("\n=== LOCAL EXPONENT, D-term removed (rung asymptotes) ===")
    print(f"{'interval':<20}{'dL/dlnN':>10}{'a_loc E=1.348':>16}{'a_loc E=0.706':>16}")
    for r in loc_asym:
        print(f"{r['from']+' -> '+r['to']:<20}{r['dL_dlnN']:>10.4f}"
              f"{r['alpha_loc_E1.348']:>16.4f}{r['alpha_loc_E0.706']:>16.4f}")
    print("  spread: E=1.348 ->",
          out["d_term_removed"]["alpha_loc_E1.348"]["spread"],
          " E=0.706 ->", out["d_term_removed"]["alpha_loc_E0.706"]["spread"])
    print("\n=== per rung ===")
    for r, m in meta.items():
        print(f"  {r:<6} N={m['N']:>10,}  final={m['final_loss_mean']:.4f}  "
              f"c={m['asymptote_c_mean']:.4f}  D/N={m['D_over_N']}  runs={m['n_runs']}")
    print("\n=== JOINT (alpha, E) PROFILE, total-N axis ===")
    print(f"  best R2 {joint['best_r2']}  best alpha {joint['best_alpha_at_best_E']} "
          f"(spread over the E grid: {joint['unconditional_alpha_range']})")
    print(f"  ridge d(alpha)/dE = {joint['ridge_dalpha_dE']}  -> "
          f"{'a RIDGE: the width in alpha is a degeneracy with E' if joint['is_a_ridge'] else 'not a ridge'}")
    print(f"  conditional on E <= 2.5152 (E must be below the loss a real model "
          f"reaches): alpha in {joint['conditional_alpha_range_given_admissible_E']}  "
          f"contains 0.34: {joint['conditional_contains_0_34']}")
    print("\n=== E SELECTED BY REQUIRING THE LOCAL EXPONENT TO BE FLAT ===")
    for label, f in flat.items():
        print(f"  {label}:")
        print(f"     E selected = {f['E_selected']:.3f}  ->  alpha_loc = "
              f"{f['alpha_loc_mean']:.4f} (sd {f['alpha_loc_sd']:.4f})  "
              f"contains 0.34: {f['contains_0_34']}")
        print(f"     at E=1.348: alpha_loc {f['at_E_1_348']['alpha_loc_mean']:.4f} "
              f"(sd {f['at_E_1_348']['alpha_loc_sd']:.4f})")
        print(f"     at E=0.706: alpha_loc {f['at_E_0_706']['alpha_loc_mean']:.4f} "
              f"(sd {f['at_E_0_706']['alpha_loc_sd']:.4f})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
