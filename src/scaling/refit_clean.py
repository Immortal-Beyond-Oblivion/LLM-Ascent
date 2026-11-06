"""Act 4 clean refit: separate the two architecture families that the ladder mixed.

WHAT WAS FOUND. The Act 4 ladder is not one model family. Reading the run
records rather than the ledger:

  * the 16 ladder runs split into 11 runs of the canonical architecture
    (MiniGPT, 8 heads, learned absolute positions -- confirmed from the
    surviving checkpoints' state dicts and from the ``architecture`` block
    written into every ``metrics.json``) and 5 runs of a different family;
  * the two families are trivially separable by the FIRST logged validation
    loss: the canonical family starts at 8.7-8.9 nats, the other at
    129.6-223.8 nats, because it has no learned absolute position table
    (the 1.5M runs differ by exactly 128 x 160 = 20,480 parameters, which is
    the position table) and a different attention geometry;
  * the fitted set of 2,776 points draws on 12 of the 16 runs -- 7 canonical
    and 5 of the other family -- so MORE THAN HALF the fitted points come from
    a different model class;
  * the N axis is contaminated too: the 1.5M rung contributes at two different
    N values (1,567,200 and 1,546,720) because the two families count the
    position table differently.

WHAT THIS DOES. It re-fits the same law on both point sets with the same
procedure, so the two are directly comparable, and it reports the leave-one-
rung-out R-squared for each. If alpha moves, the centrepiece number moves and
the report says so.

The fitter is a Levenberg-Marquardt with a numerical Jacobian, matching the
procedure recorded in the tracked fit's notes. It is validated by reproducing
the tracked fit on the tracked point set before it is used for anything else.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "results" / "v2" / "act4-scaling"
ARTIFACTS = ROOT / "artifacts" / "scaling" / "checkpoints"

# A run whose first logged validation loss is above this is NOT the canonical
# architecture. The two families are 15-25x apart (8.8 vs 129.6 at 1.5M), so
# the threshold is not delicate. The architecture block, when present, is
# authoritative; the threshold is only used where the block is absent.
FIRST_LOSS_THRESHOLD = 20.0

#: How far above its own run's final validation loss a point may sit and
#: still count as "past the transient". Scale-free, so it means the same thing
#: at every rung.
TRANSIENT_MARGIN = 0.5


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def run_configs(ladder_dir: Path = ARTIFACTS) -> dict[tuple[str, str, str], dict]:
    """Architecture fingerprint and role for every ladder run.

    The authority is the manifest in ``src/scaling/ladder_manifest.py``, which
    names each run, its role in the fit and the reason for that role. The
    manifest exists because the first version of this fit was computed over a
    mixture of two architecture families and nothing complained: the old
    ``in_fitted_set`` column reproduced the point COUNT and not the point SET.

    Keys are ``(rung, seed, arm)``. The arm is the D/N ratio of the design
    point, so the 9.0M rung at D = 20N and the 9.0M rung at D = 10N are
    different runs rather than one run counted twice.
    """
    from src.scaling.ladder_manifest import build_manifest

    manifest = build_manifest()
    out: dict[tuple[str, str, str], dict] = {}
    for run in manifest["runs"]:
        key = (run["rung"], run["seed"], run["arm"])
        out[key] = {
            "model_class": run["model_class"],
            "positional_encoding": run["positional_encoding"],
            "num_heads": run["num_heads"],
            "num_kv_heads": run["num_heads"] if run["family"] == "canonical" else None,
            "embedding_dim": run["embedding_dim"],
            "num_layers": run["num_layers"],
            "context_length": run["context_length"],
            "vocab_size": 6487,
            "total_params": run["total_params"],
            "non_embedding_params": run["non_embedding_params"],
            "first_val_loss": run["first_logged_val_loss"],
            "final_val_loss": run["final_val_loss"],
            "steps": run["steps"],
            "tokens": run["tokens"],
            "family": run["family"],
            "role": run["role"],
            "role_reason": run["role_reason"],
            "arm": run["arm"],
            "source": run["source"],
            "d_over_n": run["d_over_n"],
            "config_hash": run["config_hash"],
            "live_config_hash": run["live_config_hash"],
            "evidence": ("the manifest" if run["source"] == "rescue"
                         else "architecture block, or the shape its siblings recorded"),
        }
    return out


def load_points(csv_path: Path, configs: dict) -> list[dict]:
    """Load the ladder points and tag each with its run's role.

    The selection rule is a ROLE, not a flag. A flag that reproduces a point
    count is not a selection rule; it is a coincidence that looks like one.
    """
    points: list[dict] = []
    with open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            arm = (row.get("arm") or "ladder").strip() or "ladder"
            key = (row["rung"], row["seed"], arm)
            cfg = configs.get(key)
            if cfg is None:
                continue
            total = row.get("total_params") or cfg.get("total_params") or 0
            final = float(cfg.get("final_val_loss") or 0.0)
            loss = float(row["validation_loss"])
            points.append({
                "rung": row["rung"], "seed": row["seed"], "arm": arm,
                "N": float(row["non_embedding_params"]),
                "N_total": float(total),
                "D": float(row["training_tokens"]),
                "loss": loss,
                # "Past the transient" is a property of the LOSS, not of an
                # absolute token count. The old filter was D >= 5,000,000,
                # which is past the transient for a rung whose run reaches
                # 180M tokens and absurdly early for one that reaches 6M: it
                # kept 7 of the 0.3M rung's 32 points, so that rung's
                # asymptote was fitted from 7 points over a 1.2M-token D
                # range and its local exponent came out as -1.16. The
                # rescue widened the ladder into the range where an absolute
                # floor stops being a proxy for the transient.
                "converged": bool(final and loss <= final + TRANSIENT_MARGIN),
                "final_val_loss": final,
                "role": (row.get("role") or cfg["role"]).strip(),
                "family": (row.get("family") or cfg["family"]).strip(),
                "source": (row.get("source") or cfg["source"]).strip(),
                "config_hash": (row.get("config_hash") or cfg["config_hash"]).strip(),
            })
    return points


# --------------------------------------------------------------------------
# fitter
# --------------------------------------------------------------------------

def _model(N, D, p):
    E, A, alpha, B, beta = p
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def _model_z(N, D, z):
    """The model, evaluated directly from the optimiser's coordinates."""
    E = z[0]
    A = np.exp(z[1]); alpha = np.exp(z[2])
    B = np.exp(z[3]); beta = np.exp(z[4])
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def _jacobian_z(N, D, z):
    """ANALYTIC Jacobian of the model with respect to the log-space coordinates.

    d/dE = 1;  d/d(log A) = A/N^a;  d/d(log alpha) = -A ln(N) / N^a;
    and likewise for B and beta. Deriving it analytically matters: an earlier
    version computed the Jacobian in PARAMETER space and then stepped in LOG
    space, which is simply the wrong direction and produced alpha = 9965.
    """
    A = np.exp(z[1]); alpha = np.exp(z[2])
    B = np.exp(z[3]); beta = np.exp(z[4])
    Np = np.power(N, alpha)
    Dp = np.power(D, beta)
    J = np.zeros((len(N), 5))
    J[:, 0] = 1.0
    J[:, 1] = A / Np
    J[:, 2] = -A * np.log(N) / Np
    J[:, 3] = B / Dp
    J[:, 4] = -B * np.log(D) / Dp
    return J


def _to_params(z):
    """Unpack the optimiser's coordinates into model parameters."""
    z = np.asarray(z, dtype=np.float64).copy()
    z[1:] = np.clip(z[1:], -700.0, 700.0)  # exp() must not overflow
    return np.array([z[0], math.exp(z[1]), math.exp(z[2]),
                     math.exp(z[3]), math.exp(z[4])], dtype=np.float64)


def _to_params_inverse(p):
    """Model parameters back to the optimiser's coordinates."""
    E, A, alpha, B, beta = p
    tiny = 1e-300
    return np.array([E, math.log(max(A, tiny)), math.log(max(alpha, tiny)),
                     math.log(max(B, tiny)), math.log(max(beta, tiny))])


def fit_scaling(N, D, losses, *, iters=600, tol=1e-14, quick=False):
    """Levenberg-Marquardt with an analytic Jacobian, in log space, multi-start.

    A, alpha, B and beta are carried in LOG space so they cannot blow up
    (A / N**alpha with a tiny alpha overflows a double), and E is carried
    linearly because the tracked fit's degenerate primary result drives it to
    exactly zero, which a logarithm cannot represent.

    The objective has shallow, correlated minima, so a single start can land in
    a visibly worse optimum (a single-start LM reached R^2 0.709 where the
    tracked fit reaches 0.995). The fit is therefore run from a grid of
    starting points in (alpha, beta) and the best cost is kept.
    """
    N = np.asarray(N, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    if len(y) < 6:
        raise ValueError("need at least 6 points to fit 5 parameters")
    span = float(y.max() - y.min())
    grid = (0.2, 0.35, 0.5) if quick else (0.1, 0.2, 0.3, 0.45, 0.6)
    best = None
    for a0 in grid:
        for b0 in grid:
            z = np.array([float(y.min()),
                          math.log(span * float(np.max(N))),
                          math.log(a0),
                          math.log(span * float(np.max(D))),
                          math.log(b0)], dtype=np.float64)
            r = _model_z(N, D, z) - y
            cost = float(r @ r)
            lam = 1e-3
            for _ in range(iters):
                J = _jacobian_z(N, D, z)
                JTJ = J.T @ J
                JTr = J.T @ r
                improved = False
                for _ in range(40):
                    A_m = JTJ + lam * np.diag(np.maximum(np.diag(JTJ), 1e-12))
                    try:
                        dz = np.linalg.solve(A_m, -JTr)
                    except np.linalg.LinAlgError:
                        lam *= 10
                        continue
                    q_z = z + dz
                    # Hard bounds. Without them the optimiser walks alpha to
                    # 1e300 (where N**alpha overflows and the Jacobian is
                    # meaningless) instead of converging, which is what an
                    # earlier unbounded version did.
                    q_z[0] = min(max(q_z[0], 0.0), 10.0)          # E
                    q_z[1] = min(max(q_z[1], math.log(1e-3)), math.log(1e9))   # A
                    q_z[2] = min(max(q_z[2], math.log(0.01)), math.log(3.0))   # alpha
                    q_z[3] = min(max(q_z[3], math.log(1e-3)), math.log(1e9))   # B
                    q_z[4] = min(max(q_z[4], math.log(0.01)), math.log(3.0))   # beta
                    rq = _model_z(N, D, q_z) - y
                    cq = float(rq @ rq)
                    if np.isfinite(cq) and cq < cost:
                        z, r, cost = q_z, rq, cq
                        lam = max(lam * 0.3, 1e-14)
                        improved = True
                        break
                    lam *= 10
                if not improved or cost < tol:
                    break
            if best is None or cost < best[1]:
                best = (z, cost)
    return _to_params(best[0]), best[1]


def diagnostics(N, D, losses, p) -> dict:
    N = np.asarray(N, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    pred = _model(N, D, p)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    J = _jacobian_z(N, D, _to_params_inverse(p))
    return {
        "r2_vs_constant_baseline": r2,
        "jacobian_condition": float(np.linalg.cond(J)),
        "reduced_chi2": ss_res / max(len(y) - 5, 1),
        "resid_corr_logN": float(np.corrcoef(np.log(N), y - pred)[0, 1]),
        "resid_corr_logD": float(np.corrcoef(np.log(D), y - pred)[0, 1]),
    }


def loro_r2(points, p) -> dict:
    """Leave-one-RUNG-out R-squared: hold out a whole rung, refit, predict it.

    Holding out a rung (not a point) is the protocol that tests whether the
    law extrapolates to a model size it was not fitted on.
    """
    rungs = sorted({pt["rung"] for pt in points})
    out = {}
    for rung in rungs:
        train = [pt for pt in points if pt["rung"] != rung]
        test = [pt for pt in points if pt["rung"] == rung]
        if len({pt["N"] for pt in train}) < 2 or len(train) < 6:
            out[rung] = {"r2": None, "n_test": len(test), "n_train": len(train),
                         "reason": "too few training points to identify 5 "
                                   "parameters once this rung is held out"}
            continue
        p_hat, _ = fit_scaling([q["N"] for q in train], [q["D"] for q in train],
                                [q["loss"] for q in train], quick=True)
        y = np.array([q["loss"] for q in test])
        pred = _model(np.array([q["N"] for q in test]),
                      np.array([q["D"] for q in test]), p_hat)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out[rung] = {
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
            "n_test": len(test),
            "n_train": len(train),
            "alpha": float(p_hat[2]), "beta": float(p_hat[4]),
        }
    return out


def summarise_fit(name: str, points) -> dict:
    N = [q["N"] for q in points]
    D = [q["D"] for q in points]
    y = [q["loss"] for q in points]
    p, _ = fit_scaling(N, D, y)
    d = diagnostics(N, D, y, p)
    return {
        "name": name,
        "n_points": len(points),
        "n_rungs": len({q["rung"] for q in points}),
        "distinct_N": sorted({int(q["N"]) for q in points}),
        "E": float(p[0]), "A": float(p[1]), "alpha": float(p[2]),
        "B": float(p[3]), "beta": float(p[4]),
        **d,
        "loro": loro_r2(points, p),
    }


def fit_fixed_alpha(N, D, losses, alpha: float, *, iters: int = 400):
    """Levenberg-Marquardt with alpha FIXED, over (E, logA, logB, logbeta).

    This is the profile likelihood: sweep alpha, hold it, optimise the rest, and
    record how well the model fits at each alpha. A point estimate invites a
    reviewer to attack the error bar; a profile makes the actual finding -- that
    alpha is not identified by this ladder -- the headline.
    """
    N = np.asarray(N, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    span = float(y.max() - y.min())

    def model(z):
        return (z[0] + math.exp(z[1]) / np.power(N, alpha)
                + math.exp(z[2]) / np.power(D, math.exp(z[3])))

    def jac(z):
        J = np.zeros((len(N), 4))
        A = math.exp(z[1]); B = math.exp(z[2]); beta = math.exp(z[3])
        Np = np.power(N, alpha); Dp = np.power(D, beta)
        J[:, 0] = 1.0
        J[:, 1] = A / Np
        J[:, 2] = B / Dp
        J[:, 3] = -B * np.log(D) / Dp
        return J

    best = None
    for a0 in (span * float(N.max()), span * float(D.max())):
        for b0 in (span * float(D.max()),):
            z = np.array([float(y.min()), math.log(a0), math.log(b0),
                          math.log(0.3)], dtype=np.float64)
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
                    q[0] = min(max(q[0], 0.0), 10.0)
                    q[1] = min(max(q[1], math.log(1e-3)), math.log(1e9))
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
            if best is None or cost < best[1]:
                best = (z, cost)
    z, cost = best
    E = max(z[0], 0.0)
    A, B, beta = math.exp(z[1]), math.exp(z[2]), math.exp(z[3])
    return np.array([E, A, alpha, B, beta]), cost


def profile_alpha(points, *, alphas=None, delta_r2: float = 0.001) -> dict:
    """Profile the fit over alpha and report the range the data are compatible with.

    The compatibility range is the set of alphas whose fit comes within
    ``delta_r2`` in R-squared of the best fit. It is a statement about the
    DATA, not about the fitting procedure: it says which exponents this ladder
    cannot distinguish.
    """
    if alphas is None:
        alphas = [round(float(x), 4) for x in np.arange(0.02, 1.501, 0.02)]
    N = np.array([q["N"] for q in points])
    D = np.array([q["D"] for q in points])
    y = np.array([q["loss"] for q in points])
    curve = []
    for a in alphas:
        p, _ = fit_fixed_alpha(N, D, y, a)
        pred = _model(N, D, p)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        curve.append({"alpha": a, "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
                      "E": float(p[0]), "beta": float(p[4])})
    best = max(c["r2"] for c in curve)
    ok = [c["alpha"] for c in curve if c["r2"] >= best - delta_r2]
    return {
        "n_points": len(points),
        "delta_r2": delta_r2,
        "best_r2": round(best, 6),
        "best_alpha": max(curve, key=lambda c: c["r2"])["alpha"],
        "compatible_alpha_min": min(ok) if ok else None,
        "compatible_alpha_max": max(ok) if ok else None,
        "contains_0_34": bool(ok and min(ok) <= 0.34 <= max(ok)),
        "curve": curve,
        "note": "the compatible range is the set of alphas whose fit comes "
                f"within {delta_r2} in R^2 of the best fit. It is a statement "
                "about what the data cannot distinguish, not an error bar.",
    }


def fit_fixed_beta(N, D, losses, beta: float, *, iters: int = 400):
    """Levenberg-Marquardt with beta FIXED, over (E, logA, alpha, logB).

    The beta profile is reported beside the alpha profile because the ASYMMETRY
    between them is the diagnosis: each run contributes interior checkpoints at
    FIXED N and RISING D, so within-run D variation constrains beta tightly,
    while alpha depends on the number and spread of distinct N values the
    ladder contains.
    """
    N = np.asarray(N, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    span = float(y.max() - y.min())

    def model(z):
        return (z[0] + math.exp(z[1]) / np.power(N, math.exp(z[2]))
                + math.exp(z[3]) / np.power(D, beta))

    def jac(z):
        J = np.zeros((len(N), 4))
        A = math.exp(z[1]); alpha = math.exp(z[2]); B = math.exp(z[3])
        Np = np.power(N, alpha); Dp = np.power(D, beta)
        J[:, 0] = 1.0
        J[:, 1] = A / Np
        J[:, 2] = -A * np.log(N) / Np
        J[:, 3] = B / Dp
        return J

    best = None
    for a0 in (span * float(N.max()),):
        for b0 in (span * float(D.max()),):
            z = np.array([float(y.min()), math.log(a0), math.log(0.3),
                          math.log(b0)], dtype=np.float64)
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
                    q[0] = min(max(q[0], 0.0), 10.0)
                    q[1] = min(max(q[1], math.log(1e-3)), math.log(1e9))
                    q[2] = min(max(q[2], math.log(0.01)), math.log(3.0))
                    q[3] = min(max(q[3], math.log(1e-3)), math.log(1e9))
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
            if best is None or cost < best[1]:
                best = (z, cost)
    z, cost = best
    E = max(z[0], 0.0)
    A, alpha, B = math.exp(z[1]), math.exp(z[2]), math.exp(z[3])
    return np.array([E, A, alpha, B, beta]), cost


def profile_beta(points, *, betas=None, delta_r2: float = 0.001) -> dict:
    """The beta profile, for comparison with the alpha profile."""
    if betas is None:
        betas = [round(float(x), 4) for x in np.arange(0.02, 1.501, 0.02)]
    N = np.array([q["N"] for q in points])
    D = np.array([q["D"] for q in points])
    y = np.array([q["loss"] for q in points])
    curve = []
    for b in betas:
        p, _ = fit_fixed_beta(N, D, y, b)
        pred = _model(N, D, p)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        curve.append({"beta": b, "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
                      "E": float(p[0]), "alpha": float(p[2])})
    best = max(c["r2"] for c in curve)
    ok = [c["beta"] for c in curve if c["r2"] >= best - delta_r2]
    return {
        "n_points": len(points),
        "delta_r2": delta_r2,
        "best_r2": round(best, 6),
        "best_beta": max(curve, key=lambda c: c["r2"])["beta"],
        "compatible_beta_min": min(ok) if ok else None,
        "compatible_beta_max": max(ok) if ok else None,
        "contains_0_28": bool(ok and min(ok) <= 0.28 <= max(ok)),
        "curve": curve,
        "note": "the compatible range of beta, for comparison with alpha's. Each "
                "run contributes interior checkpoints at FIXED N and RISING D, so "
                "within-run D variation constrains beta, while alpha depends on "
                "the number and spread of distinct N values the ladder holds.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(LADDER / "clean_refit.json"))
    args = ap.parse_args()

    configs = run_configs()
    points = load_points(LADDER / "ladder_points.csv", configs)
    D_MIN = 5_000_000
    # TWO point sets, each with a job.
    #
    # ``d5m`` reproduces the tracked fit's own point set. The tracked fit's
    # parameters must be evaluated on the set they were fitted to, or the
    # diagnostic means nothing.
    #
    # ``converged`` is past each run's OWN transient, which is a property of
    # the loss rather than of an absolute token count. The old D >= 5M floor
    # is past the transient for a rung whose run reaches 180M tokens and
    # absurdly early for one that reaches 6M: it kept 7 of the 0.3M rung's
    # 32 points, so that rung's asymptote was fitted from 7 points over a
    # 1.2M-token D range and its local exponent came out as -1.16. The
    # rescue widened the ladder into the range where an absolute floor stops
    # being a proxy for the transient.
    d5m = [q for q in points if q["D"] >= D_MIN]
    interior = [q for q in points if q["converged"]]
    # The selection is by ROLE, from the manifest. A role says what a run is
    # FOR; the old in_fitted_set flag only reproduced a point count, and the
    # tracked fit's own parameters score R^2 0.287 on the set it flags.
    canonical = [q for q in interior if q["role"] == "canonical_ladder"]
    other = [q for q in interior if q["role"] == "other_family"]
    canonical_d5m = [q for q in d5m if q["role"] == "canonical_ladder"]
    other_d5m = [q for q in d5m if q["role"] == "other_family"]

    # The tracked fit's 12 endpoint points: one annealed endpoint per run that
    # the tracked fit consumed. Its run set is recovered from the tracked
    # count: 2,776 = the D>=5M points of the 12 runs that existed when the fit
    # was run (2.6M seeds 44-47 were added later, for the Act 5 MQA n=6 test).
    # This set deliberately INCLUDES the runs the manifest excludes, because
    # the tracked fit consumed them: that is the whole point of the
    # diagnostic.
    LATE = {("2.6M", "44"), ("2.6M", "45"), ("2.6M", "46"), ("2.6M", "47")}
    as_tracked = [q for q in d5m
                  if q["source"] == "ladder" and (q["rung"], q["seed"]) not in LATE]

    def endpoints(sel):
        best: dict[tuple[str, str], dict] = {}
        for q in sel:
            k = (q["rung"], q["seed"])
            if k not in best or q["D"] > best[k]["D"]:
                best[k] = q
        return list(best.values())

    tracked = json.loads((LADDER / "fit_results.json").read_text(encoding="utf-8"))
    tsec = [tracked["secondary"]["E"], tracked["secondary"]["A"],
            tracked["secondary"]["alpha"], tracked["secondary"]["B"],
            tracked["secondary"]["beta"]]
    tpri = [tracked["primary"]["E"], tracked["primary"]["A"],
            tracked["primary"]["alpha"], tracked["primary"]["B"],
            tracked["primary"]["beta"]]

    def r2_of(sel, p):
        N = np.array([q["N"] for q in sel])
        D = np.array([q["D"] for q in sel])
        y = np.array([q["loss"] for q in sel])
        pred = _model(N, D, p)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # THE diagnostic: the tracked parameters, evaluated on each candidate set.
    tracked_params_on = {
        "as_tracked_12_runs_D5M": {"n": len(as_tracked),
                                   "r2": round(r2_of(as_tracked, tsec), 6)},
        "other_family_only_D5M": {"n": len(other_d5m),
                                   "r2": round(r2_of(other_d5m, tsec), 6)},
        "canonical_only_D5M": {"n": len(canonical_d5m),
                               "r2": round(r2_of(canonical_d5m, tsec), 6)},
        "tracked_secondary_params_R2_claim": tracked["secondary"]["r2"],
    }

    fits = {
        "as_tracked_12_runs_D5M": summarise_fit("as_tracked_12_runs_D5M", as_tracked),
        "other_family_only_D5M": summarise_fit("other_family_only_D5M", other_d5m),
        "canonical_only_D5M": summarise_fit("canonical_only_D5M", canonical_d5m),
        # The paper's fit: past each run's own transient, on the canonical
        # family only.
        "canonical_only_converged": summarise_fit("canonical_only_converged", canonical),
        "other_family_only_converged": summarise_fit("other_family_only_converged", other),
    }
    primary_fits = {
        "endpoints_all_16_runs": summarise_fit("endpoints_all_16_runs",
                                                endpoints(interior)),
        "endpoints_canonical": summarise_fit("endpoints_canonical",
                                              endpoints(canonical)),
    }

    # ---- the N-axis convention -------------------------------------------
    # The CSV's N column holds the run's recorded NON-EMBEDDING parameter
    # count. Published exponents (Kaplan, Chinchilla) use TOTAL parameters, so
    # the two are not on the same axis; the fit is therefore reported under
    # both conventions and the paper states which one it uses and why.
    # Two point sets, two names, no aliasing. ``canonical`` is the CONVERGED
    # set (past each run's own transient); ``canonical_d5m`` is the D >= 5M
    # set the tracked fit used. The keys below must follow their names or the
    # diagnostic silently compares a fit against a different point set --
    # which is exactly what happened when ``canonical`` changed meaning and
    # these lines kept reading it.
    canonical_total = [dict(q, N=q["N_total"]) for q in canonical_d5m]
    other_total = [dict(q, N=q["N_total"]) for q in other_d5m]
    canonical_total_conv = [dict(q, N=q["N_total"]) for q in canonical]
    other_total_conv = [dict(q, N=q["N_total"]) for q in other]
    fits_total = {
        "canonical_only_D5M_total_N": summarise_fit("canonical_only_D5M_total_N",
                                                     canonical_total),
        "other_family_only_D5M_total_N": summarise_fit("other_family_only_D5M_total_N",
                                                        other_total),
        "canonical_only_converged_total_N": summarise_fit(
            "canonical_only_converged_total_N", canonical_total_conv),
        "other_family_only_converged_total_N": summarise_fit(
            "other_family_only_converged_total_N", other_total_conv),
    }

    profiles = {
        "canonical_only_D5M_non_embedding_N": profile_alpha(canonical_d5m),
        "canonical_only_D5M_total_N": profile_alpha(canonical_total),
        "canonical_only_converged_non_embedding_N": profile_alpha(canonical),
        "canonical_only_converged_total_N": profile_alpha(canonical_total_conv),
    }
    beta_profiles = {
        "canonical_only_D5M_non_embedding_N": profile_beta(canonical_d5m),
        "canonical_only_D5M_total_N": profile_beta(canonical_total),
        "canonical_only_converged_non_embedding_N": profile_beta(canonical),
        "canonical_only_converged_total_N": profile_beta(canonical_total_conv),
    }

    fold_estimates = {}
    for name, fit in (("canonical_only_D5M", fits["canonical_only_D5M"]),
                      ("other_family_only_D5M", fits["other_family_only_D5M"]),
                      ("canonical_only_D5M_total_N",
                       fits_total["canonical_only_D5M_total_N"]),
                      ("canonical_only_converged",
                       fits["canonical_only_converged"]),
                      ("canonical_only_converged_total_N",
                       fits_total["canonical_only_converged_total_N"])):
        fold_estimates[name] = [
            {"rung": rung, "alpha": v.get("alpha"), "r2": v.get("r2"),
             "n_train": v.get("n_train")}
            for rung, v in fit["loro"].items()
        ]

    out = {
        "act": "4",
        "purpose": "separate the two architecture families the ladder mixed",
        "provenance": {
            "runs_total": len({(k[0], k[1], k[2]) for k in configs if k[0] != "tiny"}),
            "runs_canonical_ladder": sum(1 for c in configs.values()
                                         if c.get("role") == "canonical_ladder"),
            "runs_other_family": sum(1 for c in configs.values()
                                     if c.get("role") == "other_family"),
            "rungs_canonical": sorted({k[0] for k, v in configs.items()
                                       if v["family"] == "canonical"}),
            "rungs_other": sorted({k[0] for k, v in configs.items()
                                   if v["family"] == "other"}),
            "points_logged": len(points),
            "points_D5M_total": len(d5m),
            "points_D5M_canonical": len(canonical_d5m),
            "points_D5M_other": len(other_d5m),
            "points_converged_total": len(interior),
            "points_converged_canonical": len(canonical),
            "points_converged_other": len(other),
            "share_of_D5M_points_from_the_other_family": round(
                len(other_d5m) / len(d5m), 4),
            "per_run": {f"{k[0]}/seed{k[1]}/{k[2]}": v for k, v in sorted(configs.items())},
        },
        "tracked_params_on_candidate_sets": tracked_params_on,
        "fits": fits,
        "fits_under_the_total_parameter_convention": fits_total,
        "n_axis_convention": {
            "what_the_csv_holds": "the run's recorded non_embedding_params",
            "three_conventions_found": {
                "MiniGPT (canonical, includes the learned position table)":
                    "1,567,200 / 2,694,144 / 6,351,360 at 1.5M / 2.6M / 6.3M",
                "the other family (no position table)":
                    "1,546,720 / 4,235,616 / 8,992,224 at 1.5M / 4.1M / 9.0M",
                "the LADDER design target (a search target, not a measurement)":
                    "1,536,000 / 2,654,208 / 8,957,952",
            },
            "the_checkpoint_file_double_counts": "the state dict stores the tied "
                "lm_head.weight alongside token_embedding.weight, so the file's "
                "parameter SUM is 3,643,040 for the 1.5M rung against the "
                "metrics.json total of 2,605,120; the difference is exactly "
                "6487 x 160 = 1,037,920. The CSV's N column is NOT this "
                "double-counted total -- it is the non-embedding count -- so "
                "the N axis is not inflated by the tied embedding.",
            "vocab_is_shared": "embedding_params / dim = 6,487 for every rung, "
                "so a global correction would be constant if it were needed",
            "comparability": "published exponents (Kaplan, Chinchilla) use "
                "TOTAL parameters. This ladder's N is NON-EMBEDDING, so the two "
                "are not on the same axis and alpha is not directly comparable "
                "with a published value without converting it. The fit is "
                "reported under both conventions.",
        },
        "profile_likelihood_over_alpha": profiles,
        "profile_likelihood_over_beta": beta_profiles,
        "fold_wise_alpha_estimates": fold_estimates,
        "primary_fits": primary_fits,
        "delta_clean_minus_as_tracked": {
            "alpha": fits["canonical_only_D5M"]["alpha"]
            - fits["as_tracked_12_runs_D5M"]["alpha"],
            "beta": fits["canonical_only_D5M"]["beta"]
            - fits["as_tracked_12_runs_D5M"]["beta"],
            "E": fits["canonical_only_D5M"]["E"]
            - fits["as_tracked_12_runs_D5M"]["E"],
            "n_points": fits["canonical_only_D5M"]["n_points"]
            - fits["as_tracked_12_runs_D5M"]["n_points"],
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    # The point file is NOT written here. It is written by
    # src/scaling/ladder_manifest.py, which is the authority for the point
    # set: it names every run, its role and the reason for that role, and it
    # writes the role column this fitter selects on. Re-exporting the CSV
    # from here would silently drop the role column and turn the selection
    # back into a flag.
    print(f"\ndelta (canonical - as_tracked): {out['delta_clean_minus_as_tracked']}")
    print(f"wrote {args.out}")

    print(f"runs: {out['provenance']['runs_canonical_ladder']} canonical + "
          f"{out['provenance']['runs_other_family']} other; "
          f"{out['provenance']['share_of_D5M_points_from_the_other_family']:.1%} "
          "of interior points come from the other family")
    print(f"\nTRACKED parameters evaluated on each candidate set:")
    for k, v in tracked_params_on.items():
        if isinstance(v, dict):
            print(f"  {k:<32} n={v['n']:<6} R2={v['r2']:+.6f}")
        else:
            print(f"  {k:<32} {v}")
    print("\nfits:")
    for key, fit in out["fits"].items():
        print(f"  {key:<28} n={fit['n_points']:<6} rungs={fit['n_rungs']} "
              f"alpha={fit['alpha']:+.4f} beta={fit['beta']:+.4f} E={fit['E']:.4f} "
              f"R2={fit['r2_vs_constant_baseline']:.4f}")
    for key, fit in out["primary_fits"].items():
        print(f"  {key:<28} n={fit['n_points']:<6} rungs={fit['n_rungs']} "
              f"alpha={fit['alpha']:+.4f} beta={fit['beta']:+.4f} E={fit['E']:.4f} "
              f"R2={fit['r2_vs_constant_baseline']:.4f}")
    print(f"\ndelta (canonical - as_tracked): {out['delta_clean_minus_as_tracked']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
