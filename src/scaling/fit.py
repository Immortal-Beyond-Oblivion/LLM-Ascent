"""Scaling law fitting: L(N,D) = E + A/N^alpha + B/D^beta.

Fits the Chinchilla-style scaling law to all (N, D, loss) points from the
ladder. Reports uncertainty on alpha and beta from the covariance matrix.
Compares against published Chinchilla alpha=0.34, beta=0.28.

Usage:
    .venv/bin/python -m src.scaling.fit
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_ladder_points(results_path: str = "artifacts/scaling/stage1_results.json") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all (N, D, loss) points from the ladder results."""
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))

    N_vals, D_vals, loss_vals = [], [], []
    for rung_name, rung_runs in results.items():
        for history in rung_runs:
            for entry in history:
                # N is stored in the rung metadata, not per-entry
                # We need to extract it from the rung name or store it
                pass

    # Actually, let me restructure: the ladder results should store N per rung
    # Let me load from the metrics files instead
    return np.array(N_vals), np.array(D_vals), np.array(loss_vals)


def load_points_from_metrics(
    ladder_dir: str = "artifacts/scaling/checkpoints",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Load (N, D, loss) points from per-rung metrics.json files."""
    ladder_dir = Path(ladder_dir)
    N_vals, D_vals, loss_vals = [], [], []
    metadata = []

    for rung_dir in sorted(ladder_dir.iterdir()):
        if not rung_dir.is_dir():
            continue
        metrics_path = rung_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        non_emb_params = m["non_embedding_params"]
        non_emb_params = m["non_embedding_params"]
        for entry in m["history"]:
            N_vals.append(non_emb_params)
            D_vals.append(entry["tokens"])
            loss_vals.append(entry["val_loss"])
            metadata.append({
                "rung": m["rung"],
                "step": entry["step"],
                "tokens_seen": entry["tokens"],
                "val_loss": entry["val_loss"],
            })

    return np.array(N_vals, dtype=np.float64), np.array(D_vals, dtype=np.float64), np.array(loss_vals, dtype=np.float64), metadata


def fit_chinchilla(
    N: np.ndarray,
    D: np.ndarray,
    losses: np.ndarray,
) -> dict:
    """Fit L(N,D) = E + A/N^alpha + B/D^beta by nonlinear least squares.

    Returns the fitted parameters with uncertainties from the covariance matrix.
    """
    from scipy.optimize import curve_fit

    def model(N, D, E, A, alpha, B, beta):
        return E + A / (N ** alpha) + B / (D ** beta)

    # Initial guess: E near min loss, A and B positive, alpha and beta ~0.3
    p0 = [losses.min(), losses.max() * 10, 0.3, losses.max() * 10, 0.3]
    bounds = ([0, 0, 0.01, 0, 0.01], [losses.max(), 1e9, 2.0, 1e9, 2.0])

    popt, pcov = curve_fit(model, (N, D), losses, p0=p0, bounds=bounds, maxfev=50000)
    perr = np.sqrt(np.diag(pcov))

    E, A, alpha, B, beta = popt

    # R-squared
    predictions = model(N, D, *popt)
    ss_res = np.sum((losses - predictions) ** 2)
    ss_tot = np.sum((losses - losses.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Jacobian condition number
    eps = 1e-8
    J = np.zeros((len(losses), 5))
    for i in range(5):
        p_plus = popt.copy()
        p_plus[i] += eps
        J[:, i] = (model(N, D, *p_plus) - predictions) / eps
    from numpy.linalg import cond
    condition = cond(J)

    return {
        "E": float(E), "E_err": float(perr[0]),
        "A": float(A), "A_err": float(perr[1]),
        "alpha": float(alpha), "alpha_err": float(perr[2]),
        "B": float(B), "B_err": float(perr[3]),
        "beta": float(beta), "beta_err": float(perr[4]),
        "r_squared": float(r_squared),
        "jacobian_condition": float(condition),
        "n_points": len(losses),
        "constant_baseline_loss": float(losses.mean()),
        "chinchilla_alpha": 0.34,
        "chinchilla_beta": 0.28,
    }


def main() -> None:
    print("[fit] Loading ladder points ...")
    N, D, losses, metadata = load_points_from_metrics()
    distinct_D = len(set(int(d) for d in D))
    print(f"[fit] Loaded {len(losses)} points from {len(set(m['rung'] for m in metadata))} rungs")
    print(f"[fit] Distinct D values (global): {distinct_D}")
    if distinct_D < 200:
        raise ValueError(f"Fit would be meaningless: only {distinct_D} distinct D values globally. Expected >=200.")

    # Per-rung: each rung must densely trace a D-curve (identifies beta)
    per_rung = {}
    for m, d in zip(metadata, D):
        per_rung.setdefault(m["rung"], set()).add(int(d))
    for rung, ds in per_rung.items():
        print(f"[fit] {rung}: {len(ds)} distinct D values")
        if len(ds) < 50:
            raise ValueError(f"{rung}: only {len(ds)} distinct D values, expected >=50 to identify beta.")

    if len(losses) == 0:
        print("[fit] No data found. Run the ladder first.")
        return

    print("[fit] Fitting L(N,D) = E + A/N^alpha + B/D^beta ...")
    results = fit_chinchilla(N, D, losses)

    print(f"\n{'='*60}")
    print(f"SCALING LAW FIT: L(N,D) = E + A/N^α + B/D^β")
    print(f"{'='*60}")
    print(f"  E  = {results['E']:.4f} ± {results['E_err']:.4f}  (irreducible loss)")
    print(f"  A  = {results['A']:.2f} ± {results['A_err']:.2f}")
    print(f"  α  = {results['alpha']:.4f} ± {results['alpha_err']:.4f}  (Chinchilla: 0.34)")
    print(f"  B  = {results['B']:.2f} ± {results['B_err']:.2f}")
    print(f"  β  = {results['beta']:.4f} ± {results['beta_err']:.4f}  (Chinchilla: 0.28)")
    print(f"{'='*60}")
    print(f"  Points: {results['n_points']}")
    print(f"  R² vs constant baseline: {results['r_squared']:.4f}")
    print(f"  Jacobian condition number: {results['jacobian_condition']:.2e}")
    print(f"{'='*60}")

    # Falsification check
    if results["r_squared"] < 0.5:
        print("\n[FALSIFIED] Fit does not beat constant baseline (R² < 0.5)")
    elif results["jacobian_condition"] > 1e6:
        print("\n[WARNING] Jacobian is poorly conditioned (condition > 1e6)")
    else:
        print("\n[OK] Fit beats constant baseline and is well-conditioned")

    # Save
    Path("artifacts/scaling").mkdir(parents=True, exist_ok=True)
    Path("artifacts/scaling/fit_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\n[fit] Results saved to artifacts/scaling/fit_results.json")


if __name__ == "__main__":
    main()
