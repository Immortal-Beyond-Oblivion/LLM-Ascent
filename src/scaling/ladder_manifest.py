"""The explicit point set of the scaling ladder.

Why this module exists
----------------------
The first version of Act 4's fit was fitted to a mixture of two architecture
families, and nothing complained. The ladder's own ``in_fitted_set`` column
reproduced the point COUNT (2,776) but not the point SET, the run records
carried three different conventions for the parameter count, and the committed
ladder script encoded the *other* family's architecture. A good-looking fit
was not evidence of a good fit.

So the point set is now a **manifest**: an explicit list of runs, each with a
role, each with the reason for its role, each with the source file it came
from. The fitter selects on the role column; nothing is selected by a flag
that reproduces a number.

Roles
-----
``canonical_ladder``
    A run that belongs to the fit: the canonical MiniGPT family, one run per
    rung, on the D = 20N line, plus the one off-line design point.
``other_family``
    The second family: no learned position table, identified by the
    first-logged-loss threshold. Kept because the contamination story, the
    matched-(N, D) architecture comparison and the "the withdrawn fit's own
    parameters score R^2 = +0.9957 on this family" diagnostic all need it.
``superseded``
    A recorded canonical run at a parameter count the rescue re-measured.
    Kept in the manifest for the audit trail, excluded from the fit so a rung
    is not counted twice.
``excluded_unidentifiable``
    A run whose recorded parameter count matches no known architecture, which
    cannot be placed in a family by anything but a threshold. This is the 0.8M
    rung: it carried the act's worst hold-out fold.
``excluded_not_a_rung``
    Runs that are not part of the ladder at all (the ``tiny`` smoke models).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = ROOT / "artifacts" / "scaling"
RESULTS = ROOT / "results" / "v2" / "act4-scaling"

#: A run whose first logged validation loss is at or above this was trained
#: without a learned position table. The two families are separated by more
#: than an order of magnitude, so the threshold is not delicate.
FIRST_LOSS_THRESHOLD = 20.0

#: The ladder's design point: D = 20N tokens of training per parameter.
D_OVER_N = 20

#: The off-line design point added by the rescue: same N, half the D.
D_OVER_N_OFFLINE = 10

COLUMNS = [
    "rung", "seed", "arm", "source", "role", "role_reason",
    "non_embedding_params", "total_params", "embedding_params",
    "training_tokens", "validation_loss", "d_over_n", "steps",
    "first_logged_val_loss", "final_val_loss", "device",
    "model_class", "positional_encoding", "num_layers", "num_heads",
    "embedding_dim", "context_length", "config_hash", "live_config_hash",
]


def _family(first_val_loss: float, arch: dict) -> str:
    if arch.get("model_class") == "MiniGPT" or first_val_loss < FIRST_LOSS_THRESHOLD:
        return "canonical"
    return "other"


def _config_hash(arch: dict, live: str | None = None) -> str:
    """The full architectural fingerprint, and the live hash beside it.

    The full form is what two runs must agree on to be the same model. The
    live hash is computed from the instantiated model at the moment it was
    logged, which is the only one that cannot be wrong after the fact.
    """
    full = (f"{arch.get('model_class')}|{arch.get('positional_encoding')}"
            f"|L{arch.get('num_layers')}|H{arch.get('num_heads')}"
            f"|d{arch.get('embedding_dim')}|V{arch.get('vocab_size', 6487)}"
            f"|C{arch.get('context_length')}")
    return f"{full}|live={live}" if live else full


def _ladder_runs() -> list[dict]:
    """Every run in the original ladder, read from its own metrics.json."""
    out: list[dict] = []
    root = ARTIFACTS / "checkpoints"
    if not root.is_dir():
        return out
    for rung_dir in sorted(root.iterdir()):
        if not rung_dir.is_dir():
            continue
        for seed_dir in sorted(rung_dir.iterdir()):
            metrics_path = seed_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            history = m.get("history") or []
            if not history:
                continue
            arch = dict(m.get("architecture") or {})
            arch.setdefault("vocab_size", 6487)
            first = float(history[0]["val_loss"])
            out.append({
                "rung": rung_dir.name,
                "seed": seed_dir.name.replace("seed", ""),
                "arm": "ladder",
                "source": "ladder",
                "non_embedding_params": int(m.get("non_embedding_params") or 0),
                "total_params": int(m.get("total_params") or 0),
                "embedding_params": int(m.get("embedding_params") or 0),
                "tokens": int(m.get("tokens") or 0),
                "steps": int(m.get("steps") or 0),
                "device": m.get("device"),
                "first_logged_val_loss": first,
                "final_val_loss": float(history[-1]["val_loss"]),
                "model_class": arch.get("model_class"),
                "positional_encoding": arch.get("positional_encoding"),
                "num_layers": arch.get("num_layers"),
                "num_heads": arch.get("num_heads"),
                "embedding_dim": arch.get("embedding_dim"),
                "context_length": arch.get("context_length"),
                "config_hash": _config_hash(arch),
                "live_config_hash": None,
                "history": history,
                "family": _family(first, arch),
            })
    return out


def _rescue_runs() -> list[dict]:
    """Every run the rescue produced, read from its own config/history/summary."""
    out: list[dict] = []
    root = ARTIFACTS / "rescue"
    if not root.is_dir():
        return out
    for config_path in sorted(root.glob("*_config.json")):
        tag = config_path.name[: -len("_config.json")]
        history_path = root / f"{tag}_history.json"
        summary_path = root / f"{tag}_summary.json"
        if not history_path.exists():
            continue
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        summary = (json.loads(summary_path.read_text(encoding="utf-8"))
                   if summary_path.exists() else {})
        # "<rung>_d<N>N_seed<seed>" or "<rung>_seed<seed>"
        stem = tag[: -len(f"_seed{tag.rsplit('_seed', 1)[1]}")] if "_seed" in tag else tag
        seed = tag.rsplit("_seed", 1)[1]
        if "_d" in stem and stem.rsplit("_d", 1)[1].endswith("N"):
            rung, ratio = stem.rsplit("_d", 1)[0], int(stem.rsplit("_d", 1)[1][:-1])
        else:
            rung, ratio = stem, D_OVER_N
        arch = {
            "model_class": cfg.get("model_class"),
            "positional_encoding": cfg.get("positional_encoding"),
            "num_layers": cfg.get("num_layers"),
            "num_heads": cfg.get("num_heads"),
            "embedding_dim": cfg.get("embedding_dim"),
            "context_length": cfg.get("context_length"),
            "vocab_size": cfg.get("vocab_size", 6487),
        }
        out.append({
            "rung": rung,
            "seed": seed,
            "arm": f"D{ratio}N",
            "source": "rescue",
            "non_embedding_params": int(cfg.get("non_embedding_params") or 0),
            "total_params": int(cfg.get("total_params_deduplicated") or 0),
            "embedding_params": int(cfg.get("vocab_size", 6487) * (cfg.get("embedding_dim") or 0)),
            "tokens": int(summary.get("tokens") or (history[-1]["tokens"] if history else 0)),
            "steps": int(summary.get("steps") or (history[-1]["step"] if history else 0)),
            "device": summary.get("device"),
            "first_logged_val_loss": float(history[0]["val_loss"]) if history else None,
            "final_val_loss": float(summary.get("final_val_loss")
                                    or (history[-1]["val_loss"] if history else 0.0)),
            "model_class": arch["model_class"],
            "positional_encoding": arch["positional_encoding"],
            "num_layers": arch["num_layers"],
            "num_heads": arch["num_heads"],
            "embedding_dim": arch["embedding_dim"],
            "context_length": arch["context_length"],
            "config_hash": _config_hash(arch, cfg.get("config_hash")),
            "live_config_hash": cfg.get("config_hash"),
            "history": history,
            "family": "canonical",
        })
    return out


def _propagate_architecture(runs: list[dict]) -> None:
    """Give every run at a rung the architecture its siblings recorded.

    Several ladder runs have no ``architecture`` block: the training code that
    produced them did not write one. Where a rung's runs all share a
    parameter count and at least one of them recorded its shape, that shape is
    the rung's shape, and the run that recorded it is the evidence.
    """
    by_rung: dict[tuple[str, int], dict] = {}
    for run in runs:
        if run["source"] != "ladder" or run["num_layers"] is None:
            continue
        key = (run["rung"], run["non_embedding_params"])
        by_rung.setdefault(key, run)
    for run in runs:
        if run["source"] != "ladder" or run["num_layers"] is not None:
            continue
        sibling = by_rung.get((run["rung"], run["non_embedding_params"]))
        if sibling is None:
            continue
        for field in ("model_class", "positional_encoding", "num_layers",
                      "num_heads", "embedding_dim", "context_length"):
            run[field] = sibling[field]
        run["config_hash"] = _config_hash({
            "model_class": run["model_class"],
            "positional_encoding": run["positional_encoding"],
            "num_layers": run["num_layers"],
            "num_heads": run["num_heads"],
            "embedding_dim": run["embedding_dim"],
            "context_length": run["context_length"],
            "vocab_size": 6487,
        })


def assign_roles(runs: list[dict]) -> None:
    """Set each run's role and the reason for it. This is the audit trail."""
    rescued_ns = {r["non_embedding_params"] for r in runs if r["source"] == "rescue"}
    for run in runs:
        if run["rung"] == "tiny":
            run["role"] = "excluded_not_a_rung"
            run["role_reason"] = "a smoke model, not a rung of the ladder"
            continue
        if run["source"] == "rescue":
            if run["arm"] == f"D{D_OVER_N_OFFLINE}N":
                run["role"] = "canonical_ladder"
                run["role_reason"] = (
                    "the off-line design point: the same N as the D = 20N rung at "
                    "D/N = 10, so N and D are no longer collinear across the ladder"
                )
            else:
                run["role"] = "canonical_ladder"
                run["role_reason"] = (
                    "a rescued canonical run whose config hash was computed from the "
                    "live model at the moment each point was logged"
                )
            continue
        if run["family"] == "other":
            run["role"] = "other_family"
            run["role_reason"] = (
                "the second family: no learned position table, identified by a first "
                "logged validation loss of "
                f"{run['first_logged_val_loss']:.1f} against a threshold of "
                f"{FIRST_LOSS_THRESHOLD:.0f}"
            )
            continue
        if run["non_embedding_params"] in rescued_ns:
            run["role"] = "superseded"
            run["role_reason"] = (
                "a recorded canonical run at a parameter count the rescue re-measured; "
                "kept for the audit trail, excluded so a rung is not counted twice"
            )
            continue
        if run["num_layers"] is None:
            run["role"] = "excluded_unidentifiable"
            run["role_reason"] = (
                f"a recorded non-embedding parameter count of {run['non_embedding_params']:,} "
                "matches no known architecture, so the run cannot be placed in a family "
                "by anything but a threshold; this rung carries the act's worst "
                "hold-out fold"
            )
            continue
        run["role"] = "canonical_ladder"
        run["role_reason"] = (
            "a recorded canonical run with an architecture block, at a rung the "
            "rescue did not re-measure"
        )


def build_manifest() -> dict:
    runs = _ladder_runs() + _rescue_runs()
    if not runs:
        # artifacts/ is git-ignored, so a fresh clone has no run records. Fall
        # back to the tracked manifest rather than an empty point set.
        tracked = RESULTS / "ladder_runs.json"
        if tracked.exists():
            return json.loads(tracked.read_text(encoding="utf-8"))
    _propagate_architecture(runs)
    assign_roles(runs)
    for run in runs:
        run["d_over_n"] = (round(run["tokens"] / run["non_embedding_params"], 4)
                           if run["non_embedding_params"] else None)
        run["n_points"] = len(run.pop("history"))
    runs.sort(key=lambda r: (r["role"], r["rung"], r["arm"], r["seed"]))
    canonical = [r for r in runs if r["role"] == "canonical_ladder"]
    ns = sorted({r["non_embedding_params"] for r in canonical})
    return {
        "what": (
            "The explicit point set of the scaling ladder. The fitter selects on the "
            "'role' column; there is no flag that reproduces a point count."
        ),
        "why": (
            "The first fit was computed over a mixture of two architecture families "
            "and nothing complained. The 'in_fitted_set' column reproduced the point "
            "COUNT and not the point SET, and three parameter-count conventions "
            "coexisted in one table. This manifest is the replacement."
        ),
        "first_loss_threshold": FIRST_LOSS_THRESHOLD,
        "design_point_d_over_n": D_OVER_N,
        "offline_design_point_d_over_n": D_OVER_N_OFFLINE,
        "counts": {
            role: sum(1 for r in runs if r["role"] == role)
            for role in sorted({r["role"] for r in runs})
        },
        "canonical_ladder": {
            "distinct_n": len(ns),
            "non_embedding_params": ns,
            "log_n_range": (round(__import__("math").log(ns[-1] / ns[0]), 4)
                             if len(ns) > 1 else None),
            "rungs": sorted({r["rung"] for r in canonical}),
        },
        "runs": runs,
    }


def export_points(manifest: dict) -> Path:
    """Write the merged point file: every run's points, with its role."""
    out_path = RESULTS / "ladder_points.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not ARTIFACTS.is_dir():
        # No run histories to export; keep the tracked point file intact.
        return out_path
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for run in manifest["runs"]:
            if run["role"] == "excluded_not_a_rung":
                continue
            base = {c: run.get(c) for c in COLUMNS}
            base["arm"] = run["arm"]
            for point in _points_of(run):
                row = dict(base)
                row["training_tokens"] = point["tokens"]
                row["validation_loss"] = point["val_loss"]
                writer.writerow(row)
    return out_path


def _points_of(run: dict) -> list[dict]:
    """The logged interior points of a run, read from its own history."""
    root = ARTIFACTS / "checkpoints" / run["rung"] / f"seed{run['seed']}"
    if run["source"] == "ladder":
        metrics = root / "metrics.json"
        if not metrics.exists():
            return []
        return json.loads(metrics.read_text(encoding="utf-8")).get("history") or []
    stem = f"{run['rung']}_d{run['arm'][1:-1]}" if run["arm"] != f"D{D_OVER_N}N" else run["rung"]
    tag = f"{stem}_seed{run['seed']}"
    history_path = ARTIFACTS / "rescue" / f"{tag}_history.json"
    if not history_path.exists():
        return []
    return json.loads(history_path.read_text(encoding="utf-8"))


def main() -> None:
    manifest = build_manifest()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ladder_runs.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    path = export_points(manifest)
    print(json.dumps(manifest["counts"], indent=2))
    print("canonical ladder:",
          manifest["canonical_ladder"]["distinct_n"], "distinct N values,",
          "log-N range", manifest["canonical_ladder"]["log_n_range"])
    print("wrote", path)


if __name__ == "__main__":
    main()
