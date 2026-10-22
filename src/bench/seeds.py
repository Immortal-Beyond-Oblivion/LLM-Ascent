"""Deterministic seeding for reproducible experiments.

set_seed() seeds Python random, torch CPU, torch CUDA, and torch MPS.
verify_determinism() runs the same config twice and compares metrics.json bytes.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch


def set_seed(seed: int) -> None:
    """Seed all RNGs: Python random, torch CPU, torch CUDA, torch MPS."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        # MPS generator seeding is supported in torch >= 2.0
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass  # older torch without explicit MPS seeding


def verify_determinism(
    config_a: "ExperimentConfig",
    config_b: "ExperimentConfig",
) -> tuple[bool, str]:
    """Run two configs and compare their metrics.json bytes.

    Returns (is_identical, message). If the configs are identical and the
    metrics.json files are byte-identical, returns (True, ...).
    """
    from src.bench.experiment import run_experiment

    metrics_a = run_experiment(config_a)
    metrics_b = run_experiment(config_b)

    path_a = Path(config_a.output_root) / config_a.act / config_a.experiment / "metrics.json"
    path_b = Path(config_b.output_root) / config_b.act / config_b.experiment / "metrics.json"

    bytes_a = path_a.read_bytes()
    bytes_b = path_b.read_bytes()

    # Compare loss histories (the reproducible substance), not wall-clock timing
    if len(metrics_a.loss_history) != len(metrics_b.loss_history):
        return False, f"loss history lengths differ: {len(metrics_a.loss_history)} vs {len(metrics_b.loss_history)}"

    max_diff = 0.0
    for rec_a, rec_b in zip(metrics_a.loss_history, metrics_b.loss_history):
        if rec_a["step"] != rec_b["step"]:
            return False, f"step mismatch: {rec_a['step']} vs {rec_b['step']}"
        diff = abs(rec_a["validation_loss"] - rec_b["validation_loss"])
        max_diff = max(max_diff, diff)
        diff = abs(rec_a["train_loss"] - rec_b["train_loss"])
        max_diff = max(max_diff, diff)

    if max_diff < 1e-9:
        return True, f"loss histories identical (max diff: {max_diff:.2e})"
    else:
        return False, f"loss histories differ (max diff: {max_diff:.2e})"
