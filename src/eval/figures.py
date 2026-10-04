"""Evaluation figures for the final report (TICK-403).

Reads the TICK-204 ``comparison.json`` and a trained checkpoint, then writes:

* ``loss_curves.png``       validation bits-per-character vs training step
* ``perplexity_params.png`` per-character perplexity and parameter counts
* ``latency.png``           generation speed (tokens/s and characters/s)
* ``attention.png``         per-layer, per-head attention maps for one prompt
* ``figures.json``          manifest with the data behind each figure

Per-token loss is NOT comparable across tokenizers, so curves are converted to
bits per character using each run's measured characters-per-token ratio.

Requires matplotlib (``pip install matplotlib``); the data helpers and the
attention extraction work without it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_PROMPT = "ROMEO:\nWhat light through yonder window"


def load_comparison(path: str | Path) -> list[dict]:
    results = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(results, list) or not results:
        raise ValueError("comparison file must contain a non-empty list of runs")
    return results


def bits_per_character_curve(result: dict) -> list[tuple[float, float]]:
    """Validation loss history as ``(step, bits per character)`` pairs."""
    chars_per_token = result["characters_per_token"]
    if chars_per_token <= 0:
        raise ValueError("characters_per_token must be positive")
    return [
        (entry["step"], entry["validation_loss"] / math.log(2) / chars_per_token)
        for entry in result["loss_history"]
    ]


def latency_table(result: dict) -> dict[str, dict[str, float]]:
    """Map decoding mode (``greedy``, ``top-k``) to tokens/s and characters/s."""
    return {
        sample["mode"].split()[0]: {
            "tokens_per_second": sample["tokens_per_second"],
            "characters_per_second": sample["characters_per_second"],
        }
        for sample in result["samples"]
    }


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:  # pragma: no cover - exercised only without matplotlib
        raise SystemExit("matplotlib is required for figures: .venv/bin/pip install matplotlib") from error
    return plt


def plot_loss_curves(results: list[dict], path: Path) -> list[dict]:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(6, 4))
    data = []
    for result in results:
        curve = bits_per_character_curve(result)
        axis.plot([s for s, _ in curve], [b for _, b in curve], marker="o", label=result["tokenizer"])
        data.append({"tokenizer": result["tokenizer"], "curve": curve})
    axis.set_xlabel("Training step")
    axis.set_ylabel("Validation bits per character")
    axis.set_title("Validation loss (character-normalised)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return data


def plot_perplexity_params(results: list[dict], path: Path) -> list[dict]:
    plt = _pyplot()
    names = [r["tokenizer"] for r in results]
    figure, (left, right) = plt.subplots(1, 2, figsize=(8, 4))
    perplexity = [r["validation_perplexity_per_character"] for r in results]
    parameters = [r["parameters"] for r in results]
    left.bar(names, perplexity, color=["#4c72b0", "#dd8452"][: len(names)])
    left.set_ylabel("Validation perplexity per character")
    left.set_title("Perplexity (lower is better)")
    right.bar(names, parameters, color=["#4c72b0", "#dd8452"][: len(names)])
    right.set_ylabel("Parameters")
    right.set_title("Model size")
    for axis, values in ((left, perplexity), (right, parameters)):
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:,.2f}" if value < 1000 else f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
        axis.set_ylim(0, max(values) * 1.15)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [{"tokenizer": n, "perplexity_per_character": p, "parameters": q} for n, p, q in zip(names, perplexity, parameters)]


def plot_latency(results: list[dict], path: Path) -> list[dict]:
    plt = _pyplot()
    tables = {r["tokenizer"]: latency_table(r) for r in results}
    modes = sorted({mode for table in tables.values() for mode in table})
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    width = 0.8 / max(len(tables), 1)
    for axis, key, label in (
        (axes[0], "tokens_per_second", "Tokens per second"),
        (axes[1], "characters_per_second", "Characters per second"),
    ):
        for offset, (name, table) in enumerate(tables.items()):
            positions = [i + offset * width for i in range(len(modes))]
            axis.bar(positions, [table.get(mode, {}).get(key, 0.0) for mode in modes], width, label=name)
        axis.set_xticks([i + width * (len(tables) - 1) / 2 for i in range(len(modes))])
        axis.set_xticklabels(modes)
        axis.set_ylabel(label)
        axis.set_title(label + " (CPU, higher is better)")
        axis.set_ylim(0, max((t.get(m, {}).get(key, 0.0) for t in tables.values() for m in modes), default=1.0) * 1.25)
        axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [{"tokenizer": name, "latency": table} for name, table in tables.items()]


def plot_attention(weights, tokens: list[str], path: Path) -> dict:
    """Grid of attention maps: one row per layer, one column per head."""
    plt = _pyplot()
    layers, heads, length, _ = weights.shape
    figure, axes = plt.subplots(layers, heads, figsize=(2.6 * heads, 2.6 * layers), squeeze=False)
    for layer in range(layers):
        for head in range(heads):
            axis = axes[layer][head]
            axis.imshow(weights[layer, head].numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
            axis.set_title(f"L{layer} H{head}", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(f"Causal attention, {length} tokens (rows = query, columns = key)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {"layers": layers, "heads": heads, "tokens": tokens}


def _load_checkpoint_model(checkpoint_path: str, device: str):
    import torch

    from src.model import GPTConfig, MiniGPT
    from src.tokenizer import tokenizer_from_checkpoint

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MiniGPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, tokenizer_from_checkpoint(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default="artifacts/tick-204/comparison.json")
    parser.add_argument("--checkpoint", default="artifacts/tick-204/bpe/checkpoint.pt", help="Checkpoint used for the attention figure")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", default="artifacts/tick-403")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    from src.eval.attention import attention_weights

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_comparison(args.comparison)

    manifest = {
        "loss_curves": {"file": "loss_curves.png", "data": plot_loss_curves(results, output_dir / "loss_curves.png")},
        "perplexity_params": {"file": "perplexity_params.png", "data": plot_perplexity_params(results, output_dir / "perplexity_params.png")},
        "latency": {"file": "latency.png", "data": plot_latency(results, output_dir / "latency.png")},
    }

    model, tokenizer = _load_checkpoint_model(args.checkpoint, args.device)
    ids = tokenizer.encode(args.prompt)[-model.config.context_length:]
    weights = attention_weights(model, torch.tensor([ids], dtype=torch.long, device=args.device)).cpu()
    tokens = [tokenizer.decode([i], skip_special_tokens=True) for i in ids]
    manifest["attention"] = {
        "file": "attention.png",
        "checkpoint": args.checkpoint,
        "prompt": args.prompt,
        **plot_attention(weights, tokens, output_dir / "attention.png"),
    }

    (output_dir / "figures.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for key, value in manifest.items():
        print(f"Wrote {output_dir / value['file']}")
    print(f"Wrote {output_dir / 'figures.json'}")


if __name__ == "__main__":
    main()
