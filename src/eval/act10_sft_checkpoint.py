"""Act 10c: save a TRACKED supervised-fine-tuned checkpoint.

Act 7 trained its SFT models in memory and tracked only their metrics, so the
instruction-tuned weights were not in the artefact set. That is a missing save
call, not an inherent gap: Act 7's full fine-tune is 512 examples for 24
epochs on the 6.3M canonical model, and re-running it with the weights saved
costs minutes.

Two things it unlocks:

  * 10c's SFT-versus-base comparison, whose question is sharp: does instruction
    tuning change CALIBRATION, or leave a well-calibrated model that still
    cannot follow instructions? Either answer is publishable.
  * a fourth tracked canonical checkpoint, so Act 7's weights are auditable
    like the other three.

The saved file uses the SAME format as the other three canonical checkpoints
(``model_state`` only, no optimizer state, the tied embedding present in the
state dict exactly as ``MiniGPT.state_dict()`` produces it), so it loads with
``load_state_dict(strict=True)`` like the others.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.adaptation.peft import PEFTSpec, apply_peft
from src.adaptation.sft import SFTConfig, sft_train

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "checkpoints" / "canonical"
OUT_DIR = CANON / "6.3M-sft"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", default="6.3M")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from src.data.instruction_tasks import make_splits
    from src.eval.act7_sft import evaluate, load_canonical
    from src.tokenizer.bpe import BPETokenizer

    tokenizer = BPETokenizer.load(str(CANON / "tokenizer.json"))
    splits = make_splits(n_train=512, n_held_out=128)
    base = load_canonical(args.rung)
    wrapped = apply_peft(base, PEFTSpec(method="full", size=1))
    wrapped.to(args.device)
    torch.manual_seed(args.seed)

    t0 = time.time()
    info = sft_train(wrapped, splits.train[:args.n_train], tokenizer,
                      SFTConfig(epochs=args.epochs, seed=args.seed))
    elapsed = time.time() - t0

    inner = wrapped.base
    inner.eval()
    metrics = evaluate(inner, tokenizer, splits.held_out, use_score=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": inner.state_dict(),
        "model_config": {
            "vocabulary_size": inner.config.vocabulary_size,
            "context_length": inner.config.context_length,
            "embedding_dim": inner.config.embedding_dim,
            "num_heads": inner.config.num_heads,
            "num_layers": inner.config.num_layers,
            "num_kv_heads": inner.config.num_kv_heads,
            "dropout": 0.0,
        },
        "rung": f"{args.rung}-sft",
        "seed": args.seed,
        "completed_steps": args.epochs,
        "val_loss": info.get("final_loss", info.get("loss", 0.0)),
        "note": (f"model_state only; no optimizer state. Supervised fine-tune of "
                 f"the {args.rung} canonical checkpoint on {args.n_train} Act 7 "
                 f"instruction examples for {args.epochs} epochs at seed "
                 f"{args.seed}. The tied embedding is present in the state dict "
                 f"exactly as MiniGPT.state_dict() produces it, matching the "
                 f"other canonical checkpoints."),
    }
    path = OUT_DIR / "model.pt"
    torch.save(payload, path)

    print(f"[sft] trained {args.epochs} epochs on {args.n_train} examples in "
          f"{elapsed / 60:.1f} min on {args.device}")
    print(f"[sft] final train loss: {info.get('final_loss', info.get('loss'))}")
    print(f"[sft] held-out exact match: {metrics}")
    print(f"[sft] wrote {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)")

    summary = {
        "rung": f"{args.rung}-sft",
        "n_train": args.n_train,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": args.device,
        "final_train_loss": info.get("final_loss", info.get("loss")),
        "held_out": metrics,
        "checkpoint": str(path.relative_to(ROOT)),
        "elapsed_seconds": round(elapsed, 1),
    }
    out_json = ROOT / "results" / "v2" / "act10" / "sft_checkpoint.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[sft] wrote {out_json}")


if __name__ == "__main__":
    main()
