"""Act 4 rescue: re-run the contaminated top of the ladder with the canonical architecture.

WHY. The Act 4 ladder mixed two architecture families. The 4.1M and 9.0M rungs
are ENTIRELY of the other family, so the clean fit has only four rungs and four
distinct N values, and alpha is weakly identified. This module re-runs those
two rungs with the canonical architecture (MiniGPT, 8 heads, learned absolute
positions, LayerNorm), which REPLACES the broken rungs rather than merely
adding points.

TWO CONSTRAINTS, both of which are the point of the exercise.

1. D = 20N. The canonical rungs sit on the D = 20N line (1.5M at 30.7M
   tokens, 2.6M at 53.1M, 6.3M at 125.8M). The broken 9.0M rung sat at
   D = 10N, which is precisely the confound: N and D stop being separable.
   Putting 9.0M back on 10N would rebuild the original mistake.

2. THE CONFIG HASH IS TAKEN FROM THE MODEL, NOT FROM A TABLE. Every logged
   point carries a hash computed from the live model's classes, shapes and
   positional-encoding type at the moment it was logged. Retroactive tagging
   from a rung table is what let a mid-campaign architecture change hide in
   plain sight: the ladder's ALiBi runs and MiniGPT runs produced plausible
   losses at plausible N values, and nothing in the artefacts distinguished
   them until the provenance was read by hand.

A VALIDATION MODE EXISTS BECAUSE THE RECIPE IS NOT FULLY RECOVERABLE. The
artefacts record no learning rate, so the rescue uses the recipe in the
current training code (AdamW, lr 3e-4, 100 warmup steps, cosine decay to
0.1x, batch 32, context 128, gradient clipping at 1.0). Before spending hours
on the rescue, `--validate` re-runs the 1.5M rung with that recipe and compares
the final validation loss with the canonical 1.5M's recorded 3.0571. If the
recipe does not reproduce it, the rescue would be measuring a different recipe
and is not comparable, and the script says so.

PRE-REGISTERED DECISION GATE, fixed before any rescue run:
  * if the canonical family with 6+ rungs still leaves alpha varying widely
    across fit variants, the RESULT IS THE NULL: "this ladder does not identify
    alpha at this scale". That is publishable and is already half-written.
  * a previous alpha of 0.44 does not come back unless the data support it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.model.gpt import GPTConfig, MiniGPT
from src.scaling.ladder import ASPECT_LADDER, MemmapCorpus

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "scaling" / "rescue"
RESULTS = ROOT / "results" / "v2" / "act4-scaling"

BATCH = 32
CONTEXT = 128
LR = 3e-4
WARMUP = 100
EVAL_INTERVAL = 50
EVAL_BATCHES = 10
CLIP = 1.0
D_OVER_N = 20


def build_canonical(rung: str, vocab_size: int, *,
                    target_non_embedding: int | None = None) -> MiniGPT:
    """The canonical architecture: 8 heads, learned absolute positions.

    With ``target_non_embedding`` the (dim, layers) pair is searched for
    directly, which is how the small rungs (0.3M, 0.4M) are added: the
    leverage of a power law fitted against log N is at the BOTTOM of the
    range, so a new small N buys more identifiability per minute than a new
    large one.
    """
    if target_non_embedding is not None:
        dim, layers = _search_shape(vocab_size, target_non_embedding)
    else:
        dim, layers = ASPECT_LADDER[rung]
    config = GPTConfig(vocabulary_size=vocab_size, context_length=CONTEXT,
                       embedding_dim=dim, num_heads=8, num_layers=layers,
                       dropout=0.0)
    return MiniGPT(config)


def _search_shape(vocab_size: int, target: int) -> tuple[int, int]:
    """The (dim, layers) whose non-embedding count is closest to ``target``."""
    best = None
    for dim in (64, 80, 96, 112, 128, 160):
        for layers in (2, 3, 4, 5, 6):
            model = MiniGPT(GPTConfig(vocabulary_size=vocab_size,
                                      context_length=CONTEXT, embedding_dim=dim,
                                      num_heads=8, num_layers=layers, dropout=0.0))
            diff = abs(model.non_embedding_params() - target)
            if best is None or diff < best[0]:
                best = (diff, dim, layers)
            del model
    return best[1], best[2]


def config_hash(model: MiniGPT) -> dict:
    """A hash computed FROM THE LIVE MODEL, not from a rung table.

    Retroactive tagging is what let the architecture change hide: the points
    carried a plausible N and a plausible loss, and nothing said which model
    had produced them. This is taken at the moment of logging, from the
    model's own classes and tensor shapes.
    """
    import hashlib

    parts = [type(model).__name__]
    for name, module in model.named_modules():
        parts.append(f"{name}:{type(module).__name__}")
    for name, tensor in model.state_dict().items():
        parts.append(f"{name}:{tuple(tensor.shape)}")
    blob = "|".join(parts).encode()
    return {
        "config_hash": hashlib.sha256(blob).hexdigest()[:16],
        "model_class": type(model).__name__,
        "embedding_dim": model.config.embedding_dim,
        "num_heads": model.config.num_heads,
        "num_layers": model.config.num_layers,
        "context_length": model.config.context_length,
        "positional_encoding": "learned_absolute",
        "vocab_size": model.config.vocabulary_size,
        "non_embedding_params": model.non_embedding_params(),
        "total_params_deduplicated": model.total_params(),
    }


def train(rung: str, *, device: str, seed: int, out_dir: Path,
          validate: bool, target_non_embedding: int | None = None,
          d_over_n: int = D_OVER_N) -> dict:
    corpus = MemmapCorpus()
    model = build_canonical(rung, corpus.vocab_size,
                            target_non_embedding=target_non_embedding).to(device)
    model.train()
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = model.non_embedding_params()
    if validate:
        # The canonical rung's own token budget, so the reproduction target is
        # the recorded canonical loss rather than a new one.
        tokens = {"1.5M": 30_720_000, "2.6M": 53_084_160,
                  "6.3M": 125_829_120}[rung]
    else:
        tokens = d_over_n * N
    steps = tokens // (BATCH * CONTEXT)

    cfg = config_hash(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The tag must distinguish arms that share a rung but not a D/N ratio. The
    # 9.0M arm at D=10N overwrote the 9.0M arm at D=20N's checkpoint, summary
    # and history, because both were called "9.0M_seed42". Their final losses
    # happened to survive (the summary and history are written at the END of a
    # run) but the weights did not.
    tag = (f"{rung}_seed{seed}" if d_over_n == D_OVER_N
           else f"{rung}_d{d_over_n}N_seed{seed}")
    (out_dir / f"{tag}_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[rescue] {tag}: N={N:,} D={tokens:,} steps={steps} device={device}")
    print(f"[rescue]   config_hash={cfg['config_hash']} "
          f"({cfg['model_class']}, {cfg['num_heads']} heads, "
          f"{cfg['num_layers']} layers, {cfg['positional_encoding']})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    history: list[dict] = []
    start_step = 0
    ckpt_path = out_dir / f"{tag}.pt"

    # ---- resume -----------------------------------------------------------
    # A long arm must be pausable. The checkpoint carries the model, the
    # optimizer, the history and -- crucially -- the RNG states, because the
    # BATCH ORDER comes from the global numpy RNG (the corpus samples offsets
    # with np.random.randint) and is seeded only once at the start of the run.
    # Without restoring it, a resumed run would silently see a different data
    # order from the point of the resume.
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optimizer_state"])
        except (RuntimeError, KeyError) as exc:
            print(f"[rescue] {tag}: checkpoint does not match this config "
                  f"({exc}); starting fresh", flush=True)
            if ckpt_path.exists():
                ckpt_path.unlink()
        else:
            history = list(ck.get("history", []))
            start_step = int(ck.get("completed_steps", 0))
            if "torch_rng_state" in ck:
                torch.set_rng_state(ck["torch_rng_state"])
            if "numpy_rng_state" in ck:
                np.random.set_state(ck["numpy_rng_state"])
            print(f"[rescue] {tag}: RESUMING from step {start_step}/{steps} "
                  f"({len(history)} logged points, "
                  f"{'exact RNG restore' if 'numpy_rng_state' in ck else 'NO RNG state in checkpoint -- the batch order will restart'})",
                  flush=True)

    t0 = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        if step <= WARMUP:
            lr = LR * step / WARMUP
        else:
            progress = (step - WARMUP) / max(1, steps - WARMUP)
            lr = LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = corpus.train_batch(BATCH, CONTEXT, device)
        _out = model(inputs, targets)
        loss = _out[1] if isinstance(_out, tuple) else _out
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        if step == 1 or step % EVAL_INTERVAL == 0 or step == steps:
            model.eval()
            vals = []
            with torch.no_grad():
                for _ in range(EVAL_BATCHES):
                    vi, vt = corpus.val_batch(BATCH, CONTEXT, device)
                    o = model(vi, vt)
                    vals.append(float((o[1] if isinstance(o, tuple) else o).detach()))
            model.train()
            avg = sum(vals) / len(vals)
            history.append({
                "step": step,
                "tokens": step * BATCH * CONTEXT,
                "val_loss": avg,
                "elapsed_seconds": time.perf_counter() - t0,
                "config_hash": cfg["config_hash"],
                "non_embedding_params": N,
                "total_params_deduplicated": cfg["total_params_deduplicated"],
                "model_class": cfg["model_class"],
                "positional_encoding": cfg["positional_encoding"],
            })
            if step % (EVAL_INTERVAL * 40) == 0 or step == steps:
                print(f"[rescue] {tag} step {step}/{steps} "
                      f"tokens={step * BATCH * CONTEXT:,} val={avg:.4f} "
                      f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
            # Checkpoint so a long run can be paused and resumed EXACTLY: the
            # RNG states are part of it, because the batch order comes from
            # the global numpy RNG and is seeded only once, at the start.
            torch.save({"completed_steps": step, "history": history,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": cfg,
                        "torch_rng_state": torch.get_rng_state(),
                        "numpy_rng_state": np.random.get_state()},
                       ckpt_path)
    (out_dir / f"{tag}_history.json").write_text(json.dumps(history, indent=2))
    print(f"[rescue] {tag} DONE final val_loss={history[-1]['val_loss']:.4f} "
          f"in {(time.perf_counter() - t0) / 60:.1f} min")
    return {"rung": rung, "seed": seed, "n_non_embedding": N,
            "tokens": tokens, "steps": steps,
            "final_val_loss": history[-1]["val_loss"], "config": cfg,
            "history": history}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--validate", action="store_true",
                    help="re-run a canonical rung to prove the recipe reproduces "
                         "its recorded loss before the rescue is trusted")
    ap.add_argument("--out", default=str(ARTIFACTS))
    ap.add_argument("--target-n", type=int, default=None,
                    help="target non-embedding parameter count for a NEW rung; "
                         "the (dim, layers) pair is searched for")
    ap.add_argument("--d-over-n", type=int, default=D_OVER_N,
                    help="the D/N ratio of the run. 20 keeps the ladder on its "
                         "single line; 10 is a different ratio and so is a "
                         "different design point")
    args = ap.parse_args()

    out = Path(args.out)
    res = train(args.rung, device=args.device, seed=args.seed, out_dir=out,
                validate=args.validate, target_non_embedding=args.target_n,
                d_over_n=args.d_over_n)
    tag = f"{args.rung}_seed{args.seed}"
    summary = {k: v for k, v in res.items() if k != "history"}
    if args.validate:
        canonical = {"1.5M": 3.057063579559326, "2.6M": 2.8111,
                     "6.3M": 2.5152}
        target = canonical[args.rung]
        summary["validation_target_canonical_val_loss"] = target
        summary["validation_abs_diff"] = abs(res["final_val_loss"] - target)
        summary["validation_passed"] = summary["validation_abs_diff"] < 0.05
        print(f"[rescue] VALIDATION {tag}: derived {res['final_val_loss']:.4f} "
              f"vs canonical {target:.4f} "
              f"(diff {summary['validation_abs_diff']:.4f}) -> "
              f"{'PASS' if summary['validation_passed'] else 'FAIL'}")
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
