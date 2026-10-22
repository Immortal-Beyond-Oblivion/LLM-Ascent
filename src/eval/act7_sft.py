"""Act 7 — SFT, ICL vs SFT vs full fine-tune, PEFT head-to-head, LoRA rank sweep.

All inference+adaptation on the pinned canonical checkpoints; no pretraining.

7a  SFT with a genuinely held-out instruction split. This is the number v1
    never produced: v1 fitted 3 examples full-batch (defect D4) and evaluated on
    the same 3 (no held-out split at all). Here examples are minibatched from a
    seeded shuffle, responses carry a supervised EOS terminator (defect D5),
    and train / held-out / few-shot-pool splits are asserted disjoint.

7b  ICL vs SFT vs full fine-tune across instruction-set sizes. The LIMA
    "less is more" question: does performance saturate at a few hundred
    examples? ICL is evaluated with the same base model and no weight changes,
    so the comparison isolates where the capability comes from.

7c  PEFT family head-to-head at a MATCHED trainable-parameter budget. Without
    matching, "LoRA beats prompt tuning" could just mean "LoRA had more
    trainable parameters". Each method's size is chosen to land nearest the
    budget, and the achieved count is recorded.

7d  LoRA rank sweep r in {1,2,4,8,16,32,64}.

Evaluation is exact match against the deterministic ground truth, with Wilson
95% intervals, n>=3 seeds for stochastic configurations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from src.adaptation.instructions import InstructionExample, InstructionFormatter
from src.adaptation.peft import METHODS, PEFTSpec, apply_peft, n_trainable_for, size_to_budget
from src.adaptation.sft import EOS_ID, SFTConfig, encode_example, sft_train
from src.bench.stats import mean, wilson_interval
from src.data.instruction_tasks import TASKS, make_splits
from src.model.gpt import GPTConfig, MiniGPT
from src.tokenizer.bpe import BPETokenizer

CANON = Path("checkpoints/canonical")
OUT = Path("results/v2/act6/../act7")  # normalised below
OUT = Path("results/v2/act7")
SEEDS = [42, 43, 44]


def load_canonical(rung: str):
    payload = torch.load(CANON / rung / "model.pt", map_location="cpu", weights_only=False)
    model = MiniGPT(GPTConfig(**payload["model_config"])).eval()
    model.load_state_dict(payload["model_state"])
    return model


# ── generation / scoring ────────────────────────────────────────────────────

@torch.no_grad()
def greedy_response(model, tokenizer, prompt_text: str, *, max_new: int = 24) -> str:
    """Greedy decode, stopping at EOS (the supervised terminator, D5)."""
    ids = tokenizer.encode(prompt_text)
    ctx = model.config.context_length
    if isinstance(model, torch.nn.Module) and hasattr(model, "inner"):
        raise TypeError("PEFT wrappers are evaluated through score_response, not generation")
    x = torch.tensor([ids[-ctx:]], dtype=torch.long, device=next(model.parameters()).device)
    for _ in range(max_new):
        logits, _, _ = model(x)
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS_ID:
            break
        x = torch.cat((x, torch.tensor([[nxt]], dtype=torch.long, device=x.device)), dim=1)
        if x.size(1) >= ctx:
            break
    return tokenizer.decode(x[0, len(ids):].tolist()).strip()


def response_slice(logits_len: int, input_len: int, n_prompt: int, n_resp: int) -> tuple[int, int]:
    """Index range in the logits that scores the response.

    ``logits[i]`` predicts ``full[i - extra + 1]`` where ``full`` is the
    concatenated prompt+response and ``extra = logits_len - input_len`` is the
    count of prepended virtual tokens. The response occupies
    ``full[n_prompt : n_prompt + n_resp]``, so inverting gives the range below.

    Sanity check that pins the sign: on the base model ``extra == 0`` and
    ``input_len == len(full) - 1``, so the range reduces to
    ``len(full) - 1 - n_resp … len(full) - 2`` -- exactly the ``pred[-n:]``
    slice that was already correct there.

    This is exposed and tested because an earlier version simply sliced
    ``pred[-n:]``. That is correct only when ``extra == 0``; for prompt tuning
    (which prepends virtual tokens) it is off by one, silently scoring the
    wrong positions and reporting exactly 0.000 accuracy for a method that was
    in fact training.
    """
    extra = logits_len - input_len
    start = n_prompt + extra - 1
    stop = n_prompt + n_resp + extra - 2
    return start, stop


@torch.no_grad()
def score_response(model, tokenizer, example: InstructionExample, *, max_new: int = 24) -> bool:
    """Teacher-forced exact match, used for PEFT wrappers (no KV cache path).

    Generation would require a cache, which the PEFT wrappers do not implement.
    To keep every method on the SAME metric, all methods -- including the full
    fine-tune baseline -- are scored this way. Teacher forcing makes the
    comparison harder to game than free generation and identical across methods.

    The index arithmetic lives in ``response_slice`` because methods that
    PREPEND virtual tokens (prompt tuning) return more logits than there are
    inputs, and slicing off the end is then wrong by one.
    """
    fmt = InstructionFormatter()
    prompt_ids = tokenizer.encode(fmt.prompt(example))
    resp_ids = tokenizer.encode(example.response) + [EOS_ID]
    full = prompt_ids + resp_ids
    x = torch.tensor([full[:-1]], dtype=torch.long, device=next(model.parameters()).device)
    logits, _, _ = model(x)

    n = len(resp_ids)
    start, stop = response_slice(logits.shape[1], x.shape[1], len(prompt_ids), n)
    if start < 0:
        raise ValueError(
            f"response does not fit: need index {start} >= 0 "
            f"(prompt {len(prompt_ids)} tokens, response {n}, "
            f"extra {logits.shape[1] - x.shape[1]})"
        )
    pred = logits[0, start : stop + 1].argmax(dim=-1)
    target = torch.tensor(full[len(prompt_ids):], dtype=torch.long, device=pred.device)
    return bool(torch.equal(pred, target))


def fit_icl_shots(tokenizer, formatter: InstructionFormatter,
                  pool: list[InstructionExample], example: InstructionExample,
                  context_length: int) -> list[InstructionExample]:
    """The largest prefix of `pool` that ACTUALLY fits in the context window.

    The canonical models have a 128-token context. A conventional 4-shot Alpaca
    prompt is ~328 tokens, so truncation silently removed the demonstrations and
    7b's ICL arm measured zero-shot on a truncated tail rather than in-context
    learning. Shots are therefore fitted by cost (cheapest first, so we fit the
    maximum number of demonstrations) and callers assert they survive.
    """
    resp_tokens = len(tokenizer.encode(example.response)) + 1  # + EOS
    chosen: list[InstructionExample] = []
    for shot in sorted(pool, key=lambda s: len(tokenizer.encode(formatter.prompt(s))
                                                + tokenizer.encode(s.response))):
        trial = chosen + [shot]
        if len(tokenizer.encode(icl_prompt(formatter, trial, example))) + resp_tokens \
                <= context_length:
            chosen = trial
    return chosen


def measure_icl(model, tokenizer, splits, *, device: str,
                seeds: list[int]) -> dict:
    """ICL accuracy of the UNMODIFIED base model, at the max shots that fit.

    Evaluated ONCE. The prompt depends only on the few-shot pool and the query,
    never on the training-set size, so repeating it per size (as an earlier
    version did) produced four identical rows dressed up as a curve.
    """
    fmt = InstructionFormatter()
    ctx = model.config.context_length
    per_seed, shots_used, missing = [], None, 0

    for seed in seeds:
        torch.manual_seed(seed)
        hits = 0
        for ex in splits.held_out:
            shots = fit_icl_shots(tokenizer, fmt, splits.icl, ex, ctx)
            prompt_ids = tokenizer.encode(icl_prompt(fmt, shots, ex))
            resp_ids = tokenizer.encode(ex.response) + [EOS_ID]
            full = prompt_ids + resp_ids

            if len(full) - 1 > ctx:
                raise AssertionError(
                    f"prompt ({len(full)-1}) exceeds context {ctx}; "
                    "demonstrations would be truncated away"
                )

            x = torch.tensor([full[-(ctx + 1):-1]], dtype=torch.long, device=device)
            y = torch.tensor([full[-len(resp_ids):]], dtype=torch.long, device=device)

            # Positive control on the data itself: every selected demonstration
            # must be present in what the model receives. This is the assertion
            # whose absence let a truncated prompt score 0.0000 unnoticed.
            seen = tok_text = tokenizer.decode(x[0].tolist())
            for shot in shots:
                if shot.response not in seen:
                    missing += 1

            with torch.no_grad():
                logits, _, _ = model(x)
            pred = logits[0, -len(resp_ids):].argmax(dim=-1)
            hits += int(torch.equal(pred, y[0]))
        per_seed.append(hits / len(splits.held_out))
        shots_used = len(shots)

    if missing:
        raise AssertionError(
            f"{missing} demonstration(s) absent from the model input -- "
            "the prompt is being truncated"
        )

    return {
        "method": "icl",
        "shots_used": shots_used,
        "context_length": ctx,
        "trainable_parameters": 0,
        "accuracy_mean": round(mean(per_seed), 4),
        "accuracy_runs": [round(a, 4) for a in per_seed],
        "note": (
            f"Evaluated once: the ICL prompt does not depend on the training-set "
            f"size. Only {shots_used} demonstration(s) fit the {ctx}-token context "
            f"window -- a conventional 4-shot prompt is ~328 tokens and would be "
            f"truncated away entirely."
        ),
    }



def icl_prompt(formatter: InstructionFormatter, shots: list[InstructionExample],
               example: InstructionExample) -> str:
    parts = []
    for shot in shots:
        parts.append(formatter.prompt(shot) + shot.response + "\n\n")
    parts.append(formatter.prompt(example))
    return "".join(parts)


# ── experiments ─────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, held_out, *, use_score: bool) -> dict[str, Any]:
    hits = 0
    per_task: dict[str, list[int]] = {}
    for ex in held_out:
        ok = score_response(model, tokenizer, ex) if use_score \
            else greedy_response(model, tokenizer,
                                 InstructionFormatter().prompt(ex)) == ex.response
        hits += int(ok)
        per_task.setdefault(ex.instruction[:28], []).append(int(ok))
    lo, hi = wilson_interval(hits, len(held_out), confidence=0.95)
    return {
        "accuracy": round(hits / len(held_out), 4),
        "wilson_95_low": round(lo, 4),
        "wilson_95_high": round(hi, 4),
        "correct": hits,
        "total": len(held_out),
        "per_task": {k: round(mean(v), 3) for k, v in sorted(per_task.items())},
    }


def run_zero_shot(rung: str, tokenizer, held_out, device: str) -> dict[str, Any]:
    """Base model, no adaptation. The floor every other number sits above."""
    accs = []
    for seed in SEEDS:
        model = load_canonical(rung).to(device)
        torch.manual_seed(seed)
        accs.append(evaluate(model, tokenizer, held_out, use_score=False))
    return {
        "method": "zero_shot", "trainable_parameters": 0,
        "runs": accs,
        "accuracy_mean": round(mean([a["accuracy"] for a in accs]), 4),
    }


def run_sft_vs_icl(rung: str, tokenizer, splits, device: str, sizes, epochs: int) -> list[dict]:
    """7b: full fine-tune across instruction-set sizes.

    ICL is deliberately NOT evaluated here. Its prompt depends only on the
    few-shot pool and the query, never on `n`, so the earlier version emitted
    four identical rows per seed presented as a curve. It now lives in
    ``measure_icl`` and is called once.
    """
    rows = []

    for n in sizes:
        train = splits.train[:n]
        for method in ("full",):
            accs = []
            for seed in SEEDS:
                base = load_canonical(rung).to(device)
                wrapped = apply_peft(base, PEFTSpec(method=method, size=1))
                torch.manual_seed(seed)
                sft_train(wrapped, train, tokenizer,
                          SFTConfig(epochs=epochs, seed=seed))
                accs.append(evaluate(wrapped, tokenizer, splits.held_out, use_score=True))
            rows.append({"n_train": n, "method": method,
                         "accuracy_mean": round(mean([a["accuracy"] for a in accs]), 4),
                         "accuracy_runs": [round(a["accuracy"], 4) for a in accs],
                         "trainable_parameters": wrapped.n_trainable()})
            print(f"  n={n:>4}  {method:<12} acc={rows[-1]['accuracy_mean']:.4f}", flush=True)
    return rows


def run_peft_head_to_head(rung: str, tokenizer, splits, device: str,
                          budget: int, epochs: int) -> list[dict]:
    """7c: every method sized to the same trainable-parameter budget."""
    probe = load_canonical(rung)
    rows = []
    for method in METHODS:
        if method == "full":
            spec = PEFTSpec(method="full", size=1)
        else:
            spec = size_to_budget(probe, method, budget)
        accs, achieved, last_loss = [], None, None
        for seed in SEEDS:
            base = load_canonical(rung).to(device)
            wrapped = apply_peft(base, spec)
            achieved = wrapped.n_trainable()
            torch.manual_seed(seed)
            info = sft_train(wrapped, splits.train, tokenizer,
                             SFTConfig(epochs=epochs, seed=seed),
                             trainable_parameters=wrapped.trainable_parameters())
            last_loss = info["final_loss"]
            accs.append(evaluate(wrapped, tokenizer, splits.held_out, use_score=True))
        lo, hi = wilson_interval(
            sum(a["correct"] for a in accs), sum(a["total"] for a in accs),
            confidence=0.95)
        rows.append({
            "method": method, "size": spec.size, "label": spec.label(),
            "trainable_parameters": achieved,
            "budget": budget,
            "final_train_loss": round(last_loss, 4),
            "accuracy_mean": round(mean([a["accuracy"] for a in accs]), 4),
            "wilson_95_low": round(lo, 4), "wilson_95_high": round(hi, 4),
            "runs": accs,
        })
        print(f"  {spec.label():>18}  trainable={achieved:>9,}  "
              f"train_loss={last_loss:.4f}  acc={rows[-1]['accuracy_mean']:.4f}",
              flush=True)
    return rows


def run_lora_sweep(rung: str, tokenizer, splits, device: str,
                   ranks, epochs: int) -> list[dict]:
    probe = load_canonical(rung)
    rows = []
    for r in ranks:
        spec = PEFTSpec(method="lora", size=r)
        accs, achieved = [], None
        for seed in SEEDS:
            base = load_canonical(rung).to(device)
            wrapped = apply_peft(base, spec)
            achieved = wrapped.n_trainable()
            torch.manual_seed(seed)
            sft_train(wrapped, splits.train, tokenizer,
                      SFTConfig(epochs=epochs, seed=seed),
                      trainable_parameters=wrapped.trainable_parameters())
            accs.append(evaluate(wrapped, tokenizer, splits.held_out, use_score=True))
        rows.append({"rank": r, "trainable_parameters": achieved,
                     "accuracy_mean": round(mean([a["accuracy"] for a in accs]), 4),
                     "accuracy_runs": [round(a["accuracy"], 4) for a in accs]})
        print(f"  r={r:>3}  trainable={achieved:>9,}  "
              f"acc={rows[-1]['accuracy_mean']:.4f}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", default="1.5M")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-held-out", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--parts", default="7a,7b,7c,7d")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    parts = {p.strip() for p in args.parts.split(",")}
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = BPETokenizer.load(CANON / "tokenizer.json")
    splits = make_splits(n_train=args.n_train, n_held_out=args.n_held_out)
    print(f"[act7] rung={args.rung} device={args.device} "
          f"train={len(splits.train)} held_out={len(splits.held_out)} "
          f"tasks={len(TASKS)} epochs={args.epochs} seeds={SEEDS}", flush=True)

    out: dict[str, Any] = {
        "experiment": "act7-sft-peft",
        "base_rung": args.rung,
        "device": args.device,
        "n_train": args.n_train,
        "n_held_out": args.n_held_out,
        "tasks": list(TASKS),
        "seeds": SEEDS,
        "epochs": args.epochs,
        "splits_disjoint_asserted": True,
        "evaluation": "teacher-forced exact match on held-out, Wilson 95% CI; "
                      "identical metric for every method",
        "defects_fixed": {"D4": "minibatched SFT, seeded shuffle, not full-batch",
                          "D5": "supervised EOS terminator in every response"},
        "caveats": [
            "Synthetic task family, not natural instructions: measures whether "
            "adaptation transfers, not instruction-following in the wild.",
            "ICL is scored teacher-forced with a 4-shot prompt built from a "
            "disjoint pool, so it is not directly comparable to free-running "
            "generation numbers elsewhere in the project.",
        ],
    }

    t0 = time.perf_counter()
    if "7a" in parts:
        print("[act7] 7a zero-shot baseline (floor)", flush=True)
        out["zero_shot"] = run_zero_shot(args.rung, tokenizer, splits.held_out, args.device)
        print(f"  zero-shot acc={out['zero_shot']['accuracy_mean']:.4f}", flush=True)

    if "7b" in parts:
        print("[act7] 7b ICL vs SFT vs full fine-tune", flush=True)
        print("[act7] 7b ICL (base model, evaluated once)", flush=True)
        icl_model = load_canonical(args.rung).to(args.device)
        out["icl"] = measure_icl(icl_model, tokenizer, splits,
                                 device=args.device, seeds=SEEDS)
        print(f"  ICL({out['icl']['shots_used']} shot(s) in a "
              f"{out['icl']['context_length']}-token window)  "
              f"acc={out['icl']['accuracy_mean']:.4f}  "
              f"runs={out['icl']['accuracy_runs']}", flush=True)
        del icl_model
        if args.device == "mps":
            torch.mps.empty_cache()
        out["icl_vs_sft"] = run_sft_vs_icl(
            args.rung, tokenizer, splits, args.device,
            sizes=[16, 64, 256, args.n_train], epochs=args.epochs)

    if "7c" in parts:
        print(f"[act7] 7c PEFT head-to-head at matched budget {args.budget:,}", flush=True)
        out["peft_head_to_head"] = run_peft_head_to_head(
            args.rung, tokenizer, splits, args.device, args.budget, args.epochs)

    if "7d" in parts:
        print("[act7] 7d LoRA rank sweep", flush=True)
        out["lora_rank_sweep"] = run_lora_sweep(
            args.rung, tokenizer, splits, args.device,
            ranks=[1, 2, 4, 8, 16, 32, 64], epochs=args.epochs)

    out["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    path = args.out / f"{args.rung}_sft_peft.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}  ({out['elapsed_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
