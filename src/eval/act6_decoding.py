"""Act 6 — decoding sweep, best-of-n and self-consistency.

Three experiments, all inference-only on the pinned canonical checkpoints.

6a  Decoding sweep. Generate from real C4 prompts under greedy, beam search,
    top-k, top-p and temperature, and measure the quality/diversity trade-off.
    Quality is bits-per-character of the GENERATED text under a FIXED reference
    model; diversity is distinct-2 and repeated-4-gram rate. The reference is
    held constant across every configuration so the axis is comparable.

6b  Best-of-n. Sample n continuations and rerank by mean log-probability under
    the model itself. Reports whether likelihood reranking actually improves
    quality, and what it costs in forward passes.

6c  Self-consistency on MNLI (a real, verifiable, 3-way task). The model is
    zero-shot and was never trained on MNLI, so the expected result is near
    chance. That is a legitimate measurement, not a failure: it establishes
    that self-consistency cannot rescue a model with no task competence, and it
    exercises the vote-and-interval machinery honestly.

Honest caveats, carried in the output JSON:
  * The reference model for 6a/6b is the model under test, so the quality axis
    is self-preferring. A different judge would be better; we have only these.
  * Context window is 128, prompts are 32 tokens, continuations 32 tokens.
  * n>=3 seeds per configuration; stochastic configurations are averaged.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
from torch.nn import functional as F

from src.bench.stats import wilson_interval
from src.generation.decoding import beam_search, generate
from src.model.gpt import GPTConfig, MiniGPT
from src.tokenizer.bpe import BPETokenizer

CANON = Path("checkpoints/canonical")
OUT = Path("results/v2/act6")
C4 = Path("data/v2/c4_en_shard_0.jsonl")
MNLI = Path("data/v2/glue_mnli_train.jsonl")

N_PROMPTS = 40
N_MNLI = 300
MAX_NEW = 32
SEEDS = [42, 43, 44]


# ── model / data ────────────────────────────────────────────────────────────

def load_canonical(rung: str):
    payload = torch.load(CANON / rung / "model.pt", map_location="cpu", weights_only=False)
    model = MiniGPT(GPTConfig(**payload["model_config"])).eval()
    model.load_state_dict(payload["model_state"])
    return model


def c4_prompts(tok, n: int = N_PROMPTS, tokens: int = 32) -> list[list[int]]:
    """Real documents from the C4 shard, truncated to `tokens` BPE tokens."""
    out: list[list[int]] = []
    with C4.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ids = tok.encode(json.loads(line).get("text", ""))
            if len(ids) >= tokens:
                out.append(ids[:tokens])
            if len(out) >= n:
                break
    return out


# ── generation metrics ──────────────────────────────────────────────────────

def _ngrams(seq: list[int], n: int) -> list[tuple[int, ...]]:
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def repetition_rate(seq: list[int], n: int = 4) -> float:
    grams = _ngrams(seq, n)
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def distinct2(seq: list[int]) -> float:
    grams = _ngrams(seq, 2)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


@torch.no_grad()
def bits_per_character(model, ids: list[int], tok) -> float:
    """Bits per character of `ids` under `model`, teacher-forced.

    Length-normalised in CHARACTERS, not tokens, so it is comparable across
    tokenizers and across generations of different surface length.
    """
    if len(ids) < 2:
        return float("nan")
    dev = next(model.parameters()).device
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=dev)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=dev)
    logits, _, _ = model(x)
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                          reduction="sum")
    n_chars = max(1, len(tok.decode(ids)))
    return float(nll / n_chars / math.log(2))


@torch.no_grad()
def mean_logprob(model, ids: list[int]) -> float:
    dev = next(model.parameters()).device
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=dev)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=dev)
    logits, _, _ = model(x)
    lp = F.log_softmax(logits.reshape(-1, logits.size(-1)), dim=-1)
    return float(lp.gather(1, y.reshape(-1, 1)).mean())


# ── 6a decoding sweep ───────────────────────────────────────────────────────

def decoding_configs() -> list[dict[str, Any]]:
    cfgs: list[dict[str, Any]] = [
        {"name": "greedy", "kind": "greedy"},
    ]
    for w in (2, 4, 8):
        cfgs.append({"name": f"beam{w}", "kind": "beam", "beam_width": w})
    for k in (5, 20, 50):
        cfgs.append({"name": f"topk{k}", "kind": "sample", "top_k": k})
    for p in (0.3, 0.5, 0.8, 0.95):
        cfgs.append({"name": f"topp{p}", "kind": "sample", "top_p": p})
    for t in (0.5, 0.7, 1.0, 1.3):
        cfgs.append({"name": f"temp{t}", "kind": "sample", "temperature": t})
    return cfgs


def warmup(model, prompts) -> None:
    """Pay MPS graph-compilation cost before any timing is recorded.

    Without this the FIRST configuration measured absorbs the one-time cost and
    looks artificially slow or fast depending on where it sits in the sweep.
    The earlier Act 6 cost column was measured cold, which is one reason the
    numbers did not reconcile with per-path microbenchmarks.
    """
    ctx = model.config.context_length
    dev = next(model.parameters()).device
    prompt = torch.tensor([prompts[0]], dtype=torch.long, device=dev)
    for _ in range(3):
        generate(model, prompt, context_length=ctx, max_new_tokens=MAX_NEW,
                 use_cache=True, do_sample=True, temperature=1.0)
        beam_search(model, prompt, context_length=ctx, max_new_tokens=MAX_NEW,
                    beam_width=2, use_cache=True)
    if dev.type == "mps":
        torch.mps.synchronize()


def run_sweep(model, tok, prompts, reference) -> list[dict[str, Any]]:
    ctx = model.config.context_length
    dev = next(model.parameters()).device
    warmup(model, prompts)
    results = []
    for cfg in decoding_configs():
        bpcs, reps, dis2 = [], [], []
        t0 = time.perf_counter()
        for seed in SEEDS:
            for ids in prompts:
                prompt = torch.tensor([ids], dtype=torch.long, device=dev)
                torch.manual_seed(seed)
                if cfg["kind"] == "greedy":
                    out = generate(model, prompt, context_length=ctx,
                                   max_new_tokens=MAX_NEW, use_cache=True)
                elif cfg["kind"] == "beam":
                    out = beam_search(model, prompt, context_length=ctx,
                                      max_new_tokens=MAX_NEW,
                                      beam_width=cfg["beam_width"],
                                      use_cache=True)
                else:
                    out = generate(model, prompt, context_length=ctx,
                                   max_new_tokens=MAX_NEW, use_cache=True,
                                   do_sample=True,
                                   top_k=cfg.get("top_k"),
                                   top_p=cfg.get("top_p"),
                                   temperature=cfg.get("temperature", 1.0))
                gen = out[0, prompt.shape[1]:].tolist()
                if not gen:
                    continue
                bpcs.append(bits_per_character(reference, ids + gen, tok))
                reps.append(repetition_rate(gen))
                dis2.append(distinct2(gen))
        elapsed = time.perf_counter() - t0
        results.append({
            "config": cfg["name"],
            "bits_per_char_reference": round(mean(bpcs), 4),
            "bpc_sd": round(stdev(bpcs), 4) if len(bpcs) > 1 else 0.0,
            "repetition_rate_4gram": round(mean(reps), 4),
            "distinct2": round(mean(dis2), 4),
            "n_generations": len(bpcs),
            "seconds": round(elapsed, 1),
        })
        print(f"  {cfg['name']:>10}  bpc={results[-1]['bits_per_char_reference']:.4f}"
              f"  rep4={results[-1]['repetition_rate_4gram']:.4f}"
              f"  distinct2={results[-1]['distinct2']:.4f}", flush=True)
    return results


# ── 6b best-of-n ────────────────────────────────────────────────────────────

@torch.no_grad()
def best_of_n(model, tok, prompts, reference, ns=(1, 2, 4, 8, 16)) -> list[dict[str, Any]]:
    ctx = model.config.context_length
    dev = next(model.parameters()).device
    rows = []
    for n in ns:
        picked_bpc, mean_bpc = [], []
        for seed in SEEDS:
            for ids in prompts:
                prompt = torch.tensor([ids], dtype=torch.long, device=dev)
                torch.manual_seed(seed)
                cands = []
                for _ in range(n):
                    out = generate(model, prompt, context_length=ctx,
                                   max_new_tokens=MAX_NEW, use_cache=True,
                                   do_sample=True, temperature=1.0)
                    cands.append(out[0, prompt.shape[1]:].tolist())
                cands = [c for c in cands if c]
                if not cands:
                    continue
                # rerank by mean log-probability under the model itself
                best = max(cands, key=lambda c: mean_logprob(model, ids + c))
                picked_bpc.append(bits_per_character(reference, ids + best, tok))
                mean_bpc.append(mean(bits_per_character(reference, ids + c, tok)
                                     for c in cands))
        rows.append({
            "n": n,
            "bpc_of_reranked_pick": round(mean(picked_bpc), 4),
            "bpc_mean_of_all_samples": round(mean(mean_bpc), 4),
            "improvement_bits": round(mean(mean_bpc) - mean(picked_bpc), 4),
            "n_generations": len(picked_bpc),
        })
        print(f"  best-of-{n:>2}  picked={rows[-1]['bpc_of_reranked_pick']:.4f}"
              f"  mean_sample={rows[-1]['bpc_mean_of_all_samples']:.4f}"
              f"  gain={rows[-1]['improvement_bits']:+.4f} bits", flush=True)
    return rows


# ── 6c self-consistency on MNLI ────────────────────────────────────────────

VERBALIZERS = ["Yes,", "Maybe,", "No,"]
LABELS = ["entailment", "neutral", "contradiction"]


def load_mnli(n: int) -> list[tuple[str, str, int]]:
    out = []
    with MNLI.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            label = d.get("label", d.get("gold_label"))
            if label is None or label not in (0, 1, 2):
                continue
            premise, hypothesis = d.get("premise"), d.get("hypothesis")
            if not premise or not hypothesis:
                continue
            out.append((premise, hypothesis, int(label)))
            if len(out) >= n:
                break
    return out


@torch.no_grad()
def answer_distribution(model, tok, premise: str, hypothesis: str) -> torch.Tensor:
    """Probability over the 3 verbalizers, from full-sequence scoring.

    Softmax over the three sequence log-probabilities, so the answer space is
    a proper 3-way distribution regardless of how the verbalizers tokenise.
    """
    ctx = model.config.context_length
    prompt_text = (
        f"{premise}\nQuestion: {hypothesis} True, Possible, or False?\nAnswer:"
    )
    ids = tok.encode(prompt_text)[-(ctx - 6):]
    scores = []
    for v in VERBALIZERS:
        vid = tok.encode(v)
        seq = ids + vid
        if len(seq) > ctx:
            seq = seq[-ctx:]
        scores.append(mean_logprob(model, seq) * len(vid))
    return F.softmax(torch.tensor(scores, dtype=torch.float32), dim=0)


def self_consistency(model, tok, items, ns=(1, 3, 5, 10, 20)) -> list[dict[str, Any]]:
    dists = [answer_distribution(model, tok, p, h) for p, h, _ in items]
    gold = torch.tensor([g for _, _, g in items])
    rows = []
    for n in ns:
        correct = 0
        total = 0
        for seed in SEEDS:
            torch.manual_seed(seed)
            preds = []
            for d in dists:
                counts = torch.multinomial(d, n, replacement=True)
                preds.append(int(torch.bincount(counts, minlength=3).argmax()))
            preds_t = torch.tensor(preds)
            correct += int((preds_t == gold).sum())
            total += len(gold)
        acc = correct / total
        lo, hi = wilson_interval(correct, total, confidence=0.95)
        rows.append({
            "n_votes": n,
            "accuracy": round(acc, 4),
            "wilson_95_low": round(lo, 4),
            "wilson_95_high": round(hi, 4),
            "correct": correct,
            "total": total,
        })
        print(f"  self-consistency n={n:>2}  acc={acc:.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]", flush=True)
    return rows


# ── driver ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", default="6.3M")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip", default="", help="comma list of parts to skip")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    OUT.mkdir(parents=True, exist_ok=True)

    tok = BPETokenizer.load(CANON / "tokenizer.json")
    model = load_canonical(args.rung).to(args.device)
    # Fixed reference for the quality axis. Same model: noted as a caveat.
    reference = load_canonical("6.3M").to(args.device)
    prompts = c4_prompts(tok)

    print(f"[act6] model={args.rung} device={args.device} prompts={len(prompts)} "
          f"seeds={SEEDS} max_new={MAX_NEW}", flush=True)

    out: dict[str, Any] = {
        "experiment": "act6-decoding",
        "model_rung": args.rung,
        "device": args.device,
        "n_prompts": len(prompts),
        "prompt_tokens": 32,
        "max_new_tokens": MAX_NEW,
        "seeds": SEEDS,
        "reference_model_for_bpc": "6.3M (same as model under test)",
        "caveats": [
            "The bits-per-character axis is scored by the model under test, so "
            "it is self-preferring. A separate judge would be better.",
            "Context window is 128; prompts are 32 tokens and continuations 32.",
        ],
    }

    if "6a" not in skip:
        print("[act6] 6a decoding sweep", flush=True)
        out["decoding_sweep"] = run_sweep(model, tok, prompts, reference)

    if "6b" not in skip:
        print("[act6] 6b best-of-n", flush=True)
        out["best_of_n"] = best_of_n(model, tok, prompts, reference)

    if "6c" not in skip:
        print("[act6] 6c self-consistency (MNLI)", flush=True)
        items = load_mnli(N_MNLI)
        out["self_consistency_mnli"] = {
            "n_items": len(items),
            "labels": LABELS,
            "verbalizers": VERBALIZERS,
            "chance_level": 1 / 3,
            "note": "zero-shot; the model was trained on C4, never on MNLI",
            "results": self_consistency(model, tok, items),
        }

    path = OUT / f"{args.rung}_decoding.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
