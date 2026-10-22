"""Scaling law ladder runner: resumable, checkpointed, self-evicting.

Runs the N/D ladder for Act 4 (scaling laws). Each rung trains a model at
fixed N and D, logging validation loss every ~50 steps. Checkpoints are saved
for resume, then evicted after the rung completes so a multi-hour sweep
never fills the disk.

Usage:
    .venv/bin/python -m src.scaling.ladder --probe          # 200-step probe at 8M
    .venv/bin/python -m src.scaling.ladder --stage1         # full Stage 1 ladder
    .venv/bin/python -m src.scaling.ladder --fit            # fit alpha, beta
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ──────────────────────────────────────────────────────────────
# ALiBi Transformer (fixed config across all rungs)
# ──────────────────────────────────────────────────────────────


class ALiBiAttention(nn.Module):
    """Multi-head self-attention with ALiBi positional bias.

    ALiBi adds a linear bias to attention scores: bias = -m * (i - j)
    where m is a head-specific slope. Zero parameters.
    """

    def __init__(self, dim: int, heads: int, max_len: int = 512) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        # ALiBi slopes: geometric sequence from 2^(-8/heads) to 2^(-8*heads/heads)
        slopes = torch.pow(2.0, -torch.arange(1, heads + 1) * (8.0 / heads))
        self.register_buffer("slopes", slopes)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, T, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]

        # ALiBi bias: -m_h * (query_pos - key_pos) = -m_h * distance
        # positions[i] - positions[j] >= 0 for causal (j <= i)
        positions = torch.arange(T, device=x.device)
        rel = positions.unsqueeze(1) - positions.unsqueeze(0)  # rel[a,b] = pos[a] - pos[b] = distance
        rel = rel.clamp(min=0)  # non-negative on causal side
        bias = -self.slopes.view(1, -1, 1, 1) * rel.unsqueeze(0).unsqueeze(0)  # [1, H, T, T]
        scores = scores + bias

        # Causal mask
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # [B, H, T, d]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class FFN(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, dim: int, expansion: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class ALiBiBlock(nn.Module):
    """Pre-norm Transformer block with ALiBi attention."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0, max_len: int = 512) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ALiBiAttention(dim, heads, max_len)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class ALiBiTransformer(nn.Module):
    """ALiBi Transformer language model.

    Uses pre-norm (fixed from Act 2), GELU FFN (fixed from Act 2),
    4 heads (fixed from Act 2), and ALiBi positional encoding
    (best from Act 2: tied with no-PE at 2.4851 BPC, zero parameters).
    """

    def __init__(
        self,
        vocabulary_size: int,
        context_length: int = 128,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.blocks = nn.ModuleList([
            ALiBiBlock(embedding_dim, num_heads, dropout, context_length)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocabulary_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # tied embeddings

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def non_embedding_params(self) -> int:
        """Non-embedding parameter count (Chinchilla convention)."""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.token_embedding.weight.numel()
        return total - embedding

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────
# Data loading from memmapped corpus
# ──────────────────────────────────────────────────────────────


class MemmapCorpus:
    """Memory-mapped token corpus for efficient random access."""

    def __init__(self, manifest_path: str = "artifacts/scaling/manifest.json") -> None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.mmap = np.memmap(manifest["memmap_path"], dtype=np.uint16, mode="r")
        self.total_tokens = len(self.mmap)
        self.vocab_size = manifest["tokenizer_vocab_size"]
        self.context_length = manifest.get("context_length", 128)
        # Train limit: train over [0, train_limit), validate over [train_limit, total_tokens)
        self.val_tokens = manifest.get("eval_token_subsample", 1_000_000)
        self.train_limit = self.total_tokens - self.val_tokens - self.context_length
        if self.train_limit <= 0:
            raise ValueError("train_limit <= 0: corpus too small")
        self.val_start = self.train_limit
        # Fixed validation offsets chosen ONCE from a seeded RNG, so every
        # eval event is comparable along the D-axis (no random-offset noise).
        _rng = np.random.RandomState(0)
        _val_end = max(self.val_start + 1, self.total_tokens - self.context_length - 1)
        self.fixed_val_offsets = _rng.randint(self.val_start, _val_end, size=1024).tolist()
        # Overlap check (reported in manifest)

    def _sample_offsets(
        self, n: int, lo: int, hi: int
    ) -> list[int] | np.ndarray:
        """Testable offset sampler. ``lo,hi`` define an exclusive upper-range hint."""
        return np.random.randint(lo, hi, size=n)

    def train_batch(
        self,
        batch_size: int,
        context_length: int,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a training batch from [0, train_limit) ONLY."""
        offsets = self._sample_offsets(batch_size, 0, self.train_limit).tolist()
        # Assert no training offset ever equals or exceeds train_limit
        assert all(o < self.train_limit for o in offsets), "Training offset exceeded train_limit!"
        inputs = torch.stack([
            torch.from_numpy(self.mmap[o : o + context_length].astype(np.int64))
            for o in offsets
        ]).to(device)
        targets = torch.stack([
            torch.from_numpy(self.mmap[o + 1 : o + context_length + 1].astype(np.int64))
            for o in offsets
        ]).to(device)
        return inputs, targets

    def val_batch(
        self,
        batch_size: int,
        context_length: int,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a validation batch from [train_limit, total_tokens)."""
        # Fixed (not random) validation offsets -> deterministic D-trajectory noise
        start = self.train_limit
        end = self.total_tokens - context_length - 1
        offsets = list(self.fixed_val_offsets[:batch_size])
        # Pad if a fraction larger batch than fixed pool (still deterministic)
        if len(offsets) < batch_size:
            repeats = (batch_size // len(offsets) + 1)
            offsets = (offsets * repeats)[:batch_size]
        # Assert all validation offsets are >= train_limit
        assert all(o >= self.train_limit for o in offsets), "Validation offset below train_limit!"
        inputs = torch.stack([
            torch.from_numpy(self.mmap[o : o + context_length].astype(np.int64))
            for o in offsets
        ]).to(device)
        targets = torch.stack([
            torch.from_numpy(self.mmap[o + 1 : o + context_length + 1].astype(np.int64))
            for o in offsets
        ]).to(device)
        return inputs, targets


# ──────────────────────────────────────────────────────────────
# Ladder configuration
# ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LadderRung:
    name: str
    non_embedding_params: int  # N (non-embedding, Chinchilla convention)
    tokens: int                # D (training tokens)
    seeds: list[int]
    device: str


# Stage 1 ladder (D = 20N for each rung)
# Constant aspect-ratio ladder: L = d // 32 so L*12*d^2 scales as 0.375*d^3.
# Per layer ~= 12*d^2 (attention 4d^2 + FFN 8d^2, pre-norm).
# Embedding share falls monotonically. Six rungs, 1.06 orders of magnitude.
# 3 seeds for the three smallest, 2 for the next two, 1 for the largest.
ASPECT_LADDER = {
    "0.8M": (128, 4),
    "1.5M": (160, 5),
    "2.6M": (192, 6),
    "4.1M": (224, 7),
    "6.3M": (256, 8),
    "9.0M": (288, 9),
}

LADDER = [
    LadderRung("0.8M", 786_432, 15_728_640, [42, 43, 44], "cpu"),
    LadderRung("1.5M", 1_536_000, 30_720_000, [42, 43, 44], "cpu"),
    LadderRung("2.6M", 2_654_208, 53_084_160, [42, 43], "mps"),
    LadderRung("4.1M", 4_128_768, 82_575_360, [42, 43], "mps"),
    LadderRung("6.3M", 6_291_456, 125_829_120, [42], "mps"),
    LadderRung("9.0M", 8_957_952, 89_922_240, [42], "mps"),  # D=10N (was 20N)
]


def build_model_for_rung(rung: LadderRung, vocab_size: int, context_length: int = 128) -> ALiBiTransformer:
    """Build a model from the constant aspect-ratio ladder.

    Dimension and depth are fixed per rung from ASPECT_LADDER, rather than a
    grid search that confounds width and depth across rungs.
    """
    if rung.name in ASPECT_LADDER:
        dim, layers = ASPECT_LADDER[rung.name]
    else:
        # Fallback for ad hoc / test rungs: search as before.
        best_diff = float("inf")
        best_config = (128, 4)
        for d in (128, 192, 256):
            for L in (2, 4, 6):
                tmp = ALiBiTransformer(vocab_size, context_length, d, 4, L)
                diff = abs(tmp.non_embedding_params() - rung.non_embedding_params)
                if diff < best_diff:
                    best_diff = diff
                    best_config = (d, L)
        dim, layers = best_config
    model = ALiBiTransformer(
        vocabulary_size=vocab_size,
        context_length=context_length,
        embedding_dim=dim,
        num_heads=4,
        num_layers=layers,
    )
    actual_non_emb = model.non_embedding_params()
    print(f"[ladder] {rung.name}: dim={dim} layers={layers} non_emb_params={actual_non_emb:,} (target={rung.non_embedding_params:,})")
    return model


def train_rung(
    rung: LadderRung,
    corpus: MemmapCorpus,
    *,
    context_length: int = 128,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    warmup_steps: int = 100,
    eval_interval: int = 50,
    eval_batches: int = 10,
    eval_token_subsample: int = 1_000_000,
    checkpoint_dir: str = "artifacts/scaling/checkpoints",
    save_weights: bool = False,
    pin_weights: bool = False,
) -> list[dict]:
    """Train one rung and return the loss history.

    Logs (N, D, loss) every eval_interval steps. Saves checkpoints for resume
    and evicts them after the rung completes.
    """
    device = rung.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    seed = rung.seeds[0]
    # Bug 3 fix: actually seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_dir = Path(checkpoint_dir) / rung.name / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Bug (b) fix: resume from checkpoint if exists, but only when the
    # checkpoint's architecture actually matches the current model. Otherwise a
    # different config (e.g. G=1 vs G=8 KV heads) silently loads mismatched
    # weights and crashes deep inside load_state_dict.
    start_step = 0
    history = []
    checkpoint_path = ckpt_dir / "checkpoint.pt"

    model = build_model_for_rung(rung, corpus.vocab_size, context_length)
    model = model.to(device)

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_shapes = {k: tuple(v.shape) for k, v in ckpt["model_state"].items()}
        model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        if ckpt_shapes == model_shapes:
            print(f"[ladder] {rung.name} seed={seed}: resuming from checkpoint ...")
            start_step = ckpt["completed_steps"]
            history = ckpt.get("history", [])
        else:
            print(f"[ladder] {rung.name} seed={seed}: checkpoint architecture mismatch, starting fresh")
            checkpoint_path.unlink(missing_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if checkpoint_path.exists():
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"[ladder] {rung.name} seed={seed}: resumed from step {start_step}")

    print(f"[ladder] {rung.name} seed={seed} device={device} D={rung.tokens:,}")

    # Compute steps from D: D = steps * batch_size * context_length
    steps = rung.tokens // (batch_size * context_length)
    print(f"[ladder] {rung.name}: {steps} steps")

    start_time = time.perf_counter()
    peak_memory = 0.0

    for step in range(start_step + 1, steps + 1):
        # LR schedule: warmup then cosine
        if warmup_steps > 0 and step <= warmup_steps:
            lr = learning_rate * step / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            lr = learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Bug 4 fix: training batch from [0, train_limit) ONLY
        inputs, targets = corpus.train_batch(batch_size, context_length, device)
        _out = model(inputs, targets)
        if isinstance(_out, tuple) and len(_out)==3:
            _, loss, _ = _out
        else:
            _, loss = _out
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Peak memory (Bug d fix) - track running max, not instantaneous current
        current_mem = 0.0
        if device == "mps":
            try:
                current_mem = torch.mps.current_allocated_memory() / (1024 ** 3)
            except Exception:
                try:
                    current_mem = torch.mps.driver_allocated_memory() / (1024 ** 3)
                except Exception:
                    current_mem = 0.0
        peak_memory = max(peak_memory, current_mem)

        # Eval logging
        if step == 1 or step % eval_interval == 0 or step == steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(eval_batches):
                    vi, vt = corpus.val_batch(batch_size, context_length, device)
                    _o = model(vi, vt)
                    _, vloss = (_o[0], _o[1]) if isinstance(_o,tuple) and len(_o)>=2 else (_o, None)
                    assert vloss is not None
                    val_losses.append(float(vloss.detach()))
            model.train()

            avg_val_loss = sum(val_losses) / len(val_losses)
            elapsed = time.perf_counter() - start_time
            tokens_seen = step * batch_size * context_length
            history.append({
                "step": step,
                "tokens": tokens_seen,
                "val_loss": avg_val_loss,
                "elapsed_seconds": elapsed,
            })
            print(f"[ladder] {rung.name} seed={seed} step={step}/{steps} tokens={tokens_seen:,} val_loss={avg_val_loss:.4f} elapsed={elapsed:.0f}s peak_mem={peak_memory:.2f}GB")

        # Checkpoint every 500 steps for resume
        if step % 500 == 0:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "completed_steps": step,
                "history": history,
            }, ckpt_dir / "checkpoint.pt")

    # Final checkpoint (will be evicted)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "completed_steps": steps,
        "history": history,
    }, ckpt_dir / "checkpoint.pt")

    saved_path = None
    if save_weights:
        # Retention + pinning replaces delete-after-rung eviction (which stranded
        # Acts 6-10 with no loadable weights). The first canonical script pinned a
        # step marker rather than tensors, which would have stranded 5c, so the
        # save path is gated behind a logits round-trip test.
        from src.scaling.checkpoints import CheckpointStore

        store = CheckpointStore(checkpoint_dir, keep_recent=2)
        cfg = getattr(model, "config", None)
        payload = {
            "model_state": model.state_dict(),
            "model_config": cfg.to_dict() if cfg is not None else {},
            "completed_steps": steps,
            "rung": rung.name,
            "seed": seed,
        }
        saved_path = store.save(rung.name, seed, steps, payload)
        if pin_weights:
            store.pin(rung.name, seed, steps)
        print(f"[ladder] {rung.name} seed={seed}: saved weights -> {saved_path}"
              f"{' (PINNED)' if pin_weights else ''}")
    else:
        # Legacy path: keep only the most recent rolling checkpoint, drop the rest.
        from src.scaling.checkpoints import CheckpointStore

        CheckpointStore(checkpoint_dir, keep_recent=2).evict()



    # Write metrics. Architecture fields are recorded explicitly because Act 4's
    # ladder (ALiBiTransformer, 4 heads) and Act 5's sweeps (MiniGPT, 8 heads,
    # GQA/SWA) are DIFFERENT models and losses must never be compared across them.
    arch = {}
    cfg = getattr(model, "config", None)
    if cfg is not None:
        arch = {
            "model_class": type(model).__name__,
            "num_heads": getattr(cfg, "num_heads", None),
            "num_kv_heads": getattr(cfg, "num_kv_heads", None),
            "window_size": getattr(cfg, "window_size", None),
            "embedding_dim": getattr(cfg, "embedding_dim", None),
            "num_layers": getattr(cfg, "num_layers", None),
            "context_length": getattr(cfg, "context_length", None),
            "positional_encoding": "alibi" if type(model).__name__ == "ALiBiTransformer" else "learned_absolute",
        }
    metrics = {
        "rung": rung.name,
        "architecture": arch,
        "non_embedding_params": model.non_embedding_params(),
        "embedding_params": model.token_embedding.weight.numel(),
        "total_params": model.total_params(),
        "steps": steps,
        "tokens": rung.tokens,
        "history": history,
        "device": device,
        "seed": seed,
        "peak_memory_gb": peak_memory,
    }
    metrics["saved_checkpoint"] = str(saved_path) if saved_path else None
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return history




def run_probe() -> None:
    """Run a 200-step probe at N=8M to measure throughput and memory."""
    print("[probe] Loading corpus ...")
    corpus = MemmapCorpus()
    print(f"[probe] Corpus loaded: {corpus.total_tokens:,} tokens")

    # Build 8M model
    rung = LadderRung("9.0M", 8_957_952, 200 * 32 * 128, [42], "auto")
    model = build_model_for_rung(rung, corpus.vocab_size)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    print(f"[probe] Model: {model.non_embedding_params():,} non-embedding params on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    context_length = 128
    batch_size = 32

    # Warmup
    warmup_peak = 0.0
    for _ in range(5):
        inputs, targets = corpus.train_batch(batch_size, context_length, device)
        _out = model(inputs, targets)
        _, loss, _ = (_out[0], _out[1], _out[2]) if isinstance(_out,tuple) and len(_out)==3 else (_out[0], _out[1], None)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device == "mps":
            warmup_peak = max(warmup_peak, torch.mps.current_allocated_memory() / (1024**3))

    # Timed run
    torch.mps.synchronize() if device == "mps" else torch.cuda.synchronize() if (device == "cuda" and torch.cuda.is_available()) else None
    t0 = time.perf_counter()

    peak = 0.0
    for step in range(200):
        inputs, targets = corpus.train_batch(batch_size, context_length, device)
        _out = model(inputs, targets)
        if isinstance(_out, tuple) and len(_out)==3:
            _, loss, _ = _out
        else:
            _, loss = _out
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device == "mps":
            peak = max(peak, torch.mps.current_allocated_memory() / (1024**3))
        elif device == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / (1024**3))

    torch.mps.synchronize() if device == "mps" else None
    elapsed = time.perf_counter() - t0

    tokens_per_sec = 200 * batch_size * context_length / elapsed
    steps_per_sec = 200 / elapsed

    peak_memory_gb = peak

    print(f"\n{'='*50}")
    print(f"PROBE RESULTS (N=8M, 200 steps)")
    print(f"{'='*50}")
    print(f"Device: {device}")
    print(f"Steps/sec: {steps_per_sec:.2f}")
    print(f"Tokens/sec: {tokens_per_sec:,.0f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Peak memory (probe): {peak_memory_gb:.2f} GB")
    print(f"Peak memory (warmup): {warmup_peak:.2f} GB")
    print(f"Projected 96-min run: {steps_per_sec * 96 * 60:.0f} steps")
    print(f"{'='*50}")

    # Save probe results
    probe_results = {
        "device": device,
        "steps_per_second": steps_per_sec,
        "tokens_per_second": tokens_per_sec,
        "elapsed_seconds": elapsed,
        "peak_memory_gb": round(peak_memory_gb, 3),
        "warmup_peak_memory_gb": round(warmup_peak, 3),
        "non_embedding_params": model.non_embedding_params(),
        "context_length": context_length,
        "batch_size": batch_size,
    }
    Path("artifacts/scaling").mkdir(parents=True, exist_ok=True)
    Path("artifacts/scaling/probe_results.json").write_text(
        json.dumps(probe_results, indent=2), encoding="utf-8"
    )


def run_stage1() -> None:
    """Run the full Stage 1 ladder."""
    print("[ladder] Loading corpus ...")
    corpus = MemmapCorpus()
    print(f"[ladder] Corpus loaded: {corpus.total_tokens:,} tokens")

    all_results = {}
    for rung in LADDER:
        print(f"\n{'='*50}")
        print(f"STAGE 1: {rung.name} (N={rung.non_embedding_params:,}, D={rung.tokens:,})")
        print(f"{'='*50}")

        rung_results = []
        for seed in rung.seeds:
            # Override seed
            rung = LadderRung(rung.name, rung.non_embedding_params, rung.tokens, [seed], rung.device)
            history = train_rung(rung, corpus)
            rung_results.append(history)

        all_results[rung.name] = rung_results

    # Save all results
    Path("artifacts/scaling").mkdir(parents=True, exist_ok=True)
    Path("artifacts/scaling/stage1_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    print(f"\n[ladder] Stage 1 complete. Results saved.")


# fit_scaling_laws removed (Bug c): fit.py supersedes it.
    """Fit L(N,D) = E + A/N^alpha + B/D^beta to all ladder points."""
    from scipy.optimize import curve_fit

    # Load results
    results_path = Path("artifacts/scaling/stage1_results.json")
    if not results_path.exists():
        print("[fit] No stage1_results.json found. Run --stage1 first.")
        return

    all_results = json.loads(results_path.read_text(encoding="utf-8"))

    # Collect all (N, D, loss) points
    N_vals = []
    D_vals = []
    loss_vals = []

    for rung_name, rung_runs in all_results.items():
        for history in rung_runs:
            for entry in history:
                N_vals.append(entry["non_embedding_params"])
                D_vals.append(entry["tokens"])
                loss_vals.append(entry["val_loss"])

    N_vals = np.array(N_vals, dtype=np.float64)
    D_vals = np.array(D_vals, dtype=np.float64)
    loss_vals = np.array(loss_vals, dtype=np.float64)

    print(f"[fit] Fitting {len(N_vals)} points ...")

    # Model: L(N,D) = E + A/N^alpha + B/D^beta
    def model(N, D, E, A, alpha, B, beta):
        return E + A / (N ** alpha) + B / (D ** beta)

    # Initial guess
    p0 = [loss_vals.min(), 1.0, 0.3, 1.0, 0.3]
    bounds = ([0, 0, 0, 0, 0], [loss_vals.max(), 1e6, 2, 1e6, 2])

    try:
        popt, pcov = curve_fit(model, (N_vals, D_vals), loss_vals, p0=p0, bounds=bounds, maxfev=10000)
        E, A, alpha, B, beta = popt
        perr = np.sqrt(np.diag(pcov))

        print(f"\n{'='*50}")
        print(f"SCALING LAW FIT: L(N,D) = E + A/N^alpha + B/D^beta")
        print(f"{'='*50}")
        print(f"E  = {E:.4f} ± {perr[0]:.4f}")
        print(f"A  = {A:.2f} ± {perr[1]:.2f}")
        print(f"α  = {alpha:.4f} ± {perr[2]:.4f}  (Chinchilla: 0.34)")
        print(f"B  = {B:.2f} ± {perr[3]:.2f}")
        print(f"β  = {beta:.4f} ± {perr[4]:.4f}  (Chinchilla: 0.28)")
        print(f"{'='*50}")

        # Compare with constant baseline
        constant_loss = loss_vals.mean()
        pred_loss = model(N_vals, D_vals, *popt)
        ss_res = np.sum((loss_vals - pred_loss) ** 2)
        ss_tot = np.sum((loss_vals - constant_loss) ** 2)
        r_squared = 1 - ss_res / ss_tot

        print(f"\nR² vs constant baseline: {r_squared:.4f}")
        print(f"Constant baseline loss: {constant_loss:.4f}")
        print(f"Mean fitted loss: {pred_loss.mean():.4f}")

        # Jacobian condition number
        from numpy.linalg import cond
        # Approximate Jacobian at solution
        eps = 1e-6
        J = np.zeros((len(N_vals), 5))
        for i, param_idx in enumerate(range(5)):
            p_plus = popt.copy()
            p_plus[param_idx] += eps
            J[:, i] = (model(N_vals, D_vals, *p_plus) - pred_loss) / eps
        condition = cond(J)
        print(f"Jacobian condition number: {condition:.2e}")

        # Save fit
        fit_results = {
            "E": E, "E_err": perr[0],
            "A": A, "A_err": perr[1],
            "alpha": alpha, "alpha_err": perr[2],
            "B": B, "B_err": perr[3],
            "beta": beta, "beta_err": perr[4],
            "r_squared": r_squared,
            "constant_baseline_loss": constant_loss,
            "jacobian_condition": condition,
            "n_points": len(N_vals),
        }
        Path("artifacts/scaling/fit_results.json").write_text(
            json.dumps(fit_results, indent=2), encoding="utf-8"
        )
        print(f"[fit] Results saved to artifacts/scaling/fit_results.json")

    except Exception as e:
        print(f"[fit] Fit failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaling law ladder")
    parser.add_argument("--probe", action="store_true", help="Run 200-step probe at 8M")
    parser.add_argument("--stage1", action="store_true", help="Run full Stage 1 ladder")
    parser.add_argument("--fit", action="store_true", help="Fit scaling laws")
    args = parser.parse_args()

    if args.probe:
        run_probe()
    elif args.stage1:
        run_stage1()
    elif args.fit:
        fit_scaling_laws()
    else:
        print("Specify --probe, --stage1, or --fit")
