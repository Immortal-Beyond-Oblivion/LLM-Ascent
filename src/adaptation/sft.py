"""Supervised fine-tuning for Act 7.

Fixes the two v1 defects this replaces:

**D4 — full-batch SFT.** ``finetune_lora`` fitted 3 examples in one batch for
100 steps. Here examples are minibatched and shuffled from a seeded generator,
so every example is seen on a comparable footing and the held-out split is
never touched by the optimiser.

**D5 — no response terminator.** v1 emitted responses with no end marker, so a
trained model had no way to signal "I am done" and generations ran to the token
budget. Every response here is terminated with a dedicated EOS id, and that EOS
is a supervised target: the model is explicitly trained to produce it.

Prompt tokens stay masked out of the cross-entropy (that behaviour already
lived in ``InstructionFormatter`` and is reused, not reimplemented). Loss is
therefore computed on the response only, including its terminator.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import torch

from src.adaptation.instructions import InstructionExample, InstructionFormatter

EOS_ID = 0  # dedicated terminator; also the pad id, which is masked in labels


@dataclass(frozen=True)
class SFTConfig:
    learning_rate: float = 3e-4
    epochs: int = 3
    batch_size: int = 16          # D4 fix: minibatch, not the whole split
    max_length: int = 96
    warmup_steps: int = 10
    seed: int = 42
    weight_decay: float = 0.01


def encode_example(
    example: InstructionExample, tokenizer, *, max_length: int, eos_id: int = EOS_ID
) -> tuple[list[int], list[int]]:
    """(input_ids, labels) with the prompt masked and EOS supervised."""
    formatter = InstructionFormatter()
    prompt_ids = tokenizer.encode(formatter.prompt(example))
    response_ids = tokenizer.encode(example.response) + [eos_id]
    full = prompt_ids + response_ids
    token_ids = full[:-1]
    # Positions predicting prompt tokens are masked; the EOS at the end is a
    # real target, so the model learns when to stop.
    targets = [-100] * (len(prompt_ids) - 1) + response_ids
    return token_ids, targets


def collate(
    batch: list[tuple[list[int], list[int]]], *, max_length: int, pad_id: int = EOS_ID
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, labels = [], []
    for ids, targets in batch:
        start = max(0, len(ids) - max_length)
        row, label = ids[start : start + max_length], targets[start : start + max_length]
        if start and label and label[0] != -100:
            label[0] = -100  # predecessor was truncated away
        rows.append(row + [pad_id] * (max_length - len(row)))
        labels.append(label + [-100] * (max_length - len(label)))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def _lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())


def sft_train(
    model,
    examples: list[InstructionExample],
    tokenizer,
    config: SFTConfig = SFTConfig(),
    *,
    trainable_parameters: list[torch.nn.Parameter] | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> dict[str, float]:
    """Minibatched SFT. Returns final mean loss and the loss history.

    ``trainable_parameters`` lets a PEFT method supply its own subset; when
    omitted every parameter is trained (full fine-tune).
    """
    if not examples:
        raise ValueError("SFT requires a non-empty training set")
    device = next(model.parameters()).device
    params = trainable_parameters
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters")

    max_length = min(config.max_length, model.config.context_length)
    encoded = [encode_example(e, tokenizer, max_length=max_length) for e in examples]

    n_steps = max(1, (len(encoded) // config.batch_size) * config.epochs)
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    rng = random.Random(config.seed)
    losses: list[float] = []
    step = 0
    model.train()
    for _ in range(config.epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for start in range(0, len(order) - config.batch_size + 1, config.batch_size):
            batch = [encoded[i] for i in order[start : start + config.batch_size]]
            inputs, labels = collate(batch, max_length=max_length)
            inputs, labels = inputs.to(device), labels.to(device)
            # MiniGPT returns (logits, loss, cache); PEFT wrappers return the
            # same triple. Take element 1 either way.
            out = model(inputs, labels)
            loss = out[1] if isinstance(out, (tuple, list)) else out
            assert loss is not None
            for group in optimizer.param_groups:
                group["lr"] = _lr_at(step, n_steps, config.learning_rate, config.warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if on_step:
                on_step(step, losses[-1])
            step += 1

    return {"final_loss": losses[-1] if losses else float("nan"),
            "steps": float(step),
            "n_train": float(len(examples))}
