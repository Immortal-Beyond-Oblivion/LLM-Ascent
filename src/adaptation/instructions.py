"""Instruction prompt formatting and loss-masked causal batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class InstructionExample:
    instruction: str
    response: str
    input_text: str = ""


class InstructionFormatter:
    """A consistent Alpaca-style prompt contract for fine-tuning and inference."""

    def prompt(self, example: InstructionExample) -> str:
        input_section = f"\n\n### Input:\n{example.input_text}" if example.input_text else ""
        return f"### Instruction:\n{example.instruction}{input_section}\n\n### Response:\n"

    def encode_supervised(self, example: InstructionExample, tokenizer: Encoder) -> tuple[list[int], list[int]]:
        prompt_ids = tokenizer.encode(self.prompt(example))
        response_ids = tokenizer.encode(example.response)
        if not response_ids:
            raise ValueError("instruction responses must not be empty")
        full_sequence = prompt_ids + response_ids
        # At position i, a causal decoder predicts full_sequence[i + 1].
        token_ids = full_sequence[:-1]
        targets = [-100] * max(0, len(prompt_ids) - 1) + response_ids
        return token_ids, targets


def collate_instruction_batch(
    examples: list[InstructionExample], tokenizer: Encoder, *, max_length: int, pad_token_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create padded long tensors; prompt tokens are excluded from cross entropy."""
    if not examples or max_length <= 1:
        raise ValueError("examples must be non-empty and max_length greater than one")
    formatter = InstructionFormatter()
    rows, labels = [], []
    for example in examples:
        ids, targets = formatter.encode_supervised(example, tokenizer)
        # Keep the sequence tail: it contains the supervised response. A left
        # truncation may remove the predecessor for the first retained target,
        # so mask that one label rather than training it against a misaligned ID.
        start = max(0, len(ids) - max_length)
        row, label = ids[start : start + max_length], targets[start : start + max_length]
        if start and label and label[0] != -100:
            label[0] = -100
        rows.append(row + [pad_token_id] * (max_length - len(row)))
        labels.append(label + [-100] * (max_length - len(label)))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
