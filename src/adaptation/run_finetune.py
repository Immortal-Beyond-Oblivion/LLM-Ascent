"""CLI for applying LoRA instruction tuning to a baseline checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.model import GPTConfig, MiniGPT
from src.tokenizer import BPETokenizer, CharacterTokenizer, tokenizer_from_checkpoint

from .finetune import LoRAConfig, finetune_lora
from .instructions import InstructionExample


def _load_checkpoint(path: str, device: str) -> tuple[MiniGPT, CharacterTokenizer | BPETokenizer]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MiniGPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, tokenizer_from_checkpoint(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instructions", required=True, help="JSONL rows with instruction, optional input, and response")
    parser.add_argument("--output", default="artifacts/lora_adapter.pt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    examples = [InstructionExample(item["instruction"], item["response"], item.get("input", "")) for item in (json.loads(line) for line in Path(args.instructions).read_text(encoding="utf-8").splitlines() if line)]
    model, tokenizer = _load_checkpoint(args.checkpoint, args.device)
    losses = finetune_lora(model, examples, tokenizer, LoRAConfig(steps=args.steps, rank=args.rank, max_length=model.config.context_length))
    adapters = {name: tensor.cpu() for name, tensor in model.state_dict().items() if "lora_" in name}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapters": adapters, "losses": losses, "rank": args.rank}, args.output)
    print(json.dumps({"final_loss": losses[-1], "trainable_parameters": model.parameter_count(trainable_only=True)}, indent=2))


if __name__ == "__main__":
    main()
