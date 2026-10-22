"""Adapters and CLI that connect the agent chain to a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.generation import greedy_decode
from src.model import GPTConfig, MiniGPT
from src.tokenizer import tokenizer_from_checkpoint

from .agent import Agent
from .memory import ConversationMemory


def make_generate_fn(model: MiniGPT, tokenizer, *, max_new_tokens: int = 80):
    """Wrap a model/tokenizer pair as ``generate(prompt) -> continuation`` (greedy)."""
    model.eval()
    device = next(model.parameters()).device

    def generate(prompt: str) -> str:
        ids = tokenizer.encode(prompt)
        output = greedy_decode(
            model, torch.tensor([ids], dtype=torch.long, device=device),
            context_length=model.config.context_length, max_new_tokens=max_new_tokens,
        )
        return tokenizer.decode(output[0, len(ids):].tolist(), skip_special_tokens=True)

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/tick-204/bpe/checkpoint.pt")
    parser.add_argument("--turns", nargs="*", help="Scripted user messages; omit for an interactive session")
    parser.add_argument("--memory-file", help="Load/save conversation memory as JSON (persists across runs)")
    parser.add_argument("--output", default="artifacts/tick-402/transcript.json")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--max-context-chars", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = MiniGPT(GPTConfig(**checkpoint["model_config"])).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    memory = ConversationMemory.load(args.memory_file) if args.memory_file and Path(args.memory_file).exists() else ConversationMemory()
    agent = Agent(make_generate_fn(model, tokenizer, max_new_tokens=args.max_new_tokens), memory, max_context_chars=args.max_context_chars)

    transcript = []

    def turn(message: str) -> None:
        response = agent.respond(message)
        transcript.append({"user": message, "assistant": response.text, "tool_calls": response.tool_calls, "tool_limit_reached": response.tool_limit_reached})
        print(f"User: {message}\nAssistant: {response.text}\n")

    if args.turns:
        for message in args.turns:
            turn(message)
    else:
        print("Interactive session. Use '/calc <expression>' for the calculator; empty line to quit.")
        while (message := input("User: ").strip()):
            turn(message)

    if args.memory_file:
        memory.save(args.memory_file)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checkpoint": args.checkpoint, "turns": transcript, "memory_turns": len(memory)}, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
