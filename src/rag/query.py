"""CLI retrieval-augmented generation over local text documents."""

from __future__ import annotations

import argparse

import torch

from src.generation import greedy_decode
from src.model import GPTConfig, MiniGPT
from src.tokenizer import CharacterTokenizer

from .retriever import TextRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    tokenizer = CharacterTokenizer(checkpoint["vocabulary"])
    model = MiniGPT(GPTConfig(**checkpoint["model_config"])).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    retriever = TextRetriever.from_files(args.documents)
    prompt, sources = retriever.prompt(args.question, top_k=args.top_k, max_context_chars=model.config.context_length // 2)
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=args.device)
    output = greedy_decode(model, tokens, context_length=model.config.context_length, max_new_tokens=args.max_new_tokens)
    print("Sources:")
    for source in sources:
        print(f"- {source.source} (score={source.score:.2f})")
    print("\nAnswer:\n" + tokenizer.decode(output[0].tolist()[len(tokens[0]) :], skip_special_tokens=True))


if __name__ == "__main__":
    main()
