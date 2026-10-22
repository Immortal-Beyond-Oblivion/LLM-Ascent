from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.rag import TextRetriever
from src.tokenizer import CharacterTokenizer


class RAGTests(unittest.TestCase):
    def test_retrieval_returns_relevant_source_and_bounded_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facts.txt"
            path.write_text("Mars is known as the red planet. Jupiter is the largest planet.", encoding="utf-8")
            retriever = TextRetriever.from_files([path], chunk_words=5, overlap_words=1)
            prompt, chunks = retriever.prompt("Which planet is red?", top_k=1, max_context_chars=120)
        self.assertIn("red planet", chunks[0].text)
        self.assertIn("Use only the supplied context", prompt)
        self.assertIn("facts.txt", prompt)


@unittest.skipIf(torch is None, "PyTorch unavailable")
class AdaptationTests(unittest.TestCase):
    def test_masked_instruction_batch_and_lora_freezing(self) -> None:
        from src.adaptation import InstructionExample, collate_instruction_batch, inject_lora
        from src.model import GPTConfig, MiniGPT

        tokenizer = CharacterTokenizer.from_text("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ :#\\n")
        batch, labels = collate_instruction_batch([InstructionExample("Say hi", "Hello")], tokenizer, max_length=32)
        self.assertEqual(tuple(batch.shape), (1, 32))
        self.assertEqual(batch.dtype, torch.long)
        self.assertTrue(torch.any(labels == -100))
        self.assertTrue(torch.any(labels != -100))
        self.assertTrue(torch.equal(labels[0, -4:], torch.tensor([tokenizer.encode("ello")[0], tokenizer.encode("ello")[1], tokenizer.encode("ello")[2], tokenizer.encode("ello")[3]])))
        model = MiniGPT(GPTConfig(tokenizer.vocabulary_size, context_length=32, embedding_dim=16, num_heads=4, num_layers=1))
        replaced = inject_lora(model, rank=2)
        self.assertTrue(replaced)
        self.assertTrue(all(not p.requires_grad for name, p in model.named_parameters() if ".base." in name))
        self.assertTrue(any(p.requires_grad for name, p in model.named_parameters() if "lora_" in name))
