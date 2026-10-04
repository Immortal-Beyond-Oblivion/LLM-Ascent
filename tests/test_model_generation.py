from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable")
class ModelAndGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        from src.model import GPTConfig, MiniGPT

        self.model = MiniGPT(GPTConfig(vocabulary_size=11, context_length=8, embedding_dim=16, num_heads=4, num_layers=2, dropout=0.0, normalization="rmsnorm"))

    def test_model_shape_loss_and_causal_dependence(self) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (1, 4, 11))
        self.assertTrue(torch.isfinite(loss))
        changed = tokens.clone()
        changed[0, -1] = 9
        changed_logits, _ = self.model(changed)
        self.assertTrue(torch.allclose(logits[:, :-1], changed_logits[:, :-1]))

    def test_generation_methods_respect_output_length(self) -> None:
        from src.generation import beam_search, greedy_decode
        from src.generation.decoding import generate

        prompt = torch.tensor([[1, 2]], dtype=torch.long)
        self.assertEqual(tuple(greedy_decode(self.model, prompt, context_length=8, max_new_tokens=3).shape), (1, 5))
        self.assertEqual(tuple(generate(self.model, prompt, context_length=8, max_new_tokens=3, do_sample=True, top_k=3, top_p=0.9).shape), (1, 5))
        self.assertEqual(tuple(beam_search(self.model, prompt, context_length=8, max_new_tokens=2, beam_width=2).shape), (1, 4))
