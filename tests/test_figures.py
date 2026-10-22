from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.eval.figures import bits_per_character_curve, latency_table, load_comparison

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError:
    matplotlib = None


def sample_result(name: str, chars_per_token: float, final_loss: float) -> dict:
    return {
        "tokenizer": name,
        "parameters": 1000,
        "characters_per_token": chars_per_token,
        "validation_perplexity_per_character": 9.0,
        "validation_bits_per_character": final_loss / math.log(2) / chars_per_token if chars_per_token else 0.0,
        "loss_history": [
            {"step": 1.0, "validation_loss": final_loss + 1.0},
            {"step": 2.0, "validation_loss": final_loss},
        ],
        "samples": [
            {"mode": "greedy", "tokens_per_second": 100.0, "characters_per_second": 100.0 * chars_per_token},
            {"mode": "top-k sampling (k=20, T=0.8)", "tokens_per_second": 90.0, "characters_per_second": 90.0 * chars_per_token},
        ],
    }


class FigureDataTests(unittest.TestCase):
    def test_bits_per_character_curve_matches_reported_final_value(self) -> None:
        result = sample_result("bpe", 1.62, 3.4757)
        curve = bits_per_character_curve(result)
        self.assertEqual([step for step, _ in curve], [1.0, 2.0])
        self.assertAlmostEqual(curve[-1][1], result["validation_bits_per_character"], places=9)
        self.assertGreater(curve[0][1], curve[-1][1])

    def test_invalid_characters_per_token_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bits_per_character_curve(sample_result("x", 0.0, 2.0))

    def test_latency_table_keys_by_decoding_mode(self) -> None:
        table = latency_table(sample_result("bpe", 2.0, 3.0))
        self.assertEqual(set(table), {"greedy", "top-k"})
        self.assertEqual(table["greedy"]["characters_per_second"], 200.0)

    def test_load_comparison_validates_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.json"
            good.write_text(json.dumps([sample_result("a", 1.0, 2.0)]), encoding="utf-8")
            self.assertEqual(len(load_comparison(good)), 1)
            bad = Path(directory) / "bad.json"
            bad.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_comparison(bad)

    def test_real_tick_204_artifact_is_consistent_when_present(self) -> None:
        path = Path("artifacts/tick-204/comparison.json")
        if not path.exists():
            self.skipTest("artifacts/tick-204/comparison.json not present")
        for result in load_comparison(path):
            final = bits_per_character_curve(result)[-1][1]
            self.assertAlmostEqual(final, result["validation_bits_per_character"], places=6)


@unittest.skipIf(torch is None, "PyTorch unavailable")
class AttentionTests(unittest.TestCase):
    def make_model(self, **overrides):
        from src.model import GPTConfig, MiniGPT

        torch.manual_seed(0)
        config = GPTConfig(vocabulary_size=20, context_length=16, embedding_dim=16, num_heads=4, num_layers=2, **overrides)
        return MiniGPT(config)

    def test_weights_are_normalised_causal_and_shaped(self) -> None:
        from src.eval.attention import attention_weights

        model = self.make_model()
        ids = torch.randint(0, 20, (1, 8))
        weights = attention_weights(model, ids)
        self.assertEqual(tuple(weights.shape), (2, 4, 8, 8))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 8), atol=1e-5))
        self.assertEqual(float(weights.triu(diagonal=1).abs().sum()), 0.0)
        self.assertTrue(torch.allclose(weights[:, :, 0, 0], torch.ones(2, 4)))

    def test_hooks_removed_and_training_mode_restored(self) -> None:
        from src.eval.attention import attention_weights

        model = self.make_model()
        model.train()
        attention_weights(model, torch.randint(0, 20, (1, 5)))
        self.assertTrue(model.training)
        for block in model.blocks:
            self.assertEqual(len(block.attention._forward_pre_hooks), 0)

    def test_works_with_manual_attention_path_and_rmsnorm(self) -> None:
        from src.eval.attention import attention_weights

        model = self.make_model(optimized_attention=False, normalization="rmsnorm")
        weights = attention_weights(model, torch.randint(0, 20, (1, 6)))
        self.assertEqual(tuple(weights.shape), (2, 4, 6, 6))

    def test_rejects_batches_and_wrong_dtype(self) -> None:
        from src.eval.attention import attention_weights

        model = self.make_model()
        with self.assertRaises(ValueError):
            attention_weights(model, torch.randint(0, 20, (2, 5)))
        with self.assertRaises(ValueError):
            attention_weights(model, torch.zeros(1, 5))


@unittest.skipIf(torch is None or matplotlib is None, "PyTorch or matplotlib unavailable")
class RenderTests(unittest.TestCase):
    def test_all_figures_render_to_png(self) -> None:
        from src.eval.attention import attention_weights
        from src.eval.figures import plot_attention, plot_latency, plot_loss_curves, plot_perplexity_params
        from src.model import GPTConfig, MiniGPT

        results = [sample_result("character", 1.0, 2.4), sample_result("bpe", 1.62, 3.5)]
        torch.manual_seed(0)
        model = MiniGPT(GPTConfig(vocabulary_size=20, context_length=16, embedding_dim=16, num_heads=2, num_layers=2))
        weights = attention_weights(model, torch.randint(0, 20, (1, 6)))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            plot_loss_curves(results, out / "a.png")
            plot_perplexity_params(results, out / "b.png")
            plot_latency(results, out / "c.png")
            plot_attention(weights, list("abcdef"), out / "d.png")
            for name in "abcd":
                self.assertGreater((out / f"{name}.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
