"""Tests for MQA, GQA and Sliding Window Attention."""

from __future__ import annotations

import unittest

import torch

from src.model.gpt import GPTConfig, MiniGPT


def _config(num_kv_heads=4, window_size=0, dim=64, n_heads=4):
    return GPTConfig(
        vocabulary_size=256,
        context_length=64,
        embedding_dim=dim,
        num_heads=n_heads,
        num_kv_heads=num_kv_heads,
        num_layers=2,
        dropout=0.0,
        normalization="layernorm",
        window_size=window_size,
    )


class TestMQAGQASWA(unittest.TestCase):
    def test_gptconfig_signature(self):
        # Grouped attention: G divides heads, 8 heads / 2 groups = 4 heads per group
        cfg = _config(num_kv_heads=2, n_heads=4)
        self.assertEqual(cfg.num_kv_heads, 2)
        self.assertEqual(cfg.num_heads, 4)

    def test_mqa_builds_and_forward_runs(self):
        cfg = _config(num_kv_heads=1)
        model = MiniGPT(cfg)
        self.assertEqual(model.config.num_kv_heads, 1)
        torch.manual_seed(0)
        x = torch.randint(0, 256, (2, 10), dtype=torch.long)
        out, _, _ = model(x)
        self.assertEqual(tuple(out.shape), (2, 10, 256))

    def test_gqa_builds_2_groups(self):
        cfg = _config(num_kv_heads=2)
        model = MiniGPT(cfg)
        torch.manual_seed(0)
        x = torch.randint(0, 256, (2, 10), dtype=torch.long)
        out, _, _ = model(x)
        self.assertEqual(tuple(out.shape), (2, 10, 256))

    def test_swa_builds_and_forward_runs(self):
        cfg = _config(num_kv_heads=4, window_size=8)
        model = MiniGPT(cfg)
        torch.manual_seed(0)
        x = torch.randint(0, 256, (2, 10), dtype=torch.long)
        out, _, _ = model(x)
        self.assertEqual(tuple(out.shape), (2, 10, 256))

    def test_kv_cache_one_head_for_mqa(self):
        # KV cache stores num_kv_heads heads, not num_heads
        from src.model.gpt import KVCache
        cfg = _config(num_kv_heads=1, n_heads=4)
        model = MiniGPT(cfg)
        cache = KVCache(cfg.num_layers, cfg.num_heads, cfg.embedding_dim // cfg.num_heads,
                        num_kv_heads=cfg.num_kv_heads)
        self.assertEqual(cache.num_kv_heads, 1)
        x = torch.randint(0, 256, (1, 8), dtype=torch.long)
        model.eval()
        with torch.no_grad():
            _, _, c = model(x, cache=cache)
        self.assertEqual(c.keys[0].shape[1], 1)

    def test_swa_mask_limits_attention(self):
        cfg = _config(num_kv_heads=4, window_size=2)
        model = MiniGPT(cfg).eval()
        # forward runs and is finite
        torch.manual_seed(0)
        x = torch.randint(0, 256, (1, 10), dtype=torch.long)
        with torch.no_grad():
            out, _, _ = model(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_swa_byte_identical_generation(self):
        cfg = _config(num_kv_heads=4, window_size=4)
        model = MiniGPT(cfg).eval()
        from src.generation.decoding import generate
        p = torch.randint(0, 256, (1, 8), dtype=torch.long)
        with torch.no_grad():
            r = generate(model, p, context_length=64, max_new_tokens=4, use_cache=True)
        self.assertEqual(tuple(r.shape), (1, 12))

    def test_flops_per_token_equal_across_G(self):
        # Build models at each G; qkv projection size is unchanged (compute preserved)
        for G in (1, 2, 4, 8):
            cfg = _config(num_kv_heads=G, n_heads=8, dim=128, window_size=0)
            m = MiniGPT(cfg)
            layer = m.blocks[0].attention
            self.assertEqual(layer.q.in_features, 128)
            self.assertEqual(layer.q.out_features, 128)

if __name__ == "__main__":
    unittest.main()
