from __future__ import annotations

import math
import unittest

import torch

from src.model.ablations import (
    AblationConfig,
    AblationTransformer,
    RMSNorm,
    FFN,
    SinusoidalPE,
    RoPEPE,
)


class AblationComponentTests(unittest.TestCase):
    def test_rmsnorm_forward(self):
        norm = RMSNorm(dim=16)
        x = torch.randn(2, 4, 16)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_ffn_gelu(self):
        ffn = FFN(dim=16, activation="gelu")
        x = torch.randn(2, 4, 16)
        out = ffn(x)
        self.assertEqual(out.shape, x.shape)

    def test_ffn_relu(self):
        ffn = FFN(dim=16, activation="relu")
        x = torch.randn(2, 4, 16)
        out = ffn(x)
        self.assertEqual(out.shape, x.shape)

    def test_ffn_swiglu(self):
        ffn = FFN(dim=16, activation="swiglu")
        x = torch.randn(2, 4, 16)
        out = ffn(x)
        self.assertEqual(out.shape, x.shape)

    def test_sinusoidal_pe_forward(self):
        pe = SinusoidalPE(dim=16, max_len=32)
        x = torch.randn(2, 8, 16)
        out = pe(x)
        self.assertEqual(out.shape, x.shape)

    def test_rope_pe_forward(self):
        pe = RoPEPE(dim=16)
        x = torch.randn(2, 8, 16)
        out = pe(x)
        self.assertEqual(out.shape, x.shape)

    def test_ablation_transformer_forward(self):
        config = AblationConfig(
            vocabulary_size=20,
            context_length=8,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
        )
        model = AblationTransformer(config)
        inputs = torch.randint(0, 20, (2, 8))
        logits, loss = model(inputs, inputs)
        self.assertEqual(logits.shape, (2, 8, 20))
        self.assertIsNotNone(loss)

    def test_ablation_transformer_different_norms(self):
        for norm in ["layernorm", "rmsnorm", "batchnorm", "none"]:
            config = AblationConfig(
                vocabulary_size=20,
                context_length=8,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
                normalization=norm,
            )
            model = AblationTransformer(config)
            inputs = torch.randint(0, 20, (2, 8))
            logits, loss = model(inputs, inputs)
            self.assertEqual(logits.shape, (2, 8, 20))

    def test_ablation_transformer_different_pe(self):
        for pe in ["learned", "sinusoidal", "rope", "alibi", "none"]:
            config = AblationConfig(
                vocabulary_size=20,
                context_length=8,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
                positional_encoding=pe,
            )
            model = AblationTransformer(config)
            inputs = torch.randint(0, 20, (2, 8))
            logits, loss = model(inputs, inputs)
            self.assertEqual(logits.shape, (2, 8, 20))

    def test_ablation_transformer_different_ffn(self):
        for act in ["gelu", "relu", "swiglu"]:
            config = AblationConfig(
                vocabulary_size=20,
                context_length=8,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
                ffn_activation=act,
            )
            model = AblationTransformer(config)
            inputs = torch.randint(0, 20, (2, 8))
            logits, loss = model(inputs, inputs)
            self.assertEqual(logits.shape, (2, 8, 20))

    def test_pre_vs_post_norm(self):
        for pos in ["pre", "post"]:
            config = AblationConfig(
                vocabulary_size=20,
                context_length=8,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
                norm_position=pos,
            )
            model = AblationTransformer(config)
            inputs = torch.randint(0, 20, (2, 8))
            logits, loss = model(inputs, inputs)
            self.assertEqual(logits.shape, (2, 8, 20))

    def test_parameter_count(self):
        config = AblationConfig(vocabulary_size=20, embedding_dim=16, num_heads=4, num_layers=1)
        model = AblationTransformer(config)
        self.assertGreater(model.parameter_count(), 0)


if __name__ == "__main__":
    unittest.main()
