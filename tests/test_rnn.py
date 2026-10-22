from __future__ import annotations

import math
import unittest

import torch

from src.rnn.rnn import RNNConfig, RNNLanguageModel, FixedWindowNNLM
from src.rnn.lstm import LSTMConfig, LSTMLanguageModel
from src.rnn.gru import GRUConfig, GRULanguageModel
from src.rnn.seq2seq import Seq2SeqConfig, Seq2Seq


class RNNModelTests(unittest.TestCase):
    def _make_inputs(self, batch_size=2, seq_len=8, vocab_size=20):
        torch.manual_seed(42)
        return torch.randint(0, vocab_size, (batch_size, seq_len))

    def test_rnn_forward(self):
        config = RNNConfig(vocabulary_size=20, embedding_dim=8, hidden_dim=16)
        model = RNNLanguageModel(config)
        inputs = self._make_inputs()
        logits, loss = model(inputs, inputs)
        self.assertEqual(logits.shape, (2, 8, 20))
        self.assertIsNotNone(loss)
        self.assertTrue(math.isfinite(loss.item()))

    def test_lstm_forward(self):
        config = LSTMConfig(vocabulary_size=20, embedding_dim=8, hidden_dim=16)
        model = LSTMLanguageModel(config)
        inputs = self._make_inputs()
        logits, loss = model(inputs, inputs)
        self.assertEqual(logits.shape, (2, 8, 20))
        self.assertIsNotNone(loss)

    def test_gru_forward(self):
        config = GRUConfig(vocabulary_size=20, embedding_dim=8, hidden_dim=16)
        model = GRULanguageModel(config)
        inputs = self._make_inputs()
        logits, loss = model(inputs, inputs)
        self.assertEqual(logits.shape, (2, 8, 20))
        self.assertIsNotNone(loss)

    def test_fixed_window_nnlm_forward(self):
        model = FixedWindowNNLM(vocabulary_size=20, window_size=3, embedding_dim=8, hidden_dim=16)
        inputs = self._make_inputs()
        logits, loss = model(inputs, inputs)
        self.assertEqual(logits.shape, (2, 5, 20))  # seq_len - window_size predictions
        self.assertIsNotNone(loss)

    def test_parameter_count(self):
        config = RNNConfig(vocabulary_size=20, embedding_dim=8, hidden_dim=16)
        model = RNNLanguageModel(config)
        self.assertGreater(model.parameter_count(), 0)
        self.assertEqual(model.parameter_count(), sum(p.numel() for p in model.parameters()))

    def test_lstm_has_more_params_than_rnn(self):
        """LSTM has 4 gates vs RNN's 1, so more params at same hidden_dim."""
        rnn = RNNLanguageModel(RNNConfig(vocabulary_size=20, embedding_dim=8, hidden_dim=16))
        lstm = LSTMLanguageModel(LSTMConfig(vocabulary_size=20, embedding_dim=8, hidden_dim=16))
        self.assertGreater(lstm.parameter_count(), rnn.parameter_count())

    def test_gru_has_fewer_params_than_lstm(self):
        """GRU has 2 gates vs LSTM's 3, so fewer params at same hidden_dim."""
        gru = GRULanguageModel(GRUConfig(vocabulary_size=20, embedding_dim=8, hidden_dim=16))
        lstm = LSTMLanguageModel(LSTMConfig(vocabulary_size=20, embedding_dim=8, hidden_dim=16))
        self.assertLess(gru.parameter_count(), lstm.parameter_count())

    def test_rnn_backward(self):
        """RNN can backpropagate."""
        config = RNNConfig(vocabulary_size=20, embedding_dim=8, hidden_dim=16)
        model = RNNLanguageModel(config)
        inputs = self._make_inputs()
        _, loss = model(inputs, inputs)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)

    def test_lstm_backward(self):
        """LSTM can backpropagate."""
        config = LSTMConfig(vocabulary_size=20, embedding_dim=8, hidden_dim=16)
        model = LSTMLanguageModel(config)
        inputs = self._make_inputs()
        _, loss = model(inputs, inputs)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)


class Seq2SeqTests(unittest.TestCase):
    def test_seq2seq_forward(self):
        config = Seq2SeqConfig(
            source_vocab_size=30,
            target_vocab_size=25,
            embedding_dim=8,
            hidden_dim=16,
        )
        model = Seq2Seq(config)
        source = torch.randint(2, 30, (2, 6))
        target = torch.randint(2, 25, (2, 5))
        logits, attention = model(source, target)
        self.assertEqual(logits.shape, (2, 5, 25))
        self.assertEqual(attention.shape, (2, 5, 6))

    def test_seq2seq_translate(self):
        config = Seq2SeqConfig(
            source_vocab_size=30,
            target_vocab_size=25,
            embedding_dim=8,
            hidden_dim=16,
        )
        model = Seq2Seq(config)
        source = torch.randint(2, 30, (1, 6))
        output = model.translate(source, max_length=10)
        self.assertEqual(len(output), 1)
        self.assertGreater(len(output[0]), 0)

    def test_seq2seq_parameter_count(self):
        config = Seq2SeqConfig(
            source_vocab_size=30,
            target_vocab_size=25,
            embedding_dim=8,
            hidden_dim=16,
        )
        model = Seq2Seq(config)
        self.assertGreater(model.parameter_count(), 0)


class VanishingGradientTests(unittest.TestCase):
    def test_vanishing_gradient_demo(self):
        """Vanishing gradient demonstration runs and produces results."""
        from src.rnn.run_comparison import demonstrate_vanishing_gradients

        results = demonstrate_vanishing_gradients(
            vocabulary_size=20,
            embedding_dim=8,
            hidden_dim=16,
            sequence_length=20,
        )
        self.assertIn("rnn", results)
        self.assertIn("lstm", results)
        self.assertGreater(len(results["rnn"]), 0)
        self.assertGreater(len(results["lstm"]), 0)


class MatchedModelTests(unittest.TestCase):
    def test_build_matched_models(self):
        """Matched model builder produces models with similar parameter counts."""
        from src.rnn.run_comparison import build_matched_models

        models = build_matched_models(vocabulary_size=66, target_params=20000)
        self.assertIn("rnn", models)
        self.assertIn("lstm", models)
        self.assertIn("gru", models)

        # All models should have parameter counts within 2x of each other
        param_counts = [sum(p.numel() for p in m.parameters()) for m in models.values()]
        max_params = max(param_counts)
        min_params = min(param_counts)
        self.assertLess(max_params / min_params, 2.0, f"Parameter counts not matched: {param_counts}")


if __name__ == "__main__":
    unittest.main()
