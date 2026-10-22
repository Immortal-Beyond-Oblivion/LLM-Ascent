"""Seq2seq encoder-decoder with attention for character-level translation.

Demonstrates the bottleneck problem: the encoder compresses the entire source
sequence into a single fixed-size vector, which limits performance on long
sequences. Attention mitigates this by allowing the decoder to attend to all
encoder states.

This is a simplified NMT-style architecture for demonstration purposes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Seq2SeqConfig:
    source_vocab_size: int
    target_vocab_size: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    max_length: int = 50
    sos_token: int = 0  # Start of sequence
    eos_token: int = 1  # End of sequence


class Encoder(nn.Module):
    """Bidirectional GRU encoder.

    Compresses the source sequence into a sequence of hidden states.
    The final hidden state is the "bottleneck" vector.
    """

    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.source_vocab_size, config.embedding_dim)
        self.gru = nn.GRU(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            source: [B, T_src] source token IDs
        Returns:
            outputs: [B, T_src, 2*H] (bidirectional outputs)
            hidden: [num_layers, B, 2*H] (final hidden state from both directions)
        """
        embedded = self.embedding(source)
        outputs, hidden = self.gru(embedded)
        # Concatenate forward and backward final hidden states
        # hidden: [num_layers*2, B, H] → [num_layers, B, 2*H]
        batch = source.size(0)
        hidden = hidden.view(self.config.num_layers, 2, batch, self.config.hidden_dim)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        return outputs, hidden


class Attention(nn.Module):
    """Bahdanau (additive) attention.

    Score: v^T tanh(W_s @ s + W_h @ h)
    Context: softmax(score) @ h
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: [B, 2*H] current decoder hidden state
            encoder_outputs: [B, T_src, 2*H]
        Returns:
            context: [B, 2*H] weighted sum of encoder outputs
            attention_weights: [B, T_src]
        """
        batch, src_len, _ = encoder_outputs.shape

        # Repeat decoder hidden for each source position
        # [B, 2*H] → [B, T_src, 2*H]
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)

        # Concatenate decoder hidden with each encoder output
        concat = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=-1)  # [B, T_src, 4*H]

        # Compute attention scores
        scores = self.v(torch.tanh(self.attn(concat))).squeeze(-1)  # [B, T_src]

        # Softmax over source positions
        attention_weights = F.softmax(scores, dim=1)  # [B, T_src]

        # Weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, 2*H]

        return context, attention_weights


class Decoder(nn.Module):
    """Attention-based GRU decoder."""

    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.target_vocab_size, config.embedding_dim)
        self.attention = Attention(config.hidden_dim * 2)  # bidirectional
        self.gru = nn.GRU(
            config.embedding_dim + config.hidden_dim * 2,  # input + context
            config.hidden_dim * 2,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_dim * 2, config.target_vocab_size)

    def forward(
        self,
        target: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target: [B, T_tgt] target token IDs
            hidden: [num_layers, B, 2*H] decoder hidden state
            encoder_outputs: [B, T_src, 2*H]
        Returns:
            logits: [B, T_tgt, V_tgt]
            attention_weights: [B, T_tgt, T_src]
        """
        batch, tgt_len = target.shape
        logits_list = []
        attention_list = []

        for t in range(tgt_len):
            # Current input token
            input_token = target[:, t].unsqueeze(1)  # [B, 1]
            embedded = self.embedding(input_token)  # [B, 1, E]

            # Attention
            context, attn_weights = self.attention(hidden[-1], encoder_outputs)  # [B, 2*H], [B, T_src]

            # GRU input: [embedded, context]
            gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)  # [B, 1, E+2*H]

            # GRU step
            output, hidden = self.gru(gru_input, hidden)

            # Output projection
            logits = self.output(output.squeeze(1))  # [B, V_tgt]
            logits_list.append(logits)
            attention_list.append(attn_weights)

        logits = torch.stack(logits_list, dim=1)  # [B, T_tgt, V_tgt]
        attention_weights = torch.stack(attention_list, dim=1)  # [B, T_tgt, T_src]
        return logits, attention_weights


class Seq2Seq(nn.Module):
    """Full seq2seq model with encoder, attention, and decoder."""

    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            source: [B, T_src] source token IDs
            target: [B, T_tgt] target token IDs (with SOS at start)
        Returns:
            logits: [B, T_tgt, V_tgt]
            attention_weights: [B, T_tgt, T_src]
        """
        encoder_outputs, hidden = self.encoder(source)
        logits, attention_weights = self.decoder(target, hidden, encoder_outputs)
        return logits, attention_weights

    def translate(self, source: torch.Tensor, max_length: int | None = None) -> list[int]:
        """Greedy decoding: translate source to target."""
        if max_length is None:
            max_length = self.config.max_length

        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(source)

            # Start with SOS token
            batch = source.size(0)
            input_token = torch.full((batch, 1), self.config.sos_token, dtype=torch.long, device=source.device)
            output_ids = [[] for _ in range(batch)]

            for _ in range(max_length):
                embedded = self.decoder.embedding(input_token)
                context, _ = self.decoder.attention(hidden[-1], encoder_outputs)
                gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
                output, hidden = self.decoder.gru(gru_input, hidden)
                logits = self.decoder.output(output.squeeze(1))

                # Greedy: pick highest probability token
                next_token = logits.argmax(dim=-1)  # [B]

                for b in range(batch):
                    token = int(next_token[b])
                    if token != self.config.eos_token:
                        output_ids[b].append(token)

                input_token = next_token.unsqueeze(1)

                # Stop if all sequences produced EOS
                if all(len(ids) >= max_length for ids in output_ids):
                    break

            return output_ids

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)
