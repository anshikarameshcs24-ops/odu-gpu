"""Temporal reasoning model for sequence-level agitation forecasting."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TemporalAgitationModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        num_cmai: int = 29,
        num_risk: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_dim = hidden_dim * 2
        self.cmai_head = nn.Sequential(
            nn.Linear(lstm_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_cmai),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(lstm_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_risk),
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(lstm_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor | None = None):
        encoded = self.pos_encoding(embeddings)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                encoded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.lstm(packed)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_output, _ = self.lstm(encoded)

        cmai_logits = self.cmai_head(lstm_output)
        risk_logits = self.risk_head(lstm_output)

        if lengths is None:
            final_timestep = lstm_output[:, -1, :]
        else:
            indices = lengths.to(lstm_output.device) - 1
            final_timestep = lstm_output[torch.arange(len(lengths), device=lstm_output.device), indices]

        trajectory_logits = self.trajectory_head(final_timestep)
        return cmai_logits, risk_logits, trajectory_logits

