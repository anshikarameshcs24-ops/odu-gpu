"""Sensor-only temporal model for TIHM agitation risk forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn


class TIHMSensorTemporalModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        num_risk: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_dim = hidden_dim * 2
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

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None):
        projected = self.feature_proj(features)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                projected,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.lstm(packed)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_output, _ = self.lstm(projected)

        risk_logits = self.risk_head(lstm_output)
        if lengths is None:
            final_timestep = lstm_output[:, -1, :]
        else:
            indices = lengths.to(lstm_output.device) - 1
            final_timestep = lstm_output[torch.arange(len(lengths), device=lstm_output.device), indices]
        trajectory_logits = self.trajectory_head(final_timestep)
        return risk_logits, trajectory_logits

