"""Wav2Vec2-based audio encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class AttentiveStatisticalPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(hidden_states), dim=1)
        mean = (weights * hidden_states).sum(dim=1)
        variance = (weights * (hidden_states - mean.unsqueeze(1)) ** 2).sum(dim=1)
        std = torch.sqrt(variance + 1e-9)
        return torch.cat([mean, std], dim=-1)


class CMAIAudioEncoder(nn.Module):
    def __init__(
        self,
        pretrained: str = "facebook/wav2vec2-large-robust",
        embed_dim: int = 512,
        freeze_feature_extractor: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(pretrained)
        if freeze_feature_extractor:
            self.backbone.feature_extractor._freeze_parameters()
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        hidden_size = self.backbone.config.hidden_size
        self.attention_pool = AttentiveStatisticalPooling(hidden_size)
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_values=audio_values)
        pooled = self.attention_pool(outputs.last_hidden_state)
        return self.projector(pooled)

