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
        num_audio_tokens: int = 8,
    ):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(pretrained)
        self.num_audio_tokens = num_audio_tokens
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
        self.token_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, audio_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(input_values=audio_values)
        hidden_states = outputs.last_hidden_state
        pooled = self.attention_pool(hidden_states)
        global_embedding = self.projector(pooled)

        token_states = hidden_states.transpose(1, 2)
        token_states = torch.nn.functional.adaptive_avg_pool1d(token_states, self.num_audio_tokens)
        token_states = token_states.transpose(1, 2)
        token_embeddings = self.token_projector(token_states)
        audio_tokens = torch.cat([global_embedding.unsqueeze(1), token_embeddings], dim=1)
        return global_embedding, audio_tokens
