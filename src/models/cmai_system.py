"""End-to-end chunk-level CMAI model."""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from src.models.audio_encoder import CMAIAudioEncoder
from src.models.fusion import CrossModalFusion
from src.models.vision_encoder import CMAIVisionEncoder
from src.utils.cmai_labels import NUM_CMAI_CLASSES


class CMAISystem(nn.Module):
    def __init__(
        self,
        vision_ckpt: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        audio_ckpt: str = "facebook/wav2vec2-large-robust",
        embed_dim: int = 512,
        enable_gradient_checkpointing: bool = False,
        modality_dropout_prob: float = 0.0,
        num_patch_tokens: int = 8,
        num_audio_tokens: int = 8,
    ):
        super().__init__()
        self.modality_dropout_prob = modality_dropout_prob
        self.vision_enc = CMAIVisionEncoder(
            pretrained=vision_ckpt,
            embed_dim=embed_dim,
            gradient_checkpointing=enable_gradient_checkpointing,
            num_patch_tokens=num_patch_tokens,
        )
        self.audio_enc = CMAIAudioEncoder(
            pretrained=audio_ckpt,
            embed_dim=embed_dim,
            gradient_checkpointing=enable_gradient_checkpointing,
            num_audio_tokens=num_audio_tokens,
        )
        self.fusion = CrossModalFusion(embed_dim=embed_dim)
        self.cmai_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CMAI_CLASSES),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
        )

    def forward(self, pixel_values: torch.Tensor, audio_values: torch.Tensor, return_embeddings: bool = False):
        vis_embed, vis_tokens = self.vision_enc(pixel_values)
        aud_embed, aud_tokens = self.audio_enc(audio_values)

        if self.training and self.modality_dropout_prob > 0:
            draw = random.random()
            if draw < self.modality_dropout_prob:
                vis_tokens = torch.zeros_like(vis_tokens)
                vis_embed = torch.zeros_like(vis_embed)
            elif draw < self.modality_dropout_prob * 2:
                aud_tokens = torch.zeros_like(aud_tokens)
                aud_embed = torch.zeros_like(aud_embed)

            vis_tokens[:, 0, :] = vis_embed
            aud_tokens[:, 0, :] = aud_embed

        fused = self.fusion(vis_tokens, aud_tokens)

        cmai_logits = self.cmai_head(fused)
        risk_logits = self.risk_head(fused)
        if return_embeddings:
            return cmai_logits, risk_logits, fused
        return cmai_logits, risk_logits
