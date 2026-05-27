"""End-to-end chunk-level CMAI model."""

from __future__ import annotations

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
    ):
        super().__init__()
        self.vision_enc = CMAIVisionEncoder(
            pretrained=vision_ckpt,
            embed_dim=embed_dim,
            gradient_checkpointing=enable_gradient_checkpointing,
        )
        self.audio_enc = CMAIAudioEncoder(
            pretrained=audio_ckpt,
            embed_dim=embed_dim,
            gradient_checkpointing=enable_gradient_checkpointing,
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
        vis_embed = self.vision_enc(pixel_values)
        aud_embed = self.audio_enc(audio_values)
        fused = self.fusion(vis_embed, aud_embed)

        cmai_logits = self.cmai_head(fused)
        risk_logits = self.risk_head(fused)
        if return_embeddings:
            return cmai_logits, risk_logits, fused
        return cmai_logits, risk_logits

