"""VideoMAE-based vision encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VideoMAEModel


class CMAIVisionEncoder(nn.Module):
    def __init__(
        self,
        pretrained: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        embed_dim: int = 512,
        freeze_layers: int = 8,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(pretrained)
        self.hidden_size = self.backbone.config.hidden_size

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self._freeze_layers(freeze_layers)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _freeze_layers(self, num_layers: int) -> None:
        for index, layer in enumerate(self.backbone.encoder.layer):
            if index < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.projector(cls_token)

