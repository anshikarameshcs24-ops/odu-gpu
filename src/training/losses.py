"""Loss functions for CMAI training."""

from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        probabilities_clipped = torch.clamp(probabilities - self.clip, min=0.0)

        loss_pos = targets * torch.log(probabilities.clamp_min(1e-8)) * (1 - probabilities) ** self.gamma_pos
        loss_neg = (
            (1 - targets)
            * torch.log((1 - probabilities_clipped).clamp_min(1e-8))
            * probabilities_clipped**self.gamma_neg
        )
        return -(loss_pos + loss_neg).mean()


class CombinedCMAILoss(nn.Module):
    def __init__(
        self,
        cmai_weight: float = 1.0,
        risk_weight: float = 0.5,
        risk_class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cmai_loss = AsymmetricFocalLoss()
        self.risk_loss = nn.CrossEntropyLoss(weight=risk_class_weights)
        self.cmai_weight = cmai_weight
        self.risk_weight = risk_weight

    def forward(
        self,
        cmai_logits: torch.Tensor,
        risk_logits: torch.Tensor,
        cmai_targets: torch.Tensor,
        risk_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cmai_loss = self.cmai_loss(cmai_logits, cmai_targets)
        risk_loss = self.risk_loss(risk_logits, risk_targets)
        total_loss = self.cmai_weight * cmai_loss + self.risk_weight * risk_loss
        return total_loss, cmai_loss, risk_loss

