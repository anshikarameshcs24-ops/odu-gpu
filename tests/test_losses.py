import torch

from src.training.losses import AsymmetricFocalLoss, CombinedCMAILoss


def test_asymmetric_focal_loss_returns_scalar():
    logits = torch.randn(4, 29)
    targets = torch.randint(0, 2, (4, 29)).float()
    loss = AsymmetricFocalLoss()(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_combined_loss_returns_components():
    cmai_logits = torch.randn(2, 29)
    risk_logits = torch.randn(2, 4)
    cmai_targets = torch.randint(0, 2, (2, 29)).float()
    risk_targets = torch.tensor([0, 1])
    total, cmai, risk = CombinedCMAILoss()(cmai_logits, risk_logits, cmai_targets, risk_targets)
    assert total.ndim == 0
    assert cmai.ndim == 0
    assert risk.ndim == 0

