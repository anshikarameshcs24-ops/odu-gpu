"""Stage 3 trainer for temporal agitation modelling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.data.temporal_dataset import CMAISequenceDataset, collate_sequences, load_sequence_records
from src.models.temporal import TemporalAgitationModel
from src.training.losses import AsymmetricFocalLoss
from src.utils.config import load_yaml_config, set_seed

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency behavior
    wandb = None


def _maybe_init_wandb(config: dict) -> None:
    if config.get("use_wandb", False) and wandb is not None:
        wandb.init(project="cmai-detection", config=config, name=config.get("run_name", "stage3"))


def _maybe_log(payload: dict) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(payload)


def _maybe_finish_wandb() -> None:
    if wandb is not None and wandb.run is not None:
        wandb.finish()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["cmai_labels"].to(device)
        logits, _, _ = model(embeddings, lengths)
        probs = torch.sigmoid(logits)
        mask = torch.arange(embeddings.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
        all_preds.append((probs[mask] > 0.5).cpu().numpy())
        all_targets.append(labels[mask].cpu().numpy())

    y_pred = np.concatenate(all_preds) if all_preds else np.zeros((0, 29))
    y_true = np.concatenate(all_targets) if all_targets else np.zeros((0, 29))
    return {
        "cmai_f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0) if len(y_true) else 0.0,
        "cmai_f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) else 0.0,
    }


def train_temporal(config: dict) -> None:
    set_seed(int(config.get("seed", 42)))
    _maybe_init_wandb(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records = load_sequence_records(config["train_sequences"])
    val_records = load_sequence_records(config["val_sequences"])

    train_loader = DataLoader(
        CMAISequenceDataset(train_records, max_seq_len=30),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_sequences,
    )
    val_loader = DataLoader(
        CMAISequenceDataset(val_records, max_seq_len=30),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_sequences,
    )

    model = TemporalAgitationModel().to(device)
    cmai_criterion = AsymmetricFocalLoss()
    risk_criterion = torch.nn.CrossEntropyLoss()
    traj_criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    total_steps = max(len(train_loader) * int(config["num_epochs"]), 1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(int(total_steps * 0.05), 1), total_steps)
    scaler = GradScaler(enabled=device.type == "cuda")
    autocast_enabled = device.type == "cuda"

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(int(config["num_epochs"])):
        model.train()
        for step, batch in enumerate(train_loader):
            embeddings = batch["embeddings"].to(device)
            cmai_labels = batch["cmai_labels"].to(device)
            risk_levels = batch["risk_levels"].to(device)
            trajectory = batch["trajectory"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=autocast_enabled):
                cmai_logits, risk_logits, traj_logits = model(embeddings, lengths)
                mask = torch.arange(embeddings.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
                cmai_loss = cmai_criterion(cmai_logits[mask], cmai_labels[mask])
                risk_loss = risk_criterion(risk_logits[mask].view(-1, 4), risk_levels[mask].view(-1))
                traj_loss = traj_criterion(traj_logits, trajectory)
                loss = cmai_loss + 0.5 * risk_loss + 0.3 * traj_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % int(config.get("log_every", 10)) == 0:
                _maybe_log(
                    {
                        "train/loss": loss.item(),
                        "train/loss_cmai": cmai_loss.item(),
                        "train/loss_risk": risk_loss.item(),
                        "train/loss_traj": traj_loss.item(),
                        "epoch": epoch,
                    }
                )

        metrics = evaluate(model, val_loader, device)
        _maybe_log({f"val/{key}": value for key, value in metrics.items()})
        torch.save(model.state_dict(), output_dir / f"epoch_{epoch:02d}.pt")
        if metrics["cmai_f1_micro"] > best_f1:
            best_f1 = metrics["cmai_f1_micro"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    _maybe_finish_wandb()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_temporal(load_yaml_config(args.config))


if __name__ == "__main__":
    main()

