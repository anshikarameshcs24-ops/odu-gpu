"""Train a TIHM sensor-only temporal risk model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.data.tihm_dataset import TIHMSequenceDataset, collate_tihm_sequences
from src.models.tihm_sensor_model import TIHMSensorTemporalModel
from src.utils.config import load_yaml_config, set_seed

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


def _maybe_init_wandb(config: dict) -> None:
    if config.get("use_wandb", False) and wandb is not None:
        wandb.init(project="cmai-detection", config=config, name=config.get("run_name", "stage3_tihm_sensor"))


def _maybe_log(payload: dict) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(payload)


def _maybe_finish_wandb() -> None:
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def _load_records(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_risk_pred: list[np.ndarray] = []
    all_risk_true: list[np.ndarray] = []
    all_traj_pred: list[np.ndarray] = []
    all_traj_true: list[np.ndarray] = []

    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        risk_levels = batch["risk_levels"].to(device)
        trajectory = batch["trajectory"].to(device)
        risk_logits, traj_logits = model(features, lengths)
        mask = torch.arange(features.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
        all_risk_pred.append(risk_logits.argmax(dim=-1)[mask].cpu().numpy())
        all_risk_true.append(risk_levels[mask].cpu().numpy())
        all_traj_pred.append(traj_logits.argmax(dim=-1).cpu().numpy())
        all_traj_true.append(trajectory.cpu().numpy())

    y_risk_pred = np.concatenate(all_risk_pred) if all_risk_pred else np.zeros((0,), dtype=int)
    y_risk_true = np.concatenate(all_risk_true) if all_risk_true else np.zeros((0,), dtype=int)
    y_traj_pred = np.concatenate(all_traj_pred) if all_traj_pred else np.zeros((0,), dtype=int)
    y_traj_true = np.concatenate(all_traj_true) if all_traj_true else np.zeros((0,), dtype=int)

    return {
        "risk_acc": float((y_risk_pred == y_risk_true).mean()) if len(y_risk_true) else 0.0,
        "risk_macro_f1": f1_score(y_risk_true, y_risk_pred, average="macro", zero_division=0) if len(y_risk_true) else 0.0,
        "trajectory_acc": float((y_traj_pred == y_traj_true).mean()) if len(y_traj_true) else 0.0,
    }


def train_tihm_sensor(config: dict) -> None:
    set_seed(int(config.get("seed", 42)))
    _maybe_init_wandb(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records = _load_records(config["train_sequences"])
    val_records = _load_records(config["val_sequences"])

    if not train_records:
        raise ValueError("No TIHM training sequences were found.")

    feature_dim = int(train_records[0]["feature_dim"])

    train_loader = DataLoader(
        TIHMSequenceDataset(train_records),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_tihm_sequences,
    )
    val_loader = DataLoader(
        TIHMSequenceDataset(val_records),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_tihm_sequences,
    )

    model = TIHMSensorTemporalModel(
        feature_dim=feature_dim,
        projection_dim=int(config.get("projection_dim", 256)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)

    risk_criterion = torch.nn.CrossEntropyLoss()
    traj_criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    total_steps = max(len(train_loader) * int(config["num_epochs"]), 1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(int(total_steps * 0.05), 1), total_steps)
    scaler = GradScaler(enabled=device.type == "cuda")
    autocast_enabled = device.type == "cuda"

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0

    for epoch in range(int(config["num_epochs"])):
        model.train()
        for step, batch in enumerate(train_loader):
            features = batch["features"].to(device)
            risk_levels = batch["risk_levels"].to(device)
            trajectory = batch["trajectory"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=autocast_enabled):
                risk_logits, traj_logits = model(features, lengths)
                mask = torch.arange(features.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
                risk_loss = risk_criterion(risk_logits[mask].view(-1, 4), risk_levels[mask].view(-1))
                traj_loss = traj_criterion(traj_logits, trajectory)
                loss = risk_loss + 0.3 * traj_loss

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
                        "train/loss_risk": risk_loss.item(),
                        "train/loss_traj": traj_loss.item(),
                        "epoch": epoch,
                    }
                )

        metrics = evaluate(model, val_loader, device)
        _maybe_log({f"val/{key}": value for key, value in metrics.items()})
        torch.save(model.state_dict(), output_dir / f"epoch_{epoch:02d}.pt")
        if metrics["risk_macro_f1"] > best_score:
            best_score = metrics["risk_macro_f1"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    _maybe_finish_wandb()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_tihm_sensor(load_yaml_config(args.config))


if __name__ == "__main__":
    main()

