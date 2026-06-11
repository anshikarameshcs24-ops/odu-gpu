"""Stage 1 trainer for modality-specific adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from src.data.cmai_dataset import build_dataloaders
from src.models.audio_encoder import CMAIAudioEncoder
from src.models.vision_encoder import CMAIVisionEncoder
from src.training.losses import CombinedCMAILoss
from src.utils.config import load_yaml_config, set_seed
from src.utils.cmai_labels import NUM_CMAI_CLASSES

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency behavior
    wandb = None


class Stage1SingleModalityModel(nn.Module):
    def __init__(self, modality: str, vision_ckpt: str, audio_ckpt: str):
        super().__init__()
        self.modality = modality
        if modality == "vision":
            self.encoder = CMAIVisionEncoder(pretrained=vision_ckpt, freeze_layers=0)
        elif modality == "audio":
            self.encoder = CMAIAudioEncoder(pretrained=audio_ckpt)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        self.cmai_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CMAI_CLASSES),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
        )

    def forward(self, pixel_values: torch.Tensor, audio_values: torch.Tensor):
        if self.modality == "vision":
            embedding, _ = self.encoder(pixel_values)
        else:
            embedding, _ = self.encoder(audio_values)
        return self.cmai_head(embedding), self.risk_head(embedding)


def _maybe_init_wandb(config: dict, run_name: str) -> None:
    if config.get("use_wandb", False) and wandb is not None:
        wandb.init(project="cmai-detection", config=config, name=run_name)


def _maybe_log(payload: dict) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(payload)


def _maybe_finish_wandb() -> None:
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def _load_split_data(config: dict):
    train_df = pd.read_csv(config["train_csv"])
    val_df = pd.read_csv(config["val_csv"])
    manifest_df = pd.read_csv(config["manifest_csv"])
    merge_cols = ["chunk_id", "frame_path", "audio_path"]
    train_df = train_df.merge(manifest_df[merge_cols], on="chunk_id", how="inner")
    val_df = val_df.merge(manifest_df[merge_cols], on="chunk_id", how="inner")
    return train_df, val_df


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        audio_values = batch["audio_values"].to(device)
        labels = batch["labels"].to(device)
        risk = batch["risk_level"].to(device)

        cmai_logits, risk_logits = model(pixel_values, audio_values)
        loss, _, _ = criterion(cmai_logits, risk_logits, labels, risk)
        total_loss += loss.item()

        all_preds.append((torch.sigmoid(cmai_logits) > 0.5).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds) if all_preds else np.zeros((0, NUM_CMAI_CLASSES))
    y_true = np.concatenate(all_labels) if all_labels else np.zeros((0, NUM_CMAI_CLASSES))

    return {
        "loss": total_loss / max(len(loader), 1),
        "cmai_f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0) if len(y_true) else 0.0,
    }


def train_stage1(config: dict) -> None:
    set_seed(int(config.get("seed", 42)))
    _maybe_init_wandb(config, config.get("run_name", "stage1"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df = _load_split_data(config)
    train_loader, val_loader = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        vision_ckpt=config["vision_ckpt"],
        audio_ckpt=config["audio_ckpt"],
    )

    model = Stage1SingleModalityModel(
        modality=config["modality"],
        vision_ckpt=config["vision_ckpt"],
        audio_ckpt=config["audio_ckpt"],
    ).to(device)

    risk_counts = train_df["agitation_risk"].value_counts().reindex(range(4), fill_value=1)
    risk_weights = torch.tensor((1.0 / risk_counts.values).astype(np.float32), device=device)
    criterion = CombinedCMAILoss(risk_class_weights=risk_weights / risk_weights.sum() * 4)
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    total_steps = len(train_loader) * int(config["num_epochs"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(int(total_steps * 0.05), 1), max(total_steps, 1))

    best_f1 = -1.0
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(config["num_epochs"])):
        model.train()
        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            audio_values = batch["audio_values"].to(device)
            labels = batch["labels"].to(device)
            risk = batch["risk_level"].to(device)

            optimizer.zero_grad(set_to_none=True)
            cmai_logits, risk_logits = model(pixel_values, audio_values)
            loss, cmai_loss, risk_loss = criterion(cmai_logits, risk_logits, labels, risk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
            optimizer.step()
            scheduler.step()

            if step % int(config.get("log_every", 10)) == 0:
                _maybe_log(
                    {
                        "train/loss": loss.item(),
                        "train/loss_cmai": cmai_loss.item(),
                        "train/loss_risk": risk_loss.item(),
                        "epoch": epoch,
                    }
                )

        metrics = evaluate(model, val_loader, criterion, device)
        _maybe_log({f"val/{key}": value for key, value in metrics.items()})
        if metrics["cmai_f1_micro"] > best_f1:
            best_f1 = metrics["cmai_f1_micro"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    _maybe_finish_wandb()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_stage1(load_yaml_config(args.config))


if __name__ == "__main__":
    main()
