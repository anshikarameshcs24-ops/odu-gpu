"""Stage 2 trainer for multimodal fusion fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from src.data.cmai_dataset import build_dataloaders
from src.models.cmai_system import CMAISystem
from src.training.losses import CombinedCMAILoss
from src.utils.config import load_yaml_config, set_seed

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency behavior
    wandb = None


def _maybe_init_wandb(config: dict) -> None:
    if config.get("use_wandb", False) and wandb is not None:
        wandb.init(project="cmai-detection", config=config, name=config.get("run_name", "stage2"))


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


def _build_model(config: dict, device: torch.device) -> torch.nn.Module:
    model = CMAISystem(
        vision_ckpt=config["vision_ckpt"],
        audio_ckpt=config["audio_ckpt"],
        enable_gradient_checkpointing=bool(config.get("enable_gradient_checkpointing", False)),
    ).to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = torch.nn.DataParallel(model)
    return model


def _compute_risk_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_df["agitation_risk"].value_counts().reindex(range(4), fill_value=1)
    weights = (1.0 / counts.values).astype(np.float32)
    weights = weights / weights.sum() * 4
    return torch.tensor(weights, device=device)


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_cmai_preds: list[np.ndarray] = []
    all_cmai_labels: list[np.ndarray] = []
    all_risk_preds: list[np.ndarray] = []
    all_risk_labels: list[np.ndarray] = []
    autocast_enabled = device.type == "cuda"

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        audio_values = batch["audio_values"].to(device, non_blocking=True)
        cmai_labels = batch["labels"].to(device)
        risk_labels = batch["risk_level"].to(device)

        with autocast(enabled=autocast_enabled):
            cmai_logits, risk_logits = model(pixel_values, audio_values)
            loss, _, _ = criterion(cmai_logits, risk_logits, cmai_labels, risk_labels)

        total_loss += loss.item()
        all_cmai_preds.append((torch.sigmoid(cmai_logits) > 0.5).cpu().numpy())
        all_cmai_labels.append(cmai_labels.cpu().numpy())
        all_risk_preds.append(risk_logits.argmax(dim=-1).cpu().numpy())
        all_risk_labels.append(risk_labels.cpu().numpy())

    y_cmai_pred = np.concatenate(all_cmai_preds) if all_cmai_preds else np.zeros((0, 29))
    y_cmai_true = np.concatenate(all_cmai_labels) if all_cmai_labels else np.zeros((0, 29))
    y_risk_pred = np.concatenate(all_risk_preds) if all_risk_preds else np.zeros((0,))
    y_risk_true = np.concatenate(all_risk_labels) if all_risk_labels else np.zeros((0,))

    return {
        "loss": total_loss / max(len(loader), 1),
        "cmai_f1_micro": f1_score(y_cmai_true, y_cmai_pred, average="micro", zero_division=0) if len(y_cmai_true) else 0.0,
        "cmai_f1_macro": f1_score(y_cmai_true, y_cmai_pred, average="macro", zero_division=0) if len(y_cmai_true) else 0.0,
        "risk_acc": float((y_risk_pred == y_risk_true).mean()) if len(y_risk_true) else 0.0,
    }


def train(config: dict) -> None:
    set_seed(int(config.get("seed", 42)))
    _maybe_init_wandb(config)

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

    model = _build_model(config, device)
    criterion = CombinedCMAILoss(risk_class_weights=_compute_risk_weights(train_df, device))

    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": float(config["backbone_lr"])},
            {"params": head_params, "lr": float(config["head_lr"])},
        ],
        weight_decay=float(config["weight_decay"]),
    )

    accumulation_steps = int(config.get("accumulation_steps", 1))
    total_steps = max((len(train_loader) * int(config["num_epochs"])) // accumulation_steps, 1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(int(total_steps * 0.05), 1), total_steps)
    scaler = GradScaler(enabled=device.type == "cuda")
    autocast_enabled = device.type == "cuda"

    output_dir = Path(config["output_dir"])
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    unfreeze_done = False

    for epoch in range(int(config["num_epochs"])):
        if epoch == int(config.get("unfreeze_epoch", 3)) and not unfreeze_done:
            base_model = model.module if hasattr(model, "module") else model
            base_model.vision_enc.unfreeze_all()
            unfreeze_done = True

        model.train()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            audio_values = batch["audio_values"].to(device, non_blocking=True)
            cmai_labels = batch["labels"].to(device)
            risk_labels = batch["risk_level"].to(device)

            with autocast(enabled=autocast_enabled):
                cmai_logits, risk_logits = model(pixel_values, audio_values)
                loss, cmai_loss, risk_loss = criterion(cmai_logits, risk_logits, cmai_labels, risk_labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if step % int(config.get("log_every", 10)) == 0:
                _maybe_log(
                    {
                        "train/loss": loss.item() * accumulation_steps,
                        "train/loss_cmai": cmai_loss.item(),
                        "train/loss_risk": risk_loss.item(),
                        "train/lr_backbone": optimizer.param_groups[0]["lr"],
                        "train/lr_head": optimizer.param_groups[1]["lr"],
                        "epoch": epoch,
                    }
                )

        metrics = evaluate(model, val_loader, criterion, device)
        _maybe_log({"epoch": epoch, **{f"val/{key}": value for key, value in metrics.items()}})
        if metrics["cmai_f1_micro"] > best_f1:
            best_f1 = metrics["cmai_f1_micro"]
            base_model = model.module if hasattr(model, "module") else model
            torch.save(base_model.state_dict(), best_dir / "model.pt")

    _maybe_finish_wandb()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_yaml_config(args.config))


if __name__ == "__main__":
    main()

