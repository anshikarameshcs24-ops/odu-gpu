"""Train a TIHM sensor-only temporal risk model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
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


class FocalCrossEntropyLoss(torch.nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        prob = log_prob.exp()
        gather_index = targets.unsqueeze(-1)
        target_log_prob = log_prob.gather(-1, gather_index).squeeze(-1)
        target_prob = prob.gather(-1, gather_index).squeeze(-1)
        target_alpha = self.alpha.gather(0, targets)
        loss = -target_alpha * (1.0 - target_prob).pow(self.gamma) * target_log_prob
        return loss.mean()


def _risk_counts(records: list[dict]) -> np.ndarray:
    counts = np.zeros(4, dtype=np.int64)
    for record in records:
        counts += np.bincount(np.asarray(record["risk_levels"], dtype=np.int64), minlength=4)
    return counts


def _trajectory_counts(records: list[dict]) -> np.ndarray:
    values = [int(record["trajectory"]) for record in records]
    return np.bincount(np.asarray(values, dtype=np.int64), minlength=3)


def _inverse_frequency_weights(
    counts: np.ndarray,
    device: torch.device,
    max_weight: float = 200.0,
    normalize: bool = False,
) -> torch.Tensor:
    safe_counts = np.maximum(counts.astype(np.float64), 1.0)
    weights = safe_counts.max() / safe_counts
    if normalize:
        weights = weights / weights.mean()
    weights = np.clip(weights, 1.0, max_weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _build_sampler(records: list[dict], minority_target_fraction: float = 0.25) -> WeightedRandomSampler:
    minority_indices = [
        index
        for index, record in enumerate(records)
        if max(record["risk_levels"]) > 0 or int(record["trajectory"]) > 0
    ]
    if not minority_indices:
        return WeightedRandomSampler(np.ones(len(records), dtype=np.float64), len(records), replacement=True)

    minority = len(minority_indices)
    majority = max(len(records) - minority, 1)
    minority_weight = (minority_target_fraction / max(1.0 - minority_target_fraction, 1e-6)) * (majority / minority)
    weights = np.ones(len(records), dtype=np.float64)
    weights[minority_indices] = max(minority_weight, 1.0)
    return WeightedRandomSampler(weights.tolist(), num_samples=len(records), replacement=True)


def _predict_with_thresholds(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    adjusted = probabilities - thresholds.reshape(1, -1)
    active = adjusted >= 0
    fallback = probabilities.argmax(axis=1)
    thresholded = adjusted.argmax(axis=1)
    return np.where(active.any(axis=1), thresholded, fallback)


def _tune_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.full(probabilities.shape[1], 0.5, dtype=np.float32)
    candidates = np.arange(0.05, 0.95, 0.05)
    for class_index in range(probabilities.shape[1]):
        binary_true = (y_true == class_index).astype(int)
        if binary_true.sum() == 0:
            continue
        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in candidates:
            binary_pred = (probabilities[:, class_index] >= threshold).astype(int)
            score = f1_score(binary_true, binary_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold
        thresholds[class_index] = best_threshold
    return thresholds


def _per_class_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], names: list[str]) -> dict:
    metrics = {}
    for label, name in zip(labels, names):
        true_positive = int(((y_true == label) & (y_pred == label)).sum())
        false_positive = int(((y_true != label) & (y_pred == label)).sum())
        true_negative = int(((y_true != label) & (y_pred != label)).sum())
        false_negative = int(((y_true == label) & (y_pred != label)).sum())
        sensitivity = true_positive / max(true_positive + false_negative, 1)
        specificity = true_negative / max(true_negative + false_positive, 1)
        metrics[name] = {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "tp": true_positive,
            "fp": false_positive,
            "tn": true_negative,
            "fn": false_negative,
        }
    return metrics


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    all_risk_true: list[np.ndarray] = []
    all_risk_prob: list[np.ndarray] = []
    all_traj_true: list[np.ndarray] = []
    all_traj_prob: list[np.ndarray] = []

    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        risk_levels = batch["risk_levels"].to(device)
        trajectory = batch["trajectory"].to(device)
        risk_logits, traj_logits = model(features, lengths)
        mask = torch.arange(features.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)
        all_risk_prob.append(torch.softmax(risk_logits, dim=-1)[mask].cpu().numpy())
        all_risk_true.append(risk_levels[mask].cpu().numpy())
        all_traj_prob.append(torch.softmax(traj_logits, dim=-1).cpu().numpy())
        all_traj_true.append(trajectory.cpu().numpy())

    return {
        "risk_true": np.concatenate(all_risk_true) if all_risk_true else np.zeros((0,), dtype=int),
        "risk_prob": np.concatenate(all_risk_prob) if all_risk_prob else np.zeros((0, 4), dtype=float),
        "trajectory_true": np.concatenate(all_traj_true) if all_traj_true else np.zeros((0,), dtype=int),
        "trajectory_prob": np.concatenate(all_traj_prob) if all_traj_prob else np.zeros((0, 3), dtype=float),
    }


def summarize_predictions(
    predictions: dict[str, np.ndarray],
    risk_thresholds: np.ndarray | None = None,
    trajectory_thresholds: np.ndarray | None = None,
) -> dict:
    y_risk_true = predictions["risk_true"]
    y_traj_true = predictions["trajectory_true"]
    risk_prob = predictions["risk_prob"]
    traj_prob = predictions["trajectory_prob"]
    y_risk_pred = (
        _predict_with_thresholds(risk_prob, risk_thresholds)
        if risk_thresholds is not None
        else risk_prob.argmax(axis=1)
    )
    y_traj_pred = (
        _predict_with_thresholds(traj_prob, trajectory_thresholds)
        if trajectory_thresholds is not None
        else traj_prob.argmax(axis=1)
    )
    risk_names = ["low", "moderate", "high", "imminent"]
    trajectory_names = ["stable", "rising", "peaking"]

    return {
        "risk_acc": float((y_risk_pred == y_risk_true).mean()) if len(y_risk_true) else 0.0,
        "risk_macro_f1": f1_score(y_risk_true, y_risk_pred, average="macro", zero_division=0) if len(y_risk_true) else 0.0,
        "risk_weighted_f1": f1_score(y_risk_true, y_risk_pred, average="weighted", zero_division=0) if len(y_risk_true) else 0.0,
        "trajectory_acc": float((y_traj_pred == y_traj_true).mean()) if len(y_traj_true) else 0.0,
        "trajectory_macro_f1": f1_score(y_traj_true, y_traj_pred, average="macro", zero_division=0) if len(y_traj_true) else 0.0,
        "risk_true_counts": np.bincount(y_risk_true, minlength=4).tolist(),
        "risk_pred_counts": np.bincount(y_risk_pred, minlength=4).tolist(),
        "trajectory_true_counts": np.bincount(y_traj_true, minlength=3).tolist(),
        "trajectory_pred_counts": np.bincount(y_traj_pred, minlength=3).tolist(),
        "risk_confusion": confusion_matrix(y_risk_true, y_risk_pred, labels=[0, 1, 2, 3]).tolist(),
        "trajectory_confusion": confusion_matrix(y_traj_true, y_traj_pred, labels=[0, 1, 2]).tolist(),
        "risk_report": classification_report(
            y_risk_true,
            y_risk_pred,
            labels=[0, 1, 2, 3],
            target_names=risk_names,
            zero_division=0,
            output_dict=True,
        ),
        "trajectory_report": classification_report(
            y_traj_true,
            y_traj_pred,
            labels=[0, 1, 2],
            target_names=trajectory_names,
            zero_division=0,
            output_dict=True,
        ),
        "risk_sensitivity_specificity": _per_class_binary_metrics(
            y_risk_true,
            y_risk_pred,
            labels=[0, 1, 2, 3],
            names=risk_names,
        ),
        "trajectory_sensitivity_specificity": _per_class_binary_metrics(
            y_traj_true,
            y_traj_pred,
            labels=[0, 1, 2],
            names=trajectory_names,
        ),
    }


def train_tihm_sensor(config: dict) -> None:
    set_seed(int(config.get("seed", 42)))
    _maybe_init_wandb(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records = _load_records(config["train_sequences"])
    val_records = _load_records(config["val_sequences"])
    test_records = _load_records(config["test_sequences"]) if config.get("test_sequences") else []

    if not train_records:
        raise ValueError("No TIHM training sequences were found.")

    feature_dim = int(train_records[0]["feature_dim"])

    risk_counts = _risk_counts(train_records)
    trajectory_counts = _trajectory_counts(train_records)
    risk_weights = _inverse_frequency_weights(
        risk_counts,
        device,
        max_weight=float(config.get("max_class_weight", 50.0)),
        normalize=bool(config.get("normalize_class_weights", False)),
    )
    trajectory_weights = _inverse_frequency_weights(
        trajectory_counts,
        device,
        max_weight=float(config.get("max_class_weight", 50.0)),
        normalize=bool(config.get("normalize_class_weights", False)),
    )
    print(f"train/risk_counts={risk_counts.tolist()} weights={risk_weights.detach().cpu().tolist()}")
    print(f"train/trajectory_counts={trajectory_counts.tolist()} weights={trajectory_weights.detach().cpu().tolist()}")

    train_loader = DataLoader(
        TIHMSequenceDataset(train_records),
        batch_size=int(config["batch_size"]),
        sampler=_build_sampler(
            train_records,
            minority_target_fraction=float(config.get("minority_target_fraction", 0.25)),
        ),
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
    test_loader = DataLoader(
        TIHMSequenceDataset(test_records),
        batch_size=int(config.get("eval_batch_size", config["batch_size"])),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_tihm_sequences,
    ) if test_records else None

    model = TIHMSensorTemporalModel(
        feature_dim=feature_dim,
        projection_dim=int(config.get("projection_dim", 256)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)

    gamma = float(config.get("focal_gamma", 2.0))
    risk_criterion = FocalCrossEntropyLoss(risk_weights, gamma=gamma)
    traj_criterion = FocalCrossEntropyLoss(trajectory_weights, gamma=gamma)
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

        val_predictions = collect_predictions(model, val_loader, device)
        metrics = summarize_predictions(val_predictions)
        print(
            "epoch={epoch:02d} val/risk_macro_f1={risk_macro_f1:.4f} "
            "val/risk_pred_counts={risk_pred_counts} val/traj_macro_f1={trajectory_macro_f1:.4f} "
            "val/traj_pred_counts={trajectory_pred_counts}".format(epoch=epoch, **metrics),
            flush=True,
        )
        _maybe_log({f"val/{key}": value for key, value in metrics.items()})
        torch.save(model.state_dict(), output_dir / f"epoch_{epoch:02d}.pt")
        if metrics["risk_macro_f1"] > best_score:
            best_score = metrics["risk_macro_f1"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        risk_collapse = max(metrics["risk_pred_counts"]) / max(sum(metrics["risk_pred_counts"]), 1)
        trajectory_collapse = max(metrics["trajectory_pred_counts"]) / max(sum(metrics["trajectory_pred_counts"]), 1)
        if epoch >= int(config.get("collapse_check_epoch", 4)) and (
            risk_collapse > float(config.get("collapse_threshold", 0.95))
            or trajectory_collapse > float(config.get("collapse_threshold", 0.95))
        ):
            print(
                f"early_stop=prediction_collapse epoch={epoch} "
                f"risk_max_fraction={risk_collapse:.4f} trajectory_max_fraction={trajectory_collapse:.4f}",
                flush=True,
            )
            break

    best_path = output_dir / "best_model.pt"
    model.load_state_dict(torch.load(best_path, map_location=device))
    val_predictions = collect_predictions(model, val_loader, device)
    risk_thresholds = _tune_thresholds(val_predictions["risk_true"], val_predictions["risk_prob"])
    trajectory_thresholds = _tune_thresholds(
        val_predictions["trajectory_true"],
        val_predictions["trajectory_prob"],
    )
    final_results = {
        "checkpoint": str(best_path),
        "risk_thresholds": risk_thresholds.tolist(),
        "trajectory_thresholds": trajectory_thresholds.tolist(),
        "validation": summarize_predictions(val_predictions, risk_thresholds, trajectory_thresholds),
    }
    if test_loader is not None:
        test_predictions = collect_predictions(model, test_loader, device)
        final_results["test"] = summarize_predictions(test_predictions, risk_thresholds, trajectory_thresholds)
    (output_dir / "eval_results.json").write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    print(json.dumps(final_results, indent=2), flush=True)

    _maybe_finish_wandb()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_tihm_sensor(load_yaml_config(args.config))


if __name__ == "__main__":
    main()
