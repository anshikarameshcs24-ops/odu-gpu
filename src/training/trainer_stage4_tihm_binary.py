"""Train TIHM binary low-vs-elevated early-warning model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup

from src.data.tihm_binary_dataset import (
    TIHMBinaryPackedSequenceDataset,
    TIHMBinarySequenceDataset,
    collate_tihm_binary_sequences,
)
from src.models.tihm_binary_model import TIHMBinaryTemporalModel
from src.utils.config import load_yaml_config, set_seed

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _maybe_init_wandb(config: dict) -> None:
    if config.get("use_wandb", False) and wandb is not None:
        wandb.init(project="cmai-detection", config=config, name=config.get("run_name", "stage4_tihm_binary"))


def _maybe_log(payload: dict) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(payload)


def _maybe_finish_wandb() -> None:
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def _binary_counts(records: list[dict[str, Any]]) -> np.ndarray:
    counts = np.zeros(2, dtype=np.int64)
    for record in records:
        counts += np.bincount(np.asarray(record["elevated_labels"], dtype=np.int64), minlength=2)
    return counts


def _trajectory_counts(records: list[dict[str, Any]]) -> np.ndarray:
    return np.bincount(np.asarray([int(record["trajectory"]) for record in records], dtype=np.int64), minlength=3)


def _build_sampler(records: list[dict[str, Any]], minority_target_fraction: float) -> WeightedRandomSampler:
    positive_indices = [idx for idx, record in enumerate(records) if bool(record.get("has_elevated", False))]
    if not positive_indices:
        return WeightedRandomSampler(np.ones(len(records), dtype=np.float64), len(records), replacement=True)
    positives = len(positive_indices)
    negatives = max(len(records) - positives, 1)
    positive_weight = (minority_target_fraction / max(1.0 - minority_target_fraction, 1e-6)) * (negatives / positives)
    weights = np.ones(len(records), dtype=np.float64)
    weights[positive_indices] = max(positive_weight, 1.0)
    return WeightedRandomSampler(weights.tolist(), len(records), replacement=True)


def _pack_split(records: list[dict[str, Any]], pack_path: Path) -> None:
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    features = []
    elevated = []
    trajectory = []
    for index, record in enumerate(records):
        if index and index % 10000 == 0:
            print(f"packing {pack_path.name}: {index}/{len(records)}", flush=True)
        features.append(np.load(record["feature_path"]).astype(np.float32))
        elevated.append(np.asarray(record["elevated_labels"], dtype=np.float32))
        trajectory.append(int(record["trajectory"]))
    np.savez(
        pack_path,
        features=np.stack(features),
        elevated=np.stack(elevated),
        trajectory=np.asarray(trajectory, dtype=np.int64),
    )
    elapsed = time.perf_counter() - start_time
    size_mb = pack_path.stat().st_size / (1024 * 1024)
    print(f"[pack] Wrote {pack_path.name} ({size_mb:.1f} MB) in {elapsed:.1f}s", flush=True)


def _warn_if_pack_stale(pack_path: Path, records: list[dict[str, Any]]) -> None:
    source_dirs = {
        Path(record["feature_path"]).parent
        for record in records
        if record.get("feature_path")
    }
    newest_source_mtime = 0.0
    for source_dir in source_dirs:
        if source_dir.exists():
            newest_source_mtime = max(newest_source_mtime, source_dir.stat().st_mtime)

    if newest_source_mtime and pack_path.stat().st_mtime < newest_source_mtime:
        print(
            f"[pack] Warning: {pack_path.name} is older than source sequence directory; "
            "rebuild it if source arrays changed.",
            flush=True,
        )


def _make_dataset(records: list[dict[str, Any]], split_name: str, config: dict):
    if bool(config.get("pack_dataset_arrays", True)):
        pack_dir = Path(config.get("packed_arrays_dir", "data/processed/tihm_binary/packed"))
        pack_path = pack_dir / f"{split_name}.npz"
        if pack_path.exists() and pack_path.stat().st_size > 0:
            size_mb = pack_path.stat().st_size / (1024 * 1024)
            print(f"[pack] Found existing {pack_path.name} ({size_mb:.1f} MB), skipping repack.", flush=True)
            _warn_if_pack_stale(pack_path, records)
        else:
            _pack_split(records, pack_path)
        return TIHMBinaryPackedSequenceDataset(records, pack_path)
    return TIHMBinarySequenceDataset(records, cache_in_memory=bool(config.get("cache_dataset_in_memory", False)))


def _packed_cache_ready(config: dict) -> bool:
    if not bool(config.get("pack_dataset_arrays", True)):
        return False
    pack_dir = Path(config.get("packed_arrays_dir", "data/processed/tihm_binary/packed"))
    return all(
        (pack_dir / f"{split}.npz").exists() and (pack_dir / f"{split}.npz").stat().st_size > 0
        for split in ("train", "val", "test")
    )


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_score: list[np.ndarray] = []
    traj_true: list[np.ndarray] = []
    traj_pred: list[np.ndarray] = []
    timestep_meta: list[dict[str, Any]] = []

    record_offset = 0
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["elevated_labels"].to(device)
        lengths = batch["lengths"].to(device)
        trajectory = batch["trajectory"].to(device)
        risk_logits, traj_logits = model(features, lengths)
        risk_logits = risk_logits.squeeze(-1)
        mask = torch.arange(features.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)

        y_true.append(labels[mask].cpu().numpy())
        y_score.append(torch.sigmoid(risk_logits)[mask].cpu().numpy())
        traj_true.append(trajectory.cpu().numpy())
        traj_pred.append(traj_logits.argmax(dim=-1).cpu().numpy())

        for batch_index in range(features.size(0)):
            seq_len = int(lengths[batch_index].item())
            dataset_record = loader.dataset.records[record_offset + batch_index]
            scores = torch.sigmoid(risk_logits[batch_index, :seq_len]).cpu().numpy()
            labels_np = labels[batch_index, :seq_len].cpu().numpy()
            for timestep in range(seq_len):
                timestep_meta.append(
                    {
                        "patient_id": dataset_record["patient_id"],
                        "timestamp": dataset_record["timestamps"][timestep],
                        "score": float(scores[timestep]),
                        "label": int(labels_np[timestep]),
                    }
                )
        record_offset += features.size(0)

    return {
        "y_true": np.concatenate(y_true) if y_true else np.zeros((0,), dtype=int),
        "y_score": np.concatenate(y_score) if y_score else np.zeros((0,), dtype=float),
        "trajectory_true": np.concatenate(traj_true) if traj_true else np.zeros((0,), dtype=int),
        "trajectory_pred": np.concatenate(traj_pred) if traj_pred else np.zeros((0,), dtype=int),
        "timestep_meta": timestep_meta,
    }


def _threshold_for_min_sensitivity(y_true: np.ndarray, y_score: np.ndarray, min_sensitivity: float = 0.5) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    candidates = []
    for idx, threshold in enumerate(thresholds):
        sensitivity = recall[idx]
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / max(tn + fp, 1)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if sensitivity >= min_sensitivity:
            candidates.append((f1, specificity, threshold))
    if not candidates:
        return 0.5
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return float(candidates[0][2])


def _binary_summary(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    auroc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    auprc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "threshold": float(threshold),
        "auprc": float(auprc),
        "auroc": float(auroc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "true_counts": np.bincount(y_true.astype(int), minlength=2).tolist(),
        "pred_counts": np.bincount(y_pred.astype(int), minlength=2).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["low", "elevated"],
            zero_division=0,
            output_dict=True,
        ),
    }


def _event_recall(events: list[dict[str, Any]], timestep_meta: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    lookup: dict[tuple[str, str], float] = {}
    for row in timestep_meta:
        key = (row["patient_id"], row["timestamp"])
        lookup[key] = max(float(row["score"]), lookup.get(key, 0.0))

    covered = 0
    evaluable = 0
    examples = []
    for event in events:
        lead_minutes = event.get("lead_minutes_present", [])
        present_minutes = [
            minute
            for minute in lead_minutes
            if (event["patient_id"], minute) in lookup
        ]
        if not present_minutes:
            continue
        evaluable += 1
        max_score = max((lookup.get((event["patient_id"], minute), 0.0) for minute in present_minutes), default=0.0)
        detected = max_score >= threshold
        covered += int(detected)
        if len(examples) < 10:
            examples.append(
                {
                    "patient_id": event["patient_id"],
                    "event_minute": event["event_minute"],
                    "test_lead_minutes_present": present_minutes,
                    "max_lead_score": float(max_score),
                    "detected": bool(detected),
                }
            )
    return {
        "event_recall": covered / max(evaluable, 1),
        "detected_events": covered,
        "evaluable_events": evaluable,
        "examples": examples,
    }


def train_stage4(config: dict) -> None:
    set_seed(int(config.get("seed", 42)))
    _maybe_init_wandb(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records = _load_records(config["train_sequences"])
    val_records = _load_records(config["val_sequences"])
    test_records = _load_records(config["test_sequences"])
    events = _load_records(config["events_json"])
    print(
        f"[data] train={len(train_records)} windows | val={len(val_records)} | "
        f"test={len(test_records)} | packed={'yes' if _packed_cache_ready(config) else 'no'} | "
        f"workers={int(config['num_workers'])}",
        flush=True,
    )

    feature_dim = int(train_records[0]["feature_dim"])
    binary_counts = _binary_counts(train_records)
    trajectory_counts = _trajectory_counts(train_records)
    pos_weight = torch.tensor([binary_counts[0] / max(binary_counts[1], 1)], dtype=torch.float32, device=device)

    print(f"train/binary_counts={{'low': {int(binary_counts[0])}, 'elevated': {int(binary_counts[1])}}}")
    print(f"train/pos_weight={float(pos_weight.item()):.4f}")
    print(f"train/trajectory_counts={trajectory_counts.tolist()}")

    train_loader = DataLoader(
        _make_dataset(train_records, "train", config),
        batch_size=int(config["batch_size"]),
        sampler=_build_sampler(train_records, float(config.get("minority_target_fraction", 0.3))),
        num_workers=int(config["num_workers"]),
        collate_fn=collate_tihm_binary_sequences,
    )
    val_loader = DataLoader(
        _make_dataset(val_records, "val", config),
        batch_size=int(config.get("eval_batch_size", config["batch_size"])),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_tihm_binary_sequences,
    )
    test_loader = DataLoader(
        _make_dataset(test_records, "test", config),
        batch_size=int(config.get("eval_batch_size", config["batch_size"])),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=collate_tihm_binary_sequences,
    )

    model = TIHMBinaryTemporalModel(
        feature_dim=feature_dim,
        projection_dim=int(config.get("projection_dim", 256)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.2)),
    ).to(device)
    elevated_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    trajectory_loss = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    total_steps = max(len(train_loader) * int(config["num_epochs"]), 1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(int(total_steps * 0.05), 1), total_steps)
    scaler = GradScaler(enabled=device.type == "cuda")
    autocast_enabled = device.type == "cuda"

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_auprc = -1.0
    stale_epochs = 0

    for epoch in range(int(config["num_epochs"])):
        model.train()
        for step, batch in enumerate(train_loader):
            features = batch["features"].to(device)
            elevated = batch["elevated_labels"].to(device)
            trajectory = batch["trajectory"].to(device)
            lengths = batch["lengths"].to(device)
            mask = torch.arange(features.size(1), device=device).unsqueeze(0) < lengths.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=autocast_enabled):
                risk_logits, trajectory_logits = model(features, lengths)
                risk_logits = risk_logits.squeeze(-1)
                loss_elevated = elevated_loss(risk_logits[mask], elevated[mask])
                loss_trajectory = trajectory_loss(trajectory_logits, trajectory)
                loss = loss_elevated + 0.3 * loss_trajectory

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
                        "train/loss_elevated": loss_elevated.item(),
                        "train/loss_trajectory": loss_trajectory.item(),
                        "epoch": epoch,
                    }
                )

        val_predictions = collect_predictions(model, val_loader, device)
        val_auprc = average_precision_score(val_predictions["y_true"], val_predictions["y_score"])
        val_threshold = _threshold_for_min_sensitivity(val_predictions["y_true"], val_predictions["y_score"])
        val_summary = _binary_summary(val_predictions["y_true"], val_predictions["y_score"], val_threshold)
        print(
            f"epoch={epoch:02d} val/auprc={val_auprc:.6f} "
            f"threshold={val_threshold:.4f} pred_counts={val_summary['pred_counts']}",
            flush=True,
        )
        _maybe_log({"val/auprc": val_auprc, "val/threshold": val_threshold, "epoch": epoch})

        torch.save(model.state_dict(), output_dir / f"epoch_{epoch:02d}.pt")
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            stale_epochs = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.get("early_stop_patience", 4)):
                print(f"early_stop=val_auprc_patience epoch={epoch}", flush=True)
                break

    best_path = output_dir / "best_model.pt"
    model.load_state_dict(torch.load(best_path, map_location=device))
    val_predictions = collect_predictions(model, val_loader, device)
    threshold = _threshold_for_min_sensitivity(val_predictions["y_true"], val_predictions["y_score"])
    test_predictions = collect_predictions(model, test_loader, device)
    test_summary = _binary_summary(test_predictions["y_true"], test_predictions["y_score"], threshold)
    final_results = {
        "checkpoint": str(best_path),
        "threshold_source": "validation threshold maximizing F1 among thresholds with sensitivity >= 0.50",
        "operating_threshold": threshold,
        "validation": _binary_summary(val_predictions["y_true"], val_predictions["y_score"], threshold),
        "test": test_summary,
        "trajectory_test": {
            "true_counts": np.bincount(test_predictions["trajectory_true"], minlength=3).tolist(),
            "pred_counts": np.bincount(test_predictions["trajectory_pred"], minlength=3).tolist(),
            "macro_f1": f1_score(
                test_predictions["trajectory_true"],
                test_predictions["trajectory_pred"],
                average="macro",
                zero_division=0,
            ),
        },
        "event_level": _event_recall(events, test_predictions["timestep_meta"], threshold),
    }
    (output_dir / "eval_results.json").write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    print(json.dumps(final_results, indent=2), flush=True)
    _maybe_finish_wandb()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_stage4(load_yaml_config(args.config))


if __name__ == "__main__":
    main()
