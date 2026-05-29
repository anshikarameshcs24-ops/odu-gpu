"""Binary elevated-risk TIHM dataset for early-warning detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.tihm_dataset import (
    _normalize_timestamp,
    build_tihm_feature_frame,
    build_tihm_risk_frame,
    compute_trajectory_label,
    load_tihm_tables,
)


def build_tihm_binary_frame(root_dir: str | Path, lead_minutes: int = 8) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    features = build_tihm_feature_frame(root_dir)
    labels = _normalize_timestamp(load_tihm_tables(root_dir)["labels"])
    agitation = labels[labels["type"].astype(str).str.lower() == "agitation"].copy()

    features["elevated"] = 0
    index_lookup = {(row.patient_id, row.minute): idx for idx, row in features[["patient_id", "minute"]].iterrows()}
    events: list[dict[str, Any]] = []

    for row in agitation.itertuples(index=False):
        patient_id = getattr(row, "patient_id")
        event_minute = getattr(row, "minute")
        lead_minutes_present = []

        for offset in range(1, lead_minutes + 1):
            minute = event_minute - pd.Timedelta(minutes=offset)
            idx = index_lookup.get((patient_id, minute))
            if idx is not None:
                features.at[idx, "elevated"] = 1
                lead_minutes_present.append(str(minute))

        events.append(
            {
                "patient_id": patient_id,
                "event_time": str(getattr(row, "date")),
                "event_minute": str(event_minute),
                "lead_minutes_present": sorted(lead_minutes_present),
            }
        )

    return features, events


def build_tihm_binary_sequence_records(
    root_dir: str | Path,
    output_dir: str | Path,
    sequence_length: int = 30,
    stride: int = 10,
    lead_minutes: int = 8,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    binary_frame, events = build_tihm_binary_frame(root_dir, lead_minutes=lead_minutes)
    trajectory_frame = build_tihm_risk_frame(root_dir)
    merged = binary_frame.merge(
        trajectory_frame[["patient_id", "minute", "agitation_risk"]],
        on=["patient_id", "minute"],
        how="left",
    )
    merged["agitation_risk"] = merged["agitation_risk"].fillna(0).astype(int)

    feature_columns = [
        col
        for col in merged.columns
        if col not in {"patient_id", "minute", "elevated", "agitation_risk", "agitation_event"}
    ]

    patient_records: list[dict[str, Any]] = []
    raw_counts = merged["elevated"].value_counts().reindex([0, 1], fill_value=0).astype(int).to_dict()

    for patient_id, patient_df in merged.groupby("patient_id"):
        patient_df = patient_df.sort_values("minute").reset_index(drop=True)
        features = patient_df[feature_columns].to_numpy(dtype=np.float32)
        elevated = patient_df["elevated"].astype(int).tolist()
        trajectory_risk = patient_df["agitation_risk"].astype(int).tolist()
        timestamps = patient_df["minute"].astype(str).tolist()

        if len(patient_df) < sequence_length:
            continue

        for start in range(0, len(patient_df) - sequence_length + 1, stride):
            end = start + sequence_length
            seq_features = features[start:end]
            seq_elevated = elevated[start:end]
            seq_risk = trajectory_risk[start:end]
            sequence_id = f"{patient_id}_{start:05d}_{end:05d}"
            feature_path = output_path / f"{sequence_id}.npy"
            np.save(feature_path, seq_features)
            patient_records.append(
                {
                    "sequence_id": sequence_id,
                    "patient_id": patient_id,
                    "feature_path": str(feature_path),
                    "elevated_labels": seq_elevated,
                    "trajectory": compute_trajectory_label(seq_risk),
                    "timestamps": timestamps[start:end],
                    "feature_dim": seq_features.shape[1],
                    "has_elevated": bool(max(seq_elevated) > 0),
                }
            )

    patient_records = sorted(patient_records, key=lambda item: (item["patient_id"], item["sequence_id"]))
    num_records = len(patient_records)
    train_end = int(num_records * train_ratio)
    val_end = train_end + int(num_records * val_ratio)

    timestep_counts = np.zeros(2, dtype=np.int64)
    window_counts = np.zeros(2, dtype=np.int64)
    for record in patient_records:
        labels = np.asarray(record["elevated_labels"], dtype=np.int64)
        timestep_counts += np.bincount(labels, minlength=2)
        window_counts[int(record["has_elevated"])] += 1

    return {
        "train": patient_records[:train_end],
        "val": patient_records[train_end:val_end],
        "test": patient_records[val_end:],
        "feature_columns": feature_columns,
        "events": events,
        "counts": {
            "raw_minute_counts": {"low": raw_counts[0], "elevated": raw_counts[1]},
            "window_timestep_counts": {"low": int(timestep_counts[0]), "elevated": int(timestep_counts[1])},
            "window_counts": {"low_only": int(window_counts[0]), "contains_elevated": int(window_counts[1])},
        },
    }


class TIHMBinarySequenceDataset(Dataset):
    def __init__(self, sequence_records: list[dict[str, Any]], cache_in_memory: bool = False):
        self.records = sequence_records
        self.cache_in_memory = cache_in_memory
        self._feature_cache: list[np.ndarray] | None = None
        if cache_in_memory:
            self._feature_cache = [
                np.load(record["feature_path"]).astype(np.float32)
                for record in self.records
            ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        if self._feature_cache is None:
            features = np.load(record["feature_path"]).astype(np.float32)
        else:
            features = self._feature_cache[idx]
        elevated = np.asarray(record["elevated_labels"], dtype=np.float32)
        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "elevated_labels": torch.tensor(elevated, dtype=torch.float32),
            "trajectory": torch.tensor(int(record["trajectory"]), dtype=torch.long),
            "length": torch.tensor(features.shape[0], dtype=torch.long),
        }


class TIHMBinaryPackedSequenceDataset(Dataset):
    def __init__(self, sequence_records: list[dict[str, Any]], pack_path: str | Path):
        self.records = sequence_records
        packed = np.load(pack_path)
        self.features = packed["features"]
        self.elevated = packed["elevated"]
        self.trajectory = packed["trajectory"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "elevated_labels": torch.tensor(self.elevated[idx], dtype=torch.float32),
            "trajectory": torch.tensor(int(self.trajectory[idx]), dtype=torch.long),
            "length": torch.tensor(self.features.shape[1], dtype=torch.long),
        }


def collate_tihm_binary_sequences(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    lengths = torch.stack([item["length"] for item in batch])
    max_len = int(lengths.max().item())
    feature_dim = int(batch[0]["features"].shape[-1])

    features = torch.zeros(len(batch), max_len, feature_dim, dtype=torch.float32)
    elevated = torch.zeros(len(batch), max_len, dtype=torch.float32)

    for idx, item in enumerate(batch):
        seq_len = int(item["length"].item())
        features[idx, :seq_len] = item["features"]
        elevated[idx, :seq_len] = item["elevated_labels"]

    return {
        "features": features,
        "elevated_labels": elevated,
        "trajectory": torch.stack([item["trajectory"] for item in batch]),
        "lengths": lengths,
    }


def save_binary_sequence_splits(records: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        with (output_path / f"{split}_sequences.json").open("w", encoding="utf-8") as handle:
            json.dump(records[split], handle, indent=2)
    for key in ("feature_columns", "events", "counts"):
        with (output_path / f"{key}.json").open("w", encoding="utf-8") as handle:
            json.dump(records[key], handle, indent=2)
