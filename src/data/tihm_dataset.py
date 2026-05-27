"""Dataset builders for TIHM wearable/sensor temporal risk modeling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


EXPECTED_TABLES = {
    "activity": "Activity.csv",
    "labels": "Labels.csv",
    "physiology": "Physiology.csv",
    "sleep": "Sleep.csv",
    "demographics": "Demographics.csv",
}


def _resolve_table(root_dir: str | Path, filename: str) -> Path:
    root = Path(root_dir)
    direct = root / filename
    if direct.exists():
        return direct
    lower_name = filename.lower()
    for candidate in root.rglob("*.csv"):
        if candidate.name.lower() == lower_name:
            return candidate
    raise FileNotFoundError(f"Could not locate {filename} under {root}")


def load_tihm_tables(root_dir: str | Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in EXPECTED_TABLES.items():
        path = _resolve_table(root_dir, filename)
        tables[key] = pd.read_csv(path)
    return tables


def _normalize_timestamp(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    out = out.dropna(subset=[column])
    out["minute"] = out[column].dt.floor("min")
    return out


def build_tihm_feature_frame(root_dir: str | Path) -> pd.DataFrame:
    tables = load_tihm_tables(root_dir)

    activity = _normalize_timestamp(tables["activity"])
    activity["activity_count"] = 1.0
    activity_pivot = (
        activity.pivot_table(
            index=["patient_id", "minute"],
            columns="location_name",
            values="activity_count",
            aggfunc="sum",
            fill_value=0.0,
        )
        .add_prefix("activity_location_")
        .reset_index()
    )

    physiology = _normalize_timestamp(tables["physiology"])
    physiology_pivot = (
        physiology.pivot_table(
            index=["patient_id", "minute"],
            columns="device_type",
            values="value",
            aggfunc="mean",
        )
        .add_prefix("physiology_")
        .reset_index()
    )

    sleep = _normalize_timestamp(tables["sleep"])
    sleep_numeric = (
        sleep.groupby(["patient_id", "minute"], as_index=False)[["heart_rate", "respiratory_rate", "snoring"]]
        .mean(numeric_only=True)
        .rename(
            columns={
                "heart_rate": "sleep_heart_rate",
                "respiratory_rate": "sleep_respiratory_rate",
                "snoring": "sleep_snoring",
            }
        )
    )
    sleep_state = (
        sleep.pivot_table(
            index=["patient_id", "minute"],
            columns="state",
            values="snoring",
            aggfunc="size",
            fill_value=0.0,
        )
        .add_prefix("sleep_state_")
        .reset_index()
    )

    features = activity_pivot.merge(physiology_pivot, on=["patient_id", "minute"], how="outer")
    features = features.merge(sleep_numeric, on=["patient_id", "minute"], how="outer")
    features = features.merge(sleep_state, on=["patient_id", "minute"], how="outer")
    features = features.sort_values(["patient_id", "minute"]).reset_index(drop=True)

    demographics = tables["demographics"].copy()
    demographics.columns = [str(col) for col in demographics.columns]
    demographics = demographics.rename(columns={"sex": "demographics_sex", "age": "demographics_age"})
    features = features.merge(
        demographics[["patient_id", "demographics_sex", "demographics_age"]],
        on="patient_id",
        how="left",
    )

    features["demographics_sex"] = features["demographics_sex"].astype("category")
    features["demographics_age"] = features["demographics_age"].astype("category")
    features = pd.get_dummies(
        features,
        columns=["demographics_sex", "demographics_age"],
        dummy_na=True,
    )
    numeric_cols = [col for col in features.columns if col not in {"patient_id", "minute"}]
    features[numeric_cols] = features[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return features


def build_tihm_risk_frame(
    root_dir: str | Path,
    pre_agitation_minutes: int = 8,
    imminent_minutes: int = 3,
) -> pd.DataFrame:
    features = build_tihm_feature_frame(root_dir)
    labels = _normalize_timestamp(load_tihm_tables(root_dir)["labels"])
    agitation = labels[labels["type"].astype(str).str.lower() == "agitation"].copy()

    features["agitation_risk"] = 0
    features["agitation_event"] = 0

    if agitation.empty:
        return features

    index_lookup = {(row.patient_id, row.minute): idx for idx, row in features[["patient_id", "minute"]].iterrows()}

    for row in agitation.itertuples(index=False):
        patient_id = getattr(row, "patient_id")
        minute = getattr(row, "minute")
        event_idx = index_lookup.get((patient_id, minute))
        if event_idx is not None:
            features.at[event_idx, "agitation_risk"] = 3
            features.at[event_idx, "agitation_event"] = 1

        for offset in range(1, imminent_minutes + 1):
            prior_idx = index_lookup.get((patient_id, minute - pd.Timedelta(minutes=offset)))
            if prior_idx is not None:
                features.at[prior_idx, "agitation_risk"] = max(int(features.at[prior_idx, "agitation_risk"]), 2)

        for offset in range(imminent_minutes + 1, pre_agitation_minutes + 1):
            prior_idx = index_lookup.get((patient_id, minute - pd.Timedelta(minutes=offset)))
            if prior_idx is not None:
                features.at[prior_idx, "agitation_risk"] = max(int(features.at[prior_idx, "agitation_risk"]), 1)

    return features


def compute_trajectory_label(risk_levels: list[int]) -> int:
    if not risk_levels:
        return 0
    if max(risk_levels) >= 3:
        return 2
    if risk_levels[-1] > risk_levels[0] or any(b > a for a, b in zip(risk_levels[:-1], risk_levels[1:])):
        return 1
    return 0


def build_tihm_sequence_records(
    root_dir: str | Path,
    output_dir: str | Path,
    sequence_length: int = 30,
    stride: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame = build_tihm_risk_frame(root_dir)
    feature_columns = [col for col in frame.columns if col not in {"patient_id", "minute", "agitation_risk", "agitation_event"}]

    patient_records: list[dict[str, Any]] = []
    for patient_id, patient_df in frame.groupby("patient_id"):
        patient_df = patient_df.sort_values("minute").reset_index(drop=True)
        features = patient_df[feature_columns].to_numpy(dtype=np.float32)
        risk_levels = patient_df["agitation_risk"].astype(int).tolist()
        timestamps = patient_df["minute"].astype(str).tolist()

        if len(patient_df) < sequence_length:
            continue

        for start in range(0, len(patient_df) - sequence_length + 1, stride):
            end = start + sequence_length
            seq_features = features[start:end]
            seq_risk = risk_levels[start:end]
            seq_timestamps = timestamps[start:end]
            sequence_id = f"{patient_id}_{start:05d}_{end:05d}"
            feature_path = output_path / f"{sequence_id}.npy"
            np.save(feature_path, seq_features)
            patient_records.append(
                {
                    "sequence_id": sequence_id,
                    "patient_id": patient_id,
                    "feature_path": str(feature_path),
                    "risk_levels": seq_risk,
                    "trajectory": compute_trajectory_label(seq_risk),
                    "timestamps": seq_timestamps,
                    "feature_dim": seq_features.shape[1],
                }
            )

    patient_records = sorted(patient_records, key=lambda item: (item["patient_id"], item["sequence_id"]))
    num_records = len(patient_records)
    train_end = int(num_records * train_ratio)
    val_end = train_end + int(num_records * val_ratio)

    return {
        "train": patient_records[:train_end],
        "val": patient_records[train_end:val_end],
        "test": patient_records[val_end:],
        "feature_columns": feature_columns,
    }


class TIHMSequenceDataset(Dataset):
    def __init__(self, sequence_records: list[dict[str, Any]]):
        self.records = sequence_records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        features = np.load(record["feature_path"]).astype(np.float32)
        risk_levels = np.asarray(record["risk_levels"], dtype=np.int64)
        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "risk_levels": torch.tensor(risk_levels, dtype=torch.long),
            "trajectory": torch.tensor(int(record["trajectory"]), dtype=torch.long),
            "length": torch.tensor(features.shape[0], dtype=torch.long),
        }


def collate_tihm_sequences(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    lengths = torch.stack([item["length"] for item in batch])
    max_len = int(lengths.max().item())
    feature_dim = int(batch[0]["features"].shape[-1])

    features = torch.zeros(len(batch), max_len, feature_dim, dtype=torch.float32)
    risk_levels = torch.zeros(len(batch), max_len, dtype=torch.long)

    for idx, item in enumerate(batch):
        seq_len = int(item["length"].item())
        features[idx, :seq_len] = item["features"]
        risk_levels[idx, :seq_len] = item["risk_levels"]

    return {
        "features": features,
        "risk_levels": risk_levels,
        "trajectory": torch.stack([item["trajectory"] for item in batch]),
        "lengths": lengths,
    }


def save_sequence_splits(records: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        with (output_path / f"{split}_sequences.json").open("w", encoding="utf-8") as handle:
            json.dump(records[split], handle, indent=2)
    with (output_path / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(records["feature_columns"], handle, indent=2)

