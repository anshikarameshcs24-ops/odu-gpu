"""Sequence dataset for temporal CMAI modelling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class CMAISequenceDataset(Dataset):
    def __init__(self, sequence_records: list[dict[str, Any]], max_seq_len: int = 30):
        self.records = sequence_records
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        embeddings = np.load(record["embedding_path"]).astype(np.float32)
        length = min(len(embeddings), self.max_seq_len)
        embeddings = embeddings[:length]

        cmai_targets = np.zeros((length, 29), dtype=np.float32)
        for step, labels in enumerate(record["cmai_labels"][:length]):
            for label in labels:
                cmai_targets[step, int(label)] = 1.0

        risk_levels = np.array(record["risk_levels"][:length], dtype=np.int64)

        return {
            "embeddings": torch.tensor(embeddings, dtype=torch.float32),
            "cmai_labels": torch.tensor(cmai_targets, dtype=torch.float32),
            "risk_levels": torch.tensor(risk_levels, dtype=torch.long),
            "trajectory": torch.tensor(record["trajectory"], dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }


def collate_sequences(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    lengths = torch.stack([item["length"] for item in batch])
    max_len = int(lengths.max().item())
    batch_size = len(batch)

    embeddings = torch.zeros(batch_size, max_len, 512, dtype=torch.float32)
    cmai_labels = torch.zeros(batch_size, max_len, 29, dtype=torch.float32)
    risk_levels = torch.zeros(batch_size, max_len, dtype=torch.long)

    for index, item in enumerate(batch):
        seq_len = int(item["length"].item())
        embeddings[index, :seq_len] = item["embeddings"]
        cmai_labels[index, :seq_len] = item["cmai_labels"]
        risk_levels[index, :seq_len] = item["risk_levels"]

    return {
        "embeddings": embeddings,
        "cmai_labels": cmai_labels,
        "risk_levels": risk_levels,
        "trajectory": torch.stack([item["trajectory"] for item in batch]),
        "lengths": lengths,
    }


def load_sequence_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)

