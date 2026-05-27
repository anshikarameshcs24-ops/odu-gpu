"""Helpers for working with the DAVE diagnostic audio-visual benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


@dataclass
class DAVERecord:
    video_with_overlayed_audio_path: str
    silent_video_path: str | None
    overlayed_audio_path: str | None
    audio_class: str
    choices: list[str]
    ground_truth_index: int
    split_name: str
    video_id: str | None = None
    participant_id: str | None = None
    sample_type: str | None = None


class DAVEDataset(Dataset):
    """A small manifest-based wrapper around DAVE samples."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        choice_metadata = record.get("choice_metadata", {})
        alignment = choice_metadata.get("audio_visual_alignment", {})
        return {
            "video_path": record.get("video_with_overlayed_audio_path"),
            "silent_video_path": record.get("silent_video_path"),
            "audio_path": record.get("overlayed_audio_path"),
            "audio_class": record.get("audio_class"),
            "choices": alignment.get("choices", []),
            "ground_truth_index": alignment.get("ground_truth"),
            "video_id": record.get("video_id"),
            "participant_id": record.get("participant_id"),
            "sample_type": record.get("type"),
        }

