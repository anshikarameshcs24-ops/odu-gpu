"""Chunk-level CMAI dataset and dataloader builders."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, VideoMAEImageProcessor

from src.data.augmentations import augment_frames_and_audio
from src.utils.cmai_labels import NUM_CMAI_CLASSES


def _parse_label_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(item) for item in json.loads(text)]


class CMAIDataset(Dataset):
    def __init__(self, annotation_df, video_processor, audio_processor, augment: bool = False):
        self.df = annotation_df.reset_index(drop=True)
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        frames = np.load(row["frame_path"]).astype(np.float32)
        audio = np.load(row["audio_path"]).astype(np.float32)

        if self.augment:
            frames, audio = augment_frames_and_audio(frames, audio)

        frame_list = [frames[index].astype(np.uint8) for index in range(len(frames))]
        video_inputs = self.video_processor(frame_list, return_tensors="pt")
        pixel_values = video_inputs["pixel_values"].squeeze(0)

        audio_inputs = self.audio_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=160000,
            truncation=True,
        )
        audio_values = audio_inputs["input_values"].squeeze(0)

        label_vector = torch.zeros(NUM_CMAI_CLASSES, dtype=torch.float32)
        for label_index in _parse_label_list(row.get("cmai_labels", [])):
            if 0 <= label_index < NUM_CMAI_CLASSES:
                label_vector[label_index] = 1.0

        risk_level = int(row.get("agitation_risk", 0))

        return {
            "pixel_values": pixel_values,
            "audio_values": audio_values,
            "labels": label_vector,
            "risk_level": torch.tensor(risk_level, dtype=torch.long),
            "chunk_id": row["chunk_id"],
        }


def build_processors(
    vision_ckpt: str = "MCG-NJU/videomae-base-finetuned-kinetics",
    audio_ckpt: str = "facebook/wav2vec2-large-robust",
):
    video_processor = VideoMAEImageProcessor.from_pretrained(vision_ckpt)
    audio_processor = AutoFeatureExtractor.from_pretrained(audio_ckpt)
    return video_processor, audio_processor


def build_dataloaders(
    train_df,
    val_df,
    batch_size: int = 4,
    num_workers: int = 8,
    vision_ckpt: str = "MCG-NJU/videomae-base-finetuned-kinetics",
    audio_ckpt: str = "facebook/wav2vec2-large-robust",
) -> tuple[DataLoader, DataLoader]:
    video_proc, audio_proc = build_processors(vision_ckpt=vision_ckpt, audio_ckpt=audio_ckpt)
    train_ds = CMAIDataset(train_df, video_proc, audio_proc, augment=True)
    val_ds = CMAIDataset(val_df, video_proc, audio_proc, augment=False)

    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(train_ds) >= batch_size,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader

