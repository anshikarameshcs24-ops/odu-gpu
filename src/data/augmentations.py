"""Lightweight data augmentations for clinical video and audio."""

from __future__ import annotations

import random

import numpy as np


def augment_frames_and_audio(frames: np.ndarray, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.3 and len(frames) > 1:
        drop_index = random.randint(0, len(frames) - 1)
        frames = np.concatenate(
            [frames[:drop_index], frames[drop_index : drop_index + 1], frames[drop_index:-1]],
            axis=0,
        )

    if random.random() < 0.5:
        frames = frames[:, :, ::-1, :].copy()

    if random.random() < 0.4:
        factor = 1.0 + random.uniform(-0.1, 0.1)
        frames = np.clip(frames * factor, 0, 255)

    if random.random() < 0.3:
        noise_level = random.uniform(0.001, 0.005)
        audio = audio + np.random.randn(*audio.shape).astype(np.float32) * noise_level

    if random.random() < 0.4:
        scale = random.uniform(0.8, 1.2)
        audio = np.clip(audio * scale, -1.0, 1.0)

    return frames.astype(np.float32), audio.astype(np.float32)

