"""Audio extraction helpers."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

AUDIO_SR = 16000


def extract_audio_segment(video_path: str | Path, start: float, end: float, sr: int = AUDIO_SR) -> np.ndarray:
    audio, _ = librosa.load(str(video_path), sr=sr, offset=start, duration=end - start, mono=True)
    if audio.size == 0:
        audio = np.zeros(int((end - start) * sr), dtype=np.float32)
    return audio.astype(np.float32)

