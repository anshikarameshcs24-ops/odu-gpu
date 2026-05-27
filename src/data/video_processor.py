"""Video chunk extraction helpers."""

from __future__ import annotations

from pathlib import Path

import av
import cv2
import numpy as np

CHUNK_DURATION = 10
CHUNK_OVERLAP = 2
NUM_FRAMES_MODEL = 16
FRAME_SIZE = (224, 224)


def get_video_duration(video_path: str | Path) -> float:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    if stream.duration is None:
        duration = float(container.duration / av.time_base)
    else:
        duration = float(stream.duration * stream.time_base)
    container.close()
    return duration


def extract_keyframes_uniform(
    video_path: str | Path,
    start: float,
    end: float,
    n_frames: int = NUM_FRAMES_MODEL,
    frame_size: tuple[int, int] = FRAME_SIZE,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    timestamps = np.linspace(start, max(start, end - 1e-3), n_frames)
    frames: list[np.ndarray] = []

    for ts in timestamps:
        frame_index = max(int(ts * fps), 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)

    capture.release()

    while len(frames) < n_frames:
        pad = frames[-1] if frames else np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        frames.append(pad)

    return np.stack(frames[:n_frames]).astype(np.float32)

