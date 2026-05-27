"""Preprocess raw videos into chunk-level frame and audio arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.audio_processor import extract_audio_segment
from src.data.video_processor import CHUNK_DURATION, CHUNK_OVERLAP, extract_keyframes_uniform, get_video_duration


def chunk_video(video_path: str, output_dir: str, video_id: str) -> list[dict[str, object]]:
    duration = get_video_duration(video_path)
    frames_dir = Path(output_dir) / "frames" / video_id
    audio_dir = Path(output_dir) / "audio" / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    step = CHUNK_DURATION - CHUNK_OVERLAP
    starts = np.arange(0, max(duration - CHUNK_DURATION + step, step), step)
    records: list[dict[str, object]] = []

    for index, start in enumerate(tqdm(starts, desc=f"Chunking {video_id}")):
        end = min(float(start + CHUNK_DURATION), duration)
        chunk_id = f"{video_id}_c{index:04d}"

        frames = extract_keyframes_uniform(video_path, float(start), end)
        frame_path = frames_dir / f"{chunk_id}.npy"
        np.save(frame_path, frames)

        audio = extract_audio_segment(video_path, float(start), end)
        audio_path = audio_dir / f"{chunk_id}.npy"
        np.save(audio_path, audio)

        records.append(
            {
                "video_id": video_id,
                "chunk_id": chunk_id,
                "start_sec": round(float(start), 2),
                "end_sec": round(float(end), 2),
                "frame_path": str(frame_path),
                "audio_path": str(audio_path),
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest", default="chunks_manifest.csv")
    args = parser.parse_args()

    all_records: list[dict[str, object]] = []
    for video_file in sorted(Path(args.input_dir).glob("**/*.mp4")):
        all_records.extend(chunk_video(str(video_file), args.output_dir, video_file.stem))

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / args.manifest
    pd.DataFrame(all_records).to_csv(manifest_path, index=False)
    print(f"Processed {len(all_records)} chunks -> {manifest_path}")


if __name__ == "__main__":
    main()

