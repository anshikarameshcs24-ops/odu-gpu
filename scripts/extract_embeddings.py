"""Pre-extract chunk embeddings from a trained Stage 2 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.cmai_dataset import CMAIDataset, build_processors
from src.models.cmai_system import CMAISystem


@torch.no_grad()
def extract_embeddings(model_ckpt: str, manifest_csv: str, annotation_csv: str, output_dir: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMAISystem().to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    manifest_df = pd.read_csv(manifest_csv)
    annotation_df = pd.read_csv(annotation_csv)
    dataset_df = annotation_df.merge(manifest_df, on="chunk_id", how="inner")
    video_processor, audio_processor = build_processors()
    dataset = CMAIDataset(dataset_df, video_processor, audio_processor, augment=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    autocast_enabled = device.type == "cuda"

    for batch in tqdm(loader, desc="Extracting embeddings"):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        audio_values = batch["audio_values"].to(device, non_blocking=True)
        with autocast(enabled=autocast_enabled):
            _, _, embeddings = model(pixel_values, audio_values, return_embeddings=True)
        for index, chunk_id in enumerate(batch["chunk_id"]):
            np.save(output_path / f"{chunk_id}.npy", embeddings[index].detach().cpu().numpy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--annotation_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    extract_embeddings(args.model_ckpt, args.manifest_csv, args.annotation_csv, args.output_dir)


if __name__ == "__main__":
    main()

