"""Inference pipeline for chunk-level and temporal CMAI predictions."""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.cuda.amp import autocast
from transformers import AutoFeatureExtractor, VideoMAEImageProcessor

from src.data.audio_processor import extract_audio_segment
from src.data.video_processor import CHUNK_DURATION, CHUNK_OVERLAP, extract_keyframes_uniform, get_video_duration
from src.models.cmai_system import CMAISystem
from src.models.temporal import TemporalAgitationModel
from src.utils.cmai_labels import AGITATION_RISK_LEVELS, CMAI_BEHAVIOURS, EARLY_WARNING_BEHAVIOURS


@dataclass
class BehaviourDetection:
    cmai_item: str
    category: str
    confidence: float


@dataclass
class SegmentResult:
    segment_id: str
    start_time: float
    end_time: float
    detected_behaviours: List[BehaviourDetection]
    agitation_risk: str
    risk_trajectory: str
    early_warning_flags: List[str]
    raw_scores: dict


class CMAIInferencePipeline:
    CATEGORY_MAP = {
        index: category
        for category, indices in {
            "physically_non_aggressive": list(range(0, 9)),
            "verbally_non_aggressive": list(range(9, 15)),
            "physically_aggressive": list(range(15, 23)),
            "verbally_aggressive": list(range(23, 29)),
        }.items()
        for index in indices
    }

    def __init__(self, fusion_ckpt: str, temporal_ckpt: str, thresholds_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fusion_model = CMAISystem().to(self.device)
        self.fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=self.device))
        self.fusion_model.eval()

        self.temporal_model = TemporalAgitationModel().to(self.device)
        self.temporal_model.load_state_dict(torch.load(temporal_ckpt, map_location=self.device))
        self.temporal_model.eval()

        self.thresholds = np.load(thresholds_path)
        self.video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.audio_processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-robust")

    @torch.no_grad()
    def process_video(self, video_path: str) -> list[SegmentResult]:
        duration = get_video_duration(video_path)
        step = CHUNK_DURATION - CHUNK_OVERLAP
        starts = np.arange(0, max(duration - CHUNK_DURATION + step, step), step)
        embeddings = []
        metadata = []
        autocast_enabled = self.device.type == "cuda"

        for index, start in enumerate(starts):
            end = min(float(start + CHUNK_DURATION), duration)
            frames = extract_keyframes_uniform(video_path, float(start), end)
            audio = extract_audio_segment(video_path, float(start), end)

            frame_list = [frames[i].astype(np.uint8) for i in range(len(frames))]
            video_inputs = self.video_processor(frame_list, return_tensors="pt")
            audio_inputs = self.audio_processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                max_length=160000,
                truncation=True,
            )

            pixel_values = video_inputs["pixel_values"].to(self.device)
            audio_values = audio_inputs["input_values"].to(self.device)
            with autocast(enabled=autocast_enabled):
                _, _, fused = self.fusion_model(pixel_values, audio_values, return_embeddings=True)
            embeddings.append(fused.squeeze(0).cpu())
            metadata.append({"chunk_id": f"chunk_{index:04d}", "start_sec": float(start), "end_sec": float(end)})

        emb_seq = torch.stack(embeddings).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(embeddings)], device=self.device)
        with autocast(enabled=autocast_enabled):
            cmai_logits, risk_logits, traj_logits = self.temporal_model(emb_seq, lengths)

        cmai_scores = torch.sigmoid(cmai_logits[0]).cpu().numpy()
        risk_preds = risk_logits[0].argmax(dim=-1).cpu().numpy()
        traj_pred = traj_logits[0].argmax(dim=-1).item()
        traj_labels = {0: "stable", 1: "rising", 2: "peaking"}

        results: list[SegmentResult] = []
        for step_index, meta in enumerate(metadata):
            scores = cmai_scores[step_index]
            preds = scores >= self.thresholds
            detected = [
                BehaviourDetection(
                    cmai_item=CMAI_BEHAVIOURS[class_index],
                    category=self.CATEGORY_MAP[class_index],
                    confidence=float(scores[class_index]),
                )
                for class_index, is_active in enumerate(preds)
                if is_active
            ]

            early_warning_flags = [
                f"sub-threshold_{CMAI_BEHAVIOURS[class_index]}_{meta['start_sec']:.0f}s"
                for class_index in EARLY_WARNING_BEHAVIOURS
                if scores[class_index] > 0.3 and not preds[class_index]
            ]

            results.append(
                SegmentResult(
                    segment_id=meta["chunk_id"],
                    start_time=meta["start_sec"],
                    end_time=meta["end_sec"],
                    detected_behaviours=detected,
                    agitation_risk=AGITATION_RISK_LEVELS[int(risk_preds[step_index])],
                    risk_trajectory=traj_labels[traj_pred],
                    early_warning_flags=early_warning_flags,
                    raw_scores={CMAI_BEHAVIOURS[i]: float(scores[i]) for i in range(29)},
                )
            )
        return results

    def to_json(self, results: list[SegmentResult]) -> str:
        return json.dumps([dataclasses.asdict(item) for item in results], indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion_ckpt", required=True)
    parser.add_argument("--temporal_ckpt", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_json")
    args = parser.parse_args()

    pipeline = CMAIInferencePipeline(args.fusion_ckpt, args.temporal_ckpt, args.thresholds)
    results = pipeline.process_video(args.video_path)
    payload = pipeline.to_json(results)

    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()

