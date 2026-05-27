# CMAI Multimodal AI System

GPU training scaffold for Cohen-Mansfield Agitation Inventory (CMAI) behaviour detection from video and audio. The project is structured around three stages:

1. Stage 1: modality-specific encoder adaptation
2. Stage 2: multimodal fusion fine-tuning
3. Stage 3: temporal agitation trajectory modelling

## Layout

- `configs/`: YAML configs for each training stage
- `src/data/`: preprocessing and dataset code
- `src/models/`: model components
- `src/training/`: training loops and losses
- `src/utils/`: labels, metrics, and config helpers
- `scripts/`: preprocessing, embedding extraction, and inference entrypoints
- `tests/`: small unit tests for core utilities

## External Data Sources Used

This scaffold now supports two external data sources you linked, but they are used for different purposes because they are not label-compatible:

- `TIHM-Dataset`: wearable and remote-monitoring tables for dementia care. This is useful for temporal agitation-risk forecasting, not raw audio-video training.
- `DAVE`: audio-video diagnostic benchmark. This is useful for multimodal encoder warm-up and evaluation, not for direct CMAI supervision.

The linked paper also informs the temporal and privacy-preserving design:

- it uses wearable biomarkers plus privacy-preserving pose/video signals for agitation and pre-agitation detection
- it reports pre-agitation patterns several minutes before escalation
- it uses sequential models over temporal windows rather than single independent frames

## Quick Start

1. Create the environment described in `requirements.txt`.
2. Place raw videos under `data/raw/`.
3. Prepare annotation CSVs under `data/annotations/`.
4. Preprocess videos:

```bash
python scripts/preprocess_videos.py --input_dir data/raw --output_dir data/processed
```

5. Train fusion model:

```bash
python -m src.training.trainer_stage2 --config configs/stage2_fusion.yaml
```

The default Stage 2 config in this scaffold is tuned for a 24 GB GPU with gradient accumulation:

- `batch_size: 2`
- `accumulation_steps: 4`
- `enable_gradient_checkpointing: true`
- `vision_backbone_lr: 2e-5`
- `audio_backbone_lr: 5e-6`
- `modality_dropout_prob: 0.15`
- `num_patch_tokens: 8`
- `num_audio_tokens: 8`

6. Extract frozen embeddings for temporal training:

```bash
python scripts/extract_embeddings.py \
  --model_ckpt checkpoints/stage2/best/model.pt \
  --manifest_csv data/processed/chunks_manifest.csv \
  --annotation_csv data/annotations/train.csv \
  --output_dir data/processed/embeddings
```

7. Train temporal model:

```bash
python -m src.training.trainer_stage3 --config configs/stage3_temporal.yaml
```

## Using TIHM

Prepare TIHM sensor sequences:

```bash
python scripts/prepare_tihm_dataset.py \
  --input_dir data/external/tihm \
  --output_dir data/processed/tihm
```

Train the TIHM sensor temporal branch:

```bash
python -m src.training.trainer_stage3_tihm --config configs/stage3_tihm_sensor.yaml
```

The TIHM branch is intentionally separate from the CMAI video+audio model. It trains its own risk and trajectory heads from sensor sequences rather than being forced into joint multimodal training with partially missing modalities.

## Using DAVE

Build a DAVE manifest for encoder evaluation or audio-video warm-up:

```bash
python scripts/prepare_dave_dataset.py \
  --split epic \
  --output_manifest data/processed/dave_epic.jsonl
```

Important: the DAVE dataset card states it is a diagnostic benchmark where both modalities are required and that it is not intended for large-scale model training. It also requires `datasets==3.6.0` with `trust_remote_code=True`.

## Notes

- The training scripts expect real CMAI annotations and processed chunk manifests.
- TIHM and DAVE do not provide native 29-class CMAI labels, so they are integrated as auxiliary data sources rather than dropped directly into the CMAI classifier.
- Stage 2 fusion now uses a small token set from each modality rather than single-vector cross-attention: the projected VideoMAE CLS token plus 8 patch tokens, and the projected Wav2Vec2 pooled embedding plus 8 temporally pooled audio tokens.
- W&B and Hugging Face are optional at runtime; the code degrades gracefully if they are not configured.
- The current scaffold is single-node and supports single GPU or `torch.nn.DataParallel`. You can extend it to FSDP/DeepSpeed using the provided config stubs.
