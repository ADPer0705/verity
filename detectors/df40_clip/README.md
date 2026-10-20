# DF40 CLIP Detector

CLIP ViT-L/14 vision tower fully fine-tuned for deepfake detection on the DF40
benchmark (Yan et al., "DF40: Toward Next-Generation Deepfake Detection",
NeurIPS 2024 Datasets & Benchmarks), plus a `Linear(1024, 2)` head. Port of
`DeepfakeBench_DF40/training/detectors/clip_detector.py` (YZY-stack/DF40),
loading the released `clip_large.pth` checkpoints with strict key matching.

## Setup (uv-managed)

```bash
cd detectors/df40_clip
uv sync
```

## Weights

See `weights/README.md`: the checkpoint is downloaded from Google Drive into
`weights/`; the CLIP backbone (architecture only) is fetched from HuggingFace
at first run. The default is the model trained on all DF40 families
(`train_on_df40`); per-family alternates (`train_on_fs`, `train_on_fr`,
`train_on_efs`) are selected through `weights.file` in `detector.yaml`.

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` is scanned for media files; artifacts (faces, attention
maps, timeline, top frames, `result.json`) are written to `OUTPUT_DIR`.

## Explainability

Per-face fake probability, last-layer CLS attention overlays (16x16 patch
grid), facial region energy, prediction timeline and most-suspicious frames.
