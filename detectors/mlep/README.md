# MLEP Detector

MLEP (NeurIPS 2025): Multi-granularity Local Entropy Patterns for Universal
AI-generated Image Detection (Li et al.), ported from the official repository
[fkeufss/MLEP](https://github.com/fkeufss/MLEP).

## Setup (uv-managed)

```bash
cd detectors/mlep
uv sync
```


## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` is scanned for media files; artifacts (faces, attention
maps, timeline, top frames, `result.json`) are written to `OUTPUT_DIR`.

## Weights

Place `model_epoch_best.pth` in `weights/` (see `weights/README.md`):

```bash
curl -L -o weights/model_epoch_best.pth \
  'https://raw.githubusercontent.com/fkeufss/MLEP/main/pretrained/model_epoch_best.pth'
```

Without it the model is randomly initialized and scores are not meaningful.

## Explainability

Per-face fake probability + Grad-CAM attention overlays + facial region
energy analysis + prediction timeline + most-suspicious frames.
