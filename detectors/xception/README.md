# Xception Detector

Xception deepfake detector from FaceForensics++ (Rossler et al. 2019), ported
from DeepfakeBench.

## Setup (uv-managed)

```bash
cd detectors/xception
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

Place `xception_best.pth` in `weights/` (see `weights/README.md`). Without it
the model is randomly initialized and scores are not meaningful.

## Explainability

Per-face fake probability + Grad-CAM attention overlays + facial region
energy analysis + prediction timeline + most-suspicious frames.
