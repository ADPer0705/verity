# F3Net Detector

F3Net deepfake detector ported from DeepfakeBench.

## Setup (uv-managed)

```bash
cd detectors/f3net
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

Place `f3net_best.pth` in `weights/` (see `weights/README.md`).

## Explainability

Per-face fake probability + Grad-CAM attention overlays + facial region
energy analysis + prediction timeline + most-suspicious frames.
