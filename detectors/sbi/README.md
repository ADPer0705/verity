# SBI Detector

SBI deepfake detector ported from DeepfakeBench.

## Setup (managed by uv)

```bash
cd detectors/sbi
uv sync
```

The per-detector environment is managed by `scripts/verity.py` (hardware-
adaptive: `verity.py setup --detector sbi` inside the container, or
`verity.py --profile cpu setup --detector sbi` natively).

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` is scanned for media files; artifacts (faces, attention
maps, timeline, top frames, `result.json`) are written to `OUTPUT_DIR`.

## Weights

No released weights exist; the model is randomly initialized (scores are NOT meaningful).

## Explainability

Per-face fake probability + Grad-CAM attention overlays + facial region
energy analysis + prediction timeline + most-suspicious frames.
