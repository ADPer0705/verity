# FTCN Detector

FTCN deepfake detector ported from DeepfakeBench.

## Setup (uv-managed)

```bash
cd detectors/ftcn
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

Official FTCN+TT checkpoint `weights/ftcn_tt.pth` (yinglinzheng/FTCN GitHub
release, FF++ c23, no license stated; fetched from the authors' release
only). See `weights/README.md`. With the Kinetics I3D backbone instead, the
head is untrained (research only).

## Explainability

Per-face fake probability + Grad-CAM attention overlays + facial region
energy analysis + prediction timeline + most-suspicious frames.
