# GenD Detector

GenD ("Deepfake Detection that Generalizes Across Benchmarks", WACV 2026,
arXiv 2508.06248) ported from the official repository via HuggingFace
transformers.

## Setup (uv-managed)

```bash
cd detectors/gend
uv sync
```


Note: this detector pins torch 2.8.0 / transformers 4.56.2 — every detector
gets its own uv environment, so this does not affect the others.

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` is scanned for media files; artifacts (faces, attention
maps, timeline, top frames, `result.json`) are written to `OUTPUT_DIR`.

## Weights

Weights auto-download from HuggingFace (`yermandy/GenD_CLIP_L_14`) at first
run via transformers (external kind).

## Explainability

Per-face fake probability + Grad-CAM attention overlays (computed over the
ViT patch-token features) + facial region energy analysis + prediction
timeline + most-suspicious frames.
