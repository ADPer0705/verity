# Effort Detector

Effort (Yan et al., ICML 2025): CLIP ViT-L/14 whose self-attention linears are
decomposed by SVD into a frozen principal subspace and a trainable rank-1
residual ("orthogonal subspace decomposition"), plus a 2-way head. Port of the
official `effort_detector.py` (YZY-stack/Effort-AIGI-Detection), loading the
released FaceForensics++ checkpoint with strict key matching.

## Setup (uv-managed)

```bash
cd detectors/effort
uv sync
```

## Weights

See `weights/README.md`: the checkpoint is downloaded from Google Drive into
`weights/`; the CLIP backbone (architecture only) is fetched from HuggingFace
at first run.

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
