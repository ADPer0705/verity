# SBI Detector

Self-Blended Images (Shiohara & Yamasaki, CVPR 2022 Oral), the official
inference architecture ported from github.com/mapooon/SelfBlendedImages:
EfficientNet-B4 with a 2-way head, trained on FaceForensics++ **real** faces
and self-blended pseudo-fakes only, which is why it generalises to unseen face
swaps (CDF 92.9 % / DFD 98.2 % / DFDCP 85.5 % AUC with the c23 checkpoint).

## Setup (managed by uv)

```bash
cd detectors/sbi
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` is scanned for media files; artifacts (faces, Grad-CAM
overlays, timeline, top frames, `result.json`) are written to `OUTPUT_DIR`.

## Weights

`weights/FFc23.tar` (official, FF++ c23); `FFraw.tar` is a drop-in alternate.
Research-only license, fetched from the authors' Google Drive and never
re-hosted. See `weights/README.md`.

## Preprocessing

The MTCNN box is enlarged by 25 % (`face_margin: 0.25`, upstream `w/4 * 0.5`
per side), resized to 380x380 and fed as raw [0, 1] RGB without mean/std
normalisation, matching `src/inference/inference_image.py`.

## Explainability

Per-face fake probability + Grad-CAM attention overlays (1792-channel
feature map) + facial region energy + prediction timeline + most-suspicious
frames.
