# RealForensics Detector

"Leveraging Real Talking Faces via Self-Supervision for Robust Forgery
Detection" (Haliassos, Mira, Petridis, Pantic, CVPR 2022), ported from
github.com/ahaliassos/RealForensics (MIT). A channel-separated 3D ResNet-101
first learns what *real* talking faces look like (BYOL-style audio-visual
self-supervision on LRW), then is fine-tuned on FaceForensics++ c23. The
representation generalises to unseen manipulations (CelebDF-v2 86.9 %,
FaceShifter 99.7 %, DeeperForensics 99.3 % video AUC) and survives common
corruptions.

## Pipeline

1. The engine crops the face (25 % margin) from 25 consecutive frames and
   resizes to 256x256.
2. FAN 68-point landmarks are detected on each frame, smoothed over a
   13-frame window and re-centred.
3. Each frame is warped onto the 256x256 LRW mean face with a similarity
   transform fitted on 8 stable points (28, 33, 36, 39, 42, 45, 48, 54),
   and a 150x150 patch centred on landmarks 15:68 is cut
   (`preprocessing/extract_faces.py`).
4. The patch clip is scaled to [0, 1], centre-cropped to 140, resized to
   112 and ImageNet-normalised (validation transform of
   `stage2/data/combined_dm.py`).
5. The `(1, 3, 25, 112, 112)` clip is scored; the single cosine-head logit is
   emitted as `[0, logit]` (index 1 = fake).

If FAN finds no landmarks for a frame of the clip, the clip falls back to a
centre 150x150 crop of the face.

## Setup (uv-managed)

```bash
cd detectors/realforensics
uv sync
```

Face-alignment downloads the FAN weights (~150 MB) on first run.

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

## Weights

`weights/realforensics_ff.pth` (official, MIT, Google Drive); see
`weights/README.md`.
