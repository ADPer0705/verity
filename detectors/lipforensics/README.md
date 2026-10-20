# LipForensics Detector

LipForensics ("Lips Don't Lie", Haliassos et al., CVPR 2021) deepfake
detector ported from the official repository (ahaliassos/LipForensics).
Instead of learning forgery artifacts in the whole face, it learns
high-level lip movements via a lipreading model (Conv3d frontend +
ResNet-18 trunk + multiscale multibranch temporal conv net) trained on
FaceForensics++ c23.

## Pipeline

1. The engine crops the face from every frame and resizes to 256x256
   (mean/std 0.5 round-trip).
2. FAN 68-point landmark detection runs on each face crop
   (`face_detector="none"` -- the crop is used as the detection box).
3. Landmarks are smoothed over a 12-frame sliding window and the frame is
   affine-warped (similarity transform) onto the LRW 256x256 mean face
   using the five stable points (33, 36, 39, 42, 45); the same transform
   is applied to the landmarks.
4. A 96x96 mouth region centred on the mean of mouth landmarks (48:68) is
   cropped, converted to grayscale, center-cropped to 88x88 and normalized
   with the official (0.421, 0.165) mean/std.
5. The 25-frame clip (1, 1, 25, 88, 88) is scored; the single logit is
   emitted as 2-class logits `[0, logit]` (index 1 = fake), so
   `softmax[:, 1] == sigmoid(logit)`.

If FAN returns no landmarks for any frame of a clip, the clip falls back
to a center 96x96 crop of the (un-warped) face crop.

## Setup (uv-managed)

```bash
cd detectors/lipforensics
uv sync
```


Face-alignment downloads the FAN weights (~150MB) on first run
(see `weights/README.md`).

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` is scanned for media files; artifacts (faces, timeline,
top frames, `result.json`) are written to `OUTPUT_DIR`.

## Weights

The official FF++ checkpoint is on Google Drive (~150MB with FAN weights
not included):

```bash
curl -L -o weights/lipforensics_ff.pth 'https://drive.usercontent.google.com/download?id=1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7&export=download&confirm=t'
```

## Explainability

Not available: `cam: false` (the detector emits no attention map).

## Computational cost

Per face clip: 25 FAN-68 landmark forward passes (one per frame) plus one
25-frame forward pass of the ~33M-parameter lipreading network. Landmark
detection dominates; expect roughly 1-4 seconds per clip on GPU.
