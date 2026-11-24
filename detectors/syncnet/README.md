# SyncNet Detector (lip-sync measurement)

"Out of time: automated lip sync in the wild" (Chung & Zisserman, ACCV 2016
workshop), ported from github.com/joonson/syncnet_python (MIT). SyncNet
embeds 5-frame mouth clips and 0.2 s MFCC windows into a joint space; sliding
the audio against the video gives the **AV offset**, the **minimum distance**
and the **confidence** (median minus minimum distance, the LSE-C metric used
to grade lip-sync generators such as Wav2Lip).

SyncNet is a *measurement*, not a trained deepfake classifier. Verity turns
each window's confidence into a fake probability with a fixed rule,
`p_fake = sigmoid(4 - confidence)` (confidence 6 -> 0.12, 4 -> 0.5,
2 -> 0.88), and stores the raw offset / distance / confidence per window in
`result.json` (`per_face[*].info`). Read those numbers as the evidence: real
talking heads typically score confidence 6-10 at offset 0, dubbed or
re-synthesised mouths fall below ~3-4 with arbitrary offsets. A constant
non-zero offset with high confidence means an editing delay, not a forgery.

## Setup (uv-managed)

```bash
cd detectors/syncnet
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

Videos with a soundtrack (`input_mode: av_clip`): 75-frame windows (3 s at
25 fps, up to 8 per video) are re-sampled to 25 fps, the face is detected
with MTCNN every 5th frame and cropped as upstream (`crop_scale` 0.4,
224x224 raw BGR), MFCCs are computed with python_speech_features, and the
offset/confidence are evaluated exactly as `SyncNetInstance.evaluate`.
`timeline.png` shows the per-window fake probability over the spectrogram.

## Weights

`weights/syncnet_v2.model` (MIT, VGG Oxford); see `weights/README.md`.
