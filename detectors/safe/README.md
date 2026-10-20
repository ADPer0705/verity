# SAFE Detector

SAFE: "Improving Synthetic Image Detection Towards Generalization: An Image
Transformation Perspective" (Li et al., KDD 2025 ADS track,
[arXiv 2408.06741](https://arxiv.org/abs/2408.06741)), ported from the official
repository [Ouxiang-Li/SAFE](https://github.com/Ouxiang-Li/SAFE) (Apache-2.0).

The model (`models/resnet.py`) is a lightweight truncated ResNet-50
(1.45 M parameters: 3x3 stem conv, `layer1` + `layer2`, global average pool,
`fc1 = Linear(512, 2)`) that does not look at RGB pixels directly: inside
`forward` the input is transformed with a one-level bior1.3 discrete wavelet
transform (`pytorch_wavelets.DWTForward(J=1, mode="symmetric", wave="bior1.3")`)
and only the **HH** (diagonal high-frequency) sub-band, bilinearly resized back
to the input size, is classified. The head is 2-class cross-entropy with
label 1 = fake, so `softmax[:, 1]` is P(fake) directly (no logit stacking).

## Setup (uv-managed)

```bash
cd detectors/safe
uv sync
```

Dependencies beyond the common runtime: `pytorch_wavelets==1.3.0` (+
`PyWavelets`) for the exact upstream DWT. Upstream pins torch 2.2.1; the
environment uses 2.2.2.

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` is scanned for media files; artifacts (frames, timeline,
`result.json`) are written to `OUTPUT_DIR`.

## Input mode and preprocessing

`input_mode: full_frame` — SAFE is a whole-image synthetic-image detector, not
a face detector. Images are scored once; videos on up to `max_frames` (32)
evenly spaced frames.

The official evaluation (`scripts/eval.sh`: `--input_size 256
--transform_mode crop`) uses, from `data/datasets.py::Get_Transforms`:

```python
'crop': {'eval': [transforms.CenterCrop([256, 256])]}  # then transforms.ToTensor()
```

i.e. a 256x256 **center crop at native resolution, no resize and no mean/std
normalisation** (the paper's thesis is that resizing destroys the
high-frequency evidence SAFE relies on; the network sees raw [0, 1] pixels).
`model.preprocess` reproduces this including torchvision's zero padding for
images smaller than 256 px; `detector.yaml` carries neutral mean/std so the
engine never re-normalises. Consequence: on large frames only the central
256x256 window is examined per frame (exactly what the official evaluation
does on its benchmark images).

## Weights

`weights/checkpoint-best.pth` (5,840,638 bytes) is the official checkpoint
committed in the upstream repository (`checkpoint/checkpoint-best.pth`,
format `{"model": state_dict}`, 146 keys, loaded strictly; the DWT filter
buffers are rebuilt from `pywt` and are not part of the file):

```bash
curl -L -o weights/checkpoint-best.pth \
  'https://github.com/Ouxiang-Li/SAFE/raw/main/checkpoint/checkpoint-best.pth'
```

Trained on ForenSynths/ProGAN (car, cat, chair, horse) with SAFE's
augmentation recipe; see `weights/README.md` for checksum and citation.

## Explainability

`cam: false`: the classified signal is a wavelet sub-band, so a Grad-CAM over
`layer2` would localise wavelet energy rather than image content. Per-frame
scores and the timeline are produced for videos.
