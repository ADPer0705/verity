# D³ Detector

D³: Scaling Up Deepfake Detection by Learning from Discrepancy (Yang, Qian,
Zhu, Russakovsky, Wu, CVPR 2025). Port of github.com/BigAandSmallq/D3.

A frozen CLIP ViT-L/14 encodes the image **and** a copy whose 14x14 patches
were shuffled; the shuffled view destroys global semantics and keeps only
local generator artifacts, and an attention head over the two 1024-d
features decides. Trained on a multi-generator pool (ProGAN/UFD + GenImage)
for in-domain and out-of-domain AI-generated-image detection (90.7 % total
accuracy in the paper's scaled-up benchmark).

## Setup (managed by uv)

```bash
cd detectors/d3
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

Whole images / sampled whole frames (`input_mode: full_frame`), resized to
224x224 with CLIP normalisation; no face detection. Scores and a per-frame
timeline are written to `OUTPUT_DIR`.

## Weights

`weights/classifier.pth` is the official attention head (committed in the
authors' repo, no license stated; fetched from GitHub only). The CLIP tower
`openai/clip-vit-large-patch14` is fetched from HuggingFace at first run.
See `weights/README.md`.
