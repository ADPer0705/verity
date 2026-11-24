# LAA-Net Detector

"LAA-Net: Localized Artifact Attention Network for Quality-Agnostic and
Generalizable Deepfake Detection" (Nguyen, Mejri, Singh, Kuleshova, Astrid,
Kacem, Ghorbel, Aouada, CVPR 2024), ported from github.com/10Ring/LAA-Net
(SnT academic license). An EfficientNet-B4 with an enhanced feature pyramid
decoder is trained, on real FaceForensics++ faces plus self-blended
pseudo-fakes only, with an explicit heatmap target on the blending boundary
("localized artifact attention") and a self-consistency branch. The
SBI-trained model reaches 95.4 % AUC on CDF-v2 and 86.9 % on DFDC.

## Setup (managed by uv)

```bash
cd detectors/laa_net
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

Face crops with a 25 % per-side margin, resized to 384x384, mean/std 0.5;
each face is scored together with its horizontal flip (upstream
`flip_test`).

## Weights

`weights/laa_net_sbi.pth` (official, SBI-trained); `laa_net_bi.pth`
(BI-trained) is a drop-in alternate. Academic-use license; fetched from
the authors' Dropbox folder only. See `weights/README.md`.

## Explainability

The model's own 96x96 localized-artifact heatmap (sigmoid, flip-averaged)
is the attention map: it marks the blending boundary the network believes
it found. Per-face probability, region energy, timeline and top frames as
usual.
