# Forensics Adapter Detector

"Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection"
(Cui, Li, Luo, Zhou, Dong, CVPR 2025), ported from
github.com/OUC-VAS/ForensicsAdapter (CC BY-NC 4.0). CLIP ViT-L/14 stays
frozen; a ViT-tiny adapter with 128 learnable queries learns to predict the
blending boundary of face forgeries and feeds an attention bias back into a
second pass of CLIP's blocks, so the frozen model attends to forgery traces.
Frame-level cross-dataset AUC at publication: CDF-v1 0.914, CDF-v2 0.900,
DFDC 0.843, DFDCP 0.890, DFD 0.933 (trained on FF++ c23 only).

## Setup (uv-managed)

```bash
cd detectors/forensics_adapter
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

256x256 MTCNN face crops with CLIP normalisation. Outputs per-face fake
probability, the predicted boundary map as attention overlay, region
energy, timeline and top frames.

## Weights

`weights/forensics_adapter.pth` (official, CC BY-NC 4.0, 963 MB incl. the
frozen fp16 CLIP ViT-L/14); see `weights/README.md`.
