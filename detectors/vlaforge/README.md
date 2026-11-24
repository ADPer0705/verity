# VLAForge Detector

"Unleashing Vision-Language Semantics for Deepfake Video Detection" (Zhu,
Miao, Zhang, Deng, Pang, CVPR 2026), ported from github.com/mala-lab/VLAForge
(MIT). The method keeps CLIP's vision-language alignment intact and adds

- a **ForgePerceiver**: a small ViT with 128 learnable query tokens that
  predicts a blending-boundary map and biases CLIP's attention towards
  forgery-prone regions, and
- an **identity-aware VLA score**: the CLIP image embedding of the examined
  face is injected into the prompts "This is a real/fake photo of *id*
  person.", and the patch tokens are matched against both to form a
  per-patch fake map.

The final score is the average of the CLIP-query classifier and the
VLA-gated classifier. Reported frame-level AUCs (FF++ c23 training):
CDF-v2 0.9+, DFDC/DFDCP, DFD, plus DF40 full-face generators
(DiT, SiT, PixArt, StyleGAN-XL, VQGAN).

## Setup (uv-managed)

```bash
cd detectors/vlaforge
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

256x256 MTCNN face crops with CLIP normalisation. Outputs per-face fake
probability, the two branch probabilities (`aux`), the predicted boundary
map as attention overlay, region energy, timeline and top frames.

## Weights

`weights/vlaforge.pth` (official, MIT, 966 MB incl. the frozen fp16 CLIP
ViT-L/14); see `weights/README.md`. The CLIP tokenizer is fetched from
HuggingFace at first run.
